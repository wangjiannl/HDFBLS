from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, List

import numpy as np


@dataclass
class HDFBLSFeatureMapConfig:
    """Configuration of the ARGS-based feature mapping layer."""
    max_rules: int = 60      # Maximum number of fuzzy rules
    xi: float = 743.0        # Conservative double-precision safety threshold
    lb: float = 0.0          # Lower bound after min-max normalization
    ub: float = 1.0          # Upper bound after min-max normalization


class HDFBLSFeatureMap:
    """
    Feature mapping layer of HDFBLS.

    This module implements the adaptive rule generation scheme (ARGS).
    It constructs a single zero-order TSK fuzzy subsystem and outputs the
    Gaussian firing-strength matrix F.

    Input assumption:
        X should be normalized to [lb, ub] before calling train_F/test_F.
    """

    def __init__(self, config: Optional[HDFBLSFeatureMapConfig] = None):
        self.cfg = config or HDFBLSFeatureMapConfig()

        self.centers_: Optional[np.ndarray] = None     # (R, D)
        self.sigmas_: Optional[np.ndarray] = None      # (R,)
        self.F_: Optional[np.ndarray] = None           # (N, R)
        self.n_centers_: int = 0

        self.W_out_: Optional[np.ndarray] = None
        self.Z_pinv_: Optional[np.ndarray] = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def train_F(
        self,
        X: np.ndarray,
        Y_train: np.ndarray,
        eval_every: int = 5,
        patience: int = 5,
        min_rules: int = 5,
    ) -> np.ndarray:
        """
        Train the ARGS feature mapping layer.

        Parameters
        ----------
        X : array, shape (N, D)
            Training samples normalized to [lb, ub].
        Y_train : array, shape (N, C)
            One-hot encoded training labels.
        eval_every : int
            Evaluate training accuracy every `eval_every` generated rules.
        patience : int
            Early stopping patience based on training accuracy.
        min_rules : int
            Minimum number of rules before early stopping is activated.

        Returns
        -------
        F : array, shape (N, R)
            Training firing-strength matrix.
        """
        X = np.asarray(X, dtype=np.float64)
        Y = np.asarray(Y_train, dtype=np.float64)

        N, D = X.shape
        lb, ub = self.cfg.lb, self.cfg.ub
        data_range = ub - lb

        # ------------------------------------------------------------------
        # Width lower bound.
        # Manuscript formula:
        #   sigma_min = sqrt((ub-lb)^2 D / (8 xi)).
        # For normalized data [0,1], this becomes sqrt(D / (8 xi)).
        # ------------------------------------------------------------------
        sigma_min = np.sqrt((data_range ** 2) * D / (8.0 * self.cfg.xi))

        X2 = np.einsum("ij,ij->i", X, X)

        F_raw = np.zeros((N, self.cfg.max_rules), dtype=np.float64)
        centers: List[np.ndarray] = []
        sigmas: List[float] = []

        # Residual sample weights used by ARGS.
        sample_weight = np.ones(N, dtype=np.float64)
        selected_mask = np.zeros(N, dtype=bool)
        x_mean = X.mean(axis=0)

        Z = None
        Z_pinv = None
        W_out = None

        best_acc = -np.inf
        best_snapshot = None
        bad_count = 0

        for rule_id in range(self.cfg.max_rules):
            candidate_indices = np.where(~selected_mask)[0]
            if candidate_indices.size == 0:
                break

            # --------------------------------------------------------------
            # Rule-center selection.
            # --------------------------------------------------------------
            if rule_id == 0:
                d2_to_mean = self._squared_distance_to_center(X2, X, x_mean)
                local_id = int(np.argmin(d2_to_mean[candidate_indices]))
            else:
                local_id = int(np.argmax(sample_weight[candidate_indices]))

            sample_id = int(candidate_indices[local_id])
            c_raw = X[sample_id].copy()

            # --------------------------------------------------------------
            # Width estimation.
            # --------------------------------------------------------------
            d2_raw = self._squared_distance_to_center(X2, X, c_raw)
            pool_indices = candidate_indices[candidate_indices != sample_id]

            if pool_indices.size == 0:
                sigma = float(sigma_min)
            else:
                d2_pool = d2_raw[pool_indices]
                w_pool = sample_weight[pool_indices]
                w_sum = float(np.sum(w_pool))

                if (not np.isfinite(w_sum)) or (w_sum <= 0.0):
                    d2_bar = float(np.mean(d2_pool)) if d2_pool.size > 0 else 0.0
                else:
                    d2_bar = float(np.dot(w_pool, d2_pool) / w_sum)

                sigma = float(sigma_min + np.sqrt(d2_bar / 2.0))

            # --------------------------------------------------------------
            # Feasible center interval.
            # Manuscript formula:
            #   delta = sqrt(2 sigma^2 xi / D),
            #   ub - delta <= c_d <= lb + delta.
            # --------------------------------------------------------------
            delta = np.sqrt(2.0 * (sigma ** 2) * self.cfg.xi / D)
            lower_c = ub - delta
            upper_c = lb + delta

            tol = 1e-12
            if lower_c > upper_c + tol:
                raise ValueError(
                    "Empty feasible center interval. "
                    "Please check sigma_min, xi, and the input normalization range."
                )
            center = np.clip(c_raw, lower_c, upper_c)

            # --------------------------------------------------------------
            # Gaussian firing strength.
            # Manuscript formula:
            #   f_r(x_i) = exp(-||x_i-c_r||^2 / (2 sigma_r^2)).
            # --------------------------------------------------------------
            d2 = self._squared_distance_to_center(X2, X, center)
            firing = np.exp(-d2 / (2.0 * sigma ** 2))
            firing_col = firing.reshape(-1, 1)

            # --------------------------------------------------------------
            # Output-layer update by Greville incremental pseudoinverse.
            # --------------------------------------------------------------
            Z, Z_pinv, W_out = self._greville_append(
                Z=Z,
                Y=Y,
                E=firing_col,
                Z_pinv=Z_pinv,
                W=W_out,
            )

            F_raw[:, rule_id] = firing
            centers.append(center)
            sigmas.append(sigma)

            # --------------------------------------------------------------
            # Residual sample-weight update.
            # A sample strongly covered by the current rule receives a smaller
            # residual weight in later rule generation.
            # --------------------------------------------------------------
            sample_weight *= (1.0 - firing)
            selected_mask[sample_id] = True

            current_rule_num = rule_id + 1

            # --------------------------------------------------------------
            # ARGS stopping decision based on training accuracy.
            # The test data are not used in this process.
            # --------------------------------------------------------------
            if (current_rule_num >= min_rules) and (current_rule_num % eval_every == 0):
                train_acc = self.predict(Z, W_out, y_true=Y)

                if train_acc > best_acc:
                    best_acc = train_acc
                    bad_count = 0
                    best_snapshot = {
                        "centers": np.vstack(centers).copy(),
                        "sigmas": np.asarray(sigmas, dtype=np.float64).copy(),
                        "n_rules": current_rule_num,
                        "Z_pinv": Z_pinv.copy(),
                        "W_out": W_out.copy(),
                    }
                else:
                    bad_count += 1
                    if bad_count >= patience:
                        break

        # ------------------------------------------------------------------
        # Restore the best rule set selected by the training-based criterion.
        # ------------------------------------------------------------------
        if best_snapshot is not None:
            self.centers_ = best_snapshot["centers"]
            self.sigmas_ = best_snapshot["sigmas"]
            self.n_centers_ = int(best_snapshot["n_rules"])
            self.F_ = F_raw[:, : self.n_centers_]
            self.Z_pinv_ = best_snapshot["Z_pinv"]
            self.W_out_ = best_snapshot["W_out"]
        else:
            self.centers_ = np.vstack(centers) if centers else np.zeros((0, D), dtype=np.float64)
            self.sigmas_ = np.asarray(sigmas, dtype=np.float64)
            self.n_centers_ = len(sigmas)
            self.F_ = F_raw[:, : self.n_centers_]
            self.Z_pinv_ = Z_pinv
            self.W_out_ = W_out

        return self.F_

    def test_F(self, X: np.ndarray) -> np.ndarray:
        """
        Compute firing-strength features for new samples using the fitted
        centers and widths.

        Parameters
        ----------
        X : array, shape (N, D)
            Test samples normalized by the same training-set scaler.

        Returns
        -------
        F : array, shape (N, R)
            Firing-strength matrix.
        """
        if self.centers_ is None or self.sigmas_ is None:
            raise RuntimeError("Call train_F(X, Y_train) before test_F(X).")

        X = np.asarray(X, dtype=np.float64)
        if X.ndim != 2:
            raise ValueError(f"X must be a 2-D array, but got shape {X.shape}.")
        if X.shape[1] != self.centers_.shape[1]:
            raise ValueError(
                f"Input dimension mismatch: X has {X.shape[1]} features, "
                f"but centers have {self.centers_.shape[1]} features."
            )

        if self.centers_.shape[0] == 0:
            return np.zeros((X.shape[0], 0), dtype=np.float64)

        return self._compute_firing_matrix(X, self.centers_, self.sigmas_)

    def predict(
        self,
        F: np.ndarray,
        W: np.ndarray,
        y_true: Optional[np.ndarray] = None,
    ):
        """
        Predict labels or compute accuracy.

        If y_true is None, return predicted class indices.
        If y_true is provided, return classification accuracy.
        """
        F = np.asarray(F, dtype=np.float64)
        W = np.asarray(W, dtype=np.float64)

        scores = F @ W
        y_pred = np.argmax(scores, axis=1)

        if y_true is None:
            return y_pred

        y = self._labels_from_target(y_true)

        if y.shape[0] != F.shape[0]:
            raise ValueError(f"Length mismatch: y_true has {y.shape[0]} rows, F has {F.shape[0]} rows.")

        return float(np.mean(y_pred == y))

    # ------------------------------------------------------------------
    # Internal utilities
    # ------------------------------------------------------------------
    @staticmethod
    def _labels_from_target(y_true: np.ndarray) -> np.ndarray:
        y = np.asarray(y_true)

        if y.ndim == 2 and y.shape[1] > 1:
            return np.argmax(y, axis=1)

        y = y.reshape(-1)
        if np.array_equal(np.unique(y), np.array([-1, 1])):
            return (y > 0).astype(int)

        return y.astype(int)

    @staticmethod
    def _squared_distance_to_center(
        X2: np.ndarray,
        X: np.ndarray,
        center: np.ndarray,
    ) -> np.ndarray:
        d2 = X2 - 2.0 * (X @ center) + float(np.dot(center, center))
        return np.maximum(d2, 0.0)

    @staticmethod
    def _compute_firing_matrix(
        X: np.ndarray,
        centers: np.ndarray,
        sigmas: np.ndarray,
    ) -> np.ndarray:
        X2 = np.einsum("ij,ij->i", X, X)
        C2 = np.einsum("ij,ij->i", centers, centers)
        XC = X @ centers.T
        D2 = X2[:, None] - 2.0 * XC + C2[None, :]
        D2 = np.maximum(D2, 0.0)

        denom = 2.0 * (sigmas[None, :] ** 2)
        return np.exp(-D2 / denom)

    @staticmethod
    def _greville_append(
        Z: Optional[np.ndarray],
        Y: np.ndarray,
        E: np.ndarray,
        Z_pinv: Optional[np.ndarray] = None,
        W: Optional[np.ndarray] = None,
    ):
        """
        Incrementally update the pseudoinverse and output weights after
        appending a new feature block E to Z.
        """
        Y = np.asarray(Y, dtype=np.float64)
        E = np.asarray(E, dtype=np.float64)

        if Z is None or Z.size == 0:
            Z_new = E.copy()
            Z_pinv_new = np.linalg.pinv(Z_new)
            W_new = Z_pinv_new @ Y
            return Z_new, Z_pinv_new, W_new

        Z = np.asarray(Z, dtype=np.float64)

        if Z_pinv is None:
            Z_pinv = np.linalg.pinv(Z)
        if W is None:
            W = Z_pinv @ Y

        D = Z_pinv @ E
        C = E - Z @ D

        if np.linalg.matrix_rank(C) > 0:
            B_T = np.linalg.pinv(C)
        else:
            K = D.T @ D
            K.flat[:: K.shape[0] + 1] += 1.0
            RHS = D.T @ Z_pinv
            try:
                B_T = np.linalg.solve(K, RHS)
            except np.linalg.LinAlgError:
                B_T = np.linalg.pinv(K) @ RHS

        Z_pinv_new = np.vstack([Z_pinv - D @ B_T, B_T])

        BY = B_T @ Y
        W_new = np.vstack([W - D @ BY, BY])

        Z_new = np.concatenate([Z, E], axis=1)
        return Z_new, Z_pinv_new, W_new
