from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np


class CFBLSNumericalError(FloatingPointError):
    """Raised when the original CFBLS computation becomes numerically invalid."""


@dataclass(frozen=True)
class CFBLSConfig:
    """Configuration following Feng et al. (IEEE TFS, 2021)."""

    n_rules: int
    n_fuzzy_sets: int
    n_enhancement_nodes: int
    random_state: int = 2026
    feature_chunk_size: int = 4096

    def __post_init__(self) -> None:
        if self.n_rules <= 0:
            raise ValueError("n_rules must be positive.")
        if self.n_fuzzy_sets < 2:
            raise ValueError("n_fuzzy_sets must be at least 2.")
        if self.n_enhancement_nodes <= 0:
            raise ValueError("n_enhancement_nodes must be positive.")
        if self.feature_chunk_size <= 0:
            raise ValueError("feature_chunk_size must be positive.")


class CFBLS:
    r"""
    Compact Fuzzy Broad Learning System (CFBLS).

    This implementation follows Algorithm 1 and Eqs. (16)--(21) of:
    S. Feng, C. L. P. Chen, L. Xu, and Z. Liu,
    "On the Accuracy-Complexity Tradeoff of Fuzzy Broad Learning System,"
    IEEE Transactions on Fuzzy Systems, 2021.

    Model-specific choices retained from the paper:
      * one first-order TSK fuzzy system;
      * equally spaced fuzzy-set centers in [0, 1];
      * widths sampled from (0, 1);
      * binary feature-selection matrix S;
      * one-hot rule-combination matrix R;
      * consequent coefficients, enhancement weights, and biases sampled
        uniformly from [0, 1];
      * Gaussian MF exp(-((x-c)/sigma)^2), with no factor 1/2;
      * tanh enhancement transformation inherited from FBLS;
      * output weights calculated by the Moore-Penrose pseudoinverse.

    No epsilon is inserted into the firing-strength denominator. If the
    original formulation produces an all-zero denominator or another
    non-finite quantity, CFBLSNumericalError is raised so the experiment
    driver can record that fold as a numerical failure and continue.
    """

    def __init__(self, input_dim: int, config: CFBLSConfig) -> None:
        if input_dim <= 0:
            raise ValueError("input_dim must be positive.")

        self.input_dim = int(input_dim)
        self.config = config
        self.rng = np.random.default_rng(config.random_state)

        self.n_rules = config.n_rules
        self.n_fuzzy_sets = config.n_fuzzy_sets
        self.n_enhancement_nodes = config.n_enhancement_nodes

        # The paper uses G partitions and G+1 fuzzy sets. Therefore,
        # n_fuzzy_sets = G+1 and c_g = (g-1)/G.
        self.centers = np.linspace(0.0, 1.0, self.n_fuzzy_sets)

        # Uniform sampling on the open interval (0, 1).
        positive_zero = np.nextafter(0.0, 1.0)
        self.widths = self.rng.uniform(
            low=positive_zero,
            high=1.0,
            size=self.n_fuzzy_sets,
        )

        # S(t,k) in {0,1}: whether feature t is used by rule k.
        self.S = self.rng.integers(
            0,
            2,
            size=(self.input_dim, self.n_rules),
            dtype=np.int8,
        )

        # R(t,g,k) is one-hot over g. Storing the selected fuzzy-set index
        # is exactly equivalent to storing the full one-hot tensor and is
        # much more memory efficient for very high-dimensional datasets.
        self.R_index = self.rng.integers(
            0,
            self.n_fuzzy_sets,
            size=(self.input_dim, self.n_rules),
            dtype=np.int16,
        )

        # p_k in Eq. (17), sampled uniformly from [0,1].
        self.alpha = self.rng.random((self.input_dim, self.n_rules))

        # W_e and b in Eq. (20), sampled uniformly from [0,1].
        self.W_e = self.rng.random(
            (self.n_rules, self.n_enhancement_nodes)
        )
        self.b = self.rng.random(self.n_enhancement_nodes)

        self.W: np.ndarray | None = None
        self.last_activation_diagnostics: dict[str, float] = {}
        self.train_activation_diagnostics: dict[str, float] = {}
        self.test_activation_diagnostics: dict[str, float] = {}

    @staticmethod
    def _check_matrix(name: str, values: np.ndarray) -> None:
        if not np.all(np.isfinite(values)):
            raise CFBLSNumericalError(
                f"{name} contains NaN or Inf values."
            )

    def _raw_firing_strength(self, X: np.ndarray) -> np.ndarray:
        """Compute the numerator of Eq. (18) in the ordinary domain."""
        X = np.asarray(X, dtype=np.float64)
        if X.ndim != 2 or X.shape[1] != self.input_dim:
            raise ValueError(
                f"X must have shape (N, {self.input_dim}), got {X.shape}."
            )
        self._check_matrix("X", X)

        n_samples = X.shape[0]
        raw = np.ones((n_samples, self.n_rules), dtype=np.float64)
        chunk_size = self.config.feature_chunk_size

        # Eq. (18): ignored features contribute a multiplicative factor 1;
        # selected features contribute the selected Gaussian membership.
        for rule_idx in range(self.n_rules):
            feature_indices = np.flatnonzero(self.S[:, rule_idx])
            if feature_indices.size == 0:
                continue

            selected_sets = self.R_index[feature_indices, rule_idx]

            for start in range(0, feature_indices.size, chunk_size):
                stop = min(start + chunk_size, feature_indices.size)
                idx = feature_indices[start:stop]
                set_idx = selected_sets[start:stop]

                centers = self.centers[set_idx]
                widths = self.widths[set_idx]

                with np.errstate(
                    over="raise",
                    divide="raise",
                    invalid="raise",
                    under="ignore",
                ):
                    scaled_distance = (X[:, idx] - centers) / widths
                    membership = np.exp(-(scaled_distance ** 2))
                    raw[:, rule_idx] *= np.prod(membership, axis=1)

        self._check_matrix("raw firing-strength matrix", raw)
        return raw

    def compute_activation(self, X: np.ndarray) -> np.ndarray:
        """Compute normalized firing strengths exactly as in Eq. (18)."""
        raw = self._raw_firing_strength(X)
        denominator = np.sum(raw, axis=1, keepdims=True)

        zero_ratio = float(np.mean(raw == 0.0) * 100.0)
        all_zero_rows = np.all(raw == 0.0, axis=1)
        all_zero_ratio = float(np.mean(all_zero_rows) * 100.0)
        min_positive = (
            float(np.min(raw[raw > 0.0]))
            if np.any(raw > 0.0)
            else np.nan
        )
        max_value = float(np.max(raw)) if raw.size else np.nan

        self.last_activation_diagnostics = {
            "raw_zero_ratio_percent": zero_ratio,
            "raw_all_zero_row_ratio_percent": all_zero_ratio,
            "raw_min_positive": min_positive,
            "raw_max": max_value,
        }

        if np.any(denominator == 0.0):
            raise CFBLSNumericalError(
                "All-zero firing-strength row detected before normalization."
            )
        self._check_matrix("firing-strength denominator", denominator)

        with np.errstate(
            over="raise",
            divide="raise",
            invalid="raise",
            under="ignore",
        ):
            activation = raw / denominator

        self._check_matrix("normalized firing-strength matrix", activation)
        return activation

    def fuzzy_rule(self, X: np.ndarray) -> np.ndarray:
        """Compute first-order consequents z_sk = p_k x_s^T."""
        consequent = np.asarray(X, dtype=np.float64) @ self.alpha
        self._check_matrix("TSK consequent matrix", consequent)
        return consequent

    def compute_Z(self, X: np.ndarray) -> np.ndarray:
        """Compute the intermediate matrix Z in Eq. (17)."""
        activation = self.compute_activation(X)
        consequent = self.fuzzy_rule(X)
        Z = activation * consequent
        self._check_matrix("intermediate matrix Z", Z)
        return Z

    def compute_H(self, Z: np.ndarray) -> np.ndarray:
        """Compute enhancement output H = tanh(Z W_e + b)."""
        H = np.tanh(Z @ self.W_e + self.b)
        self._check_matrix("enhancement matrix H", H)
        return H

    def feature_matrix(self, X: np.ndarray) -> np.ndarray:
        """Compute Theta = [Z, H] in Eq. (21)."""
        Z = self.compute_Z(X)
        H = self.compute_H(Z)
        theta = np.concatenate((Z, H), axis=1)
        self._check_matrix("extended feature matrix Theta", theta)
        return theta

    def fit(self, X: np.ndarray, Y: np.ndarray) -> "CFBLS":
        """Fit W = Theta^+ Y using the Moore-Penrose pseudoinverse."""
        X = np.asarray(X, dtype=np.float64)
        Y = np.asarray(Y, dtype=np.float64)
        if Y.ndim != 2 or Y.shape[0] != X.shape[0]:
            raise ValueError("Y must be a 2-D matrix aligned with X.")
        self._check_matrix("Y", Y)

        theta = self.feature_matrix(X)
        self.train_activation_diagnostics = dict(
            self.last_activation_diagnostics
        )

        try:
            self.W = np.linalg.pinv(theta) @ Y
        except np.linalg.LinAlgError as exc:
            raise CFBLSNumericalError(
                f"Pseudoinverse failed: {exc}"
            ) from exc

        self._check_matrix("output weight matrix W", self.W)
        return self

    def predict_scores(self, X: np.ndarray) -> np.ndarray:
        if self.W is None:
            raise RuntimeError("The model must be fitted before prediction.")

        theta = self.feature_matrix(X)
        self.test_activation_diagnostics = dict(
            self.last_activation_diagnostics
        )
        scores = theta @ self.W
        self._check_matrix("prediction score matrix", scores)
        return scores

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict class indices by the largest output entry."""
        return np.argmax(self.predict_scores(X), axis=1)

    def get_rule_combination_tensor(self) -> np.ndarray:
        """Materialize the one-hot R tensor only when explicitly needed."""
        R = np.zeros(
            (self.input_dim, self.n_fuzzy_sets, self.n_rules),
            dtype=np.int8,
        )
        feature_idx = np.arange(self.input_dim)[:, None]
        rule_idx = np.arange(self.n_rules)[None, :]
        R[feature_idx, self.R_index, rule_idx] = 1
        return R

    def get_diagnostics(self) -> dict[str, Any]:
        return {
            "train": dict(self.train_activation_diagnostics),
            "test": dict(self.test_activation_diagnostics),
        }
