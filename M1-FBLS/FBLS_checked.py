from __future__ import annotations

import os
os.environ.setdefault("OMP_NUM_THREADS", "1")

from dataclasses import dataclass

import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import accuracy_score


@dataclass(frozen=True)
class FBLSConfig:
    """Configuration of the original fuzzy broad learning system."""

    n_subsystems: int
    n_rules: int
    n_enhancement_nodes: int
    random_state: int = 2026
    pinv_rcond: float = 1e-12


class FBLS:
    """
    Fuzzy Broad Learning System following Feng and Chen (2020).

    For each fuzzy subsystem, the implementation uses:
      1. first-order TSK consequents with coefficients sampled from U[0, 1];
      2. Gaussian membership functions with sigma = 1;
      3. k-means centers obtained from the training data;
      4. normalized product firing strengths;
      5. enhancement weights and biases sampled from U[0, 1]; and
      6. output weights solved by the Moore--Penrose pseudoinverse.

    The matrix sent to the output layer is [Z^n, H^m]. This is the same
    quantity represented as [D Omega, H^m] in Eq. (18) of the paper,
    because Z^n contains the weighted rule outputs defined in Eqs. (11)--(13).
    """

    def __init__(
        self,
        fuzz_sys: int,
        fuzz_rule: int,
        enhance_node: int,
        random_state: int = 2026,
        pinv_rcond: float = 1e-12,
    ) -> None:
        self.config = FBLSConfig(
            n_subsystems=int(fuzz_sys),
            n_rules=int(fuzz_rule),
            n_enhancement_nodes=int(enhance_node),
            random_state=int(random_state),
            pinv_rcond=float(pinv_rcond),
        )

        if self.config.n_subsystems <= 0:
            raise ValueError("fuzz_sys must be positive.")
        if self.config.n_rules <= 0:
            raise ValueError("fuzz_rule must be positive.")
        if self.config.n_enhancement_nodes <= 0:
            raise ValueError("enhance_node must be positive.")

        self.alpha: list[np.ndarray] = []
        self.centrals: list[np.ndarray] = []
        self.Wh: np.ndarray | None = None
        self.W: np.ndarray | None = None
        self.features: np.ndarray | None = None
        self.n_classes_: int | None = None

    @property
    def fuzz_sys(self) -> int:
        return self.config.n_subsystems

    @property
    def fuzz_rule(self) -> int:
        return self.config.n_rules

    @property
    def enhance_node(self) -> int:
        return self.config.n_enhancement_nodes

    def _check_X(self, X: np.ndarray) -> np.ndarray:
        X = np.asarray(X, dtype=np.float64)
        if X.ndim != 2:
            raise ValueError("X must be a two-dimensional array.")
        if not np.all(np.isfinite(X)):
            raise ValueError("X contains NaN or infinite values.")
        return X

    def _get_centers(
        self,
        X: np.ndarray,
        random_state: int,
    ) -> np.ndarray:
        if X.shape[0] < self.fuzz_rule:
            raise ValueError(
                f"n_rules={self.fuzz_rule} exceeds the number of training "
                f"samples ({X.shape[0]})."
            )

        # The paper uses k-means to obtain rule centers. A different seed is
        # used for each subsystem so that their centers can differ.
        kmeans = KMeans(
            n_clusters=self.fuzz_rule,
            init="k-means++",
            n_init=1,
            random_state=random_state,
        )
        kmeans.fit(X)
        return np.asarray(kmeans.cluster_centers_, dtype=np.float64)

    @staticmethod
    def _rule_activation(
        X: np.ndarray,
        centers: np.ndarray,
    ) -> np.ndarray:
        """
        Compute Eqs. (8)--(10) with sigma = 1.

        No epsilon is added to the denominator because Eq. (9) in the
        original method uses direct normalization. If an all-zero row occurs,
        the fold is reported as a numerical failure by the experiment code.
        """
        distances = X[:, None, :] - centers[None, :, :]
        memberships = np.exp(-(distances ** 2))
        firing = np.prod(memberships, axis=2)
        denominator = np.sum(firing, axis=1, keepdims=True)

        if np.any(denominator == 0.0):
            n_failed = int(np.sum(denominator[:, 0] == 0.0))
            raise FloatingPointError(
                "FBLS produced all-zero firing strengths for "
                f"{n_failed} sample(s)."
            )

        normalized = firing / denominator
        if not np.all(np.isfinite(normalized)):
            raise FloatingPointError(
                "FBLS produced non-finite normalized firing strengths."
            )
        return normalized

    @staticmethod
    def _first_order_consequent(
        X: np.ndarray,
        alpha_i: np.ndarray,
    ) -> np.ndarray:
        # Eq. (7): z_sk^i = sum_t alpha_kt^i x_st.
        return X @ alpha_i

    def _initialize_parameters(self, X: np.ndarray) -> None:
        _, n_features = X.shape
        rng = np.random.default_rng(self.config.random_state)

        mapped_dim = self.fuzz_sys * self.fuzz_rule

        # Eq. (14): enhancement weights and biases are sampled from [0, 1].
        self.Wh = rng.uniform(
            0.0,
            1.0,
            size=(mapped_dim + 1, self.enhance_node),
        )

        self.alpha = []
        self.centrals = []
        for _ in range(self.fuzz_sys):
            # Eq. (7) and Algorithm 1: alpha is initialized from U[0, 1].
            self.alpha.append(
                rng.uniform(0.0, 1.0, size=(n_features, self.fuzz_rule))
            )
            subsystem_seed = int(rng.integers(0, np.iinfo(np.int32).max))
            self.centrals.append(
                self._get_centers(X, random_state=subsystem_seed)
            )

    def _compute_Zn(self, X: np.ndarray) -> np.ndarray:
        mapped = np.empty(
            (X.shape[0], self.fuzz_sys * self.fuzz_rule),
            dtype=np.float64,
        )

        for subsystem in range(self.fuzz_sys):
            omega = self._rule_activation(X, self.centrals[subsystem])
            consequent = self._first_order_consequent(
                X,
                self.alpha[subsystem],
            )
            start = subsystem * self.fuzz_rule
            stop = start + self.fuzz_rule
            mapped[:, start:stop] = omega * consequent

        return mapped

    def _compute_H(self, Zn: np.ndarray) -> np.ndarray:
        if self.Wh is None:
            raise RuntimeError("The model has not been initialized.")
        bias = np.ones((Zn.shape[0], 1), dtype=np.float64)
        H_input = np.hstack((Zn, bias))
        return np.tanh(H_input @ self.Wh)

    def _build_features(self, X: np.ndarray) -> np.ndarray:
        Zn = self._compute_Zn(X)
        H = self._compute_H(Zn)
        features = np.hstack((Zn, H))
        if not np.all(np.isfinite(features)):
            raise FloatingPointError("FBLS feature matrix contains NaN or Inf.")
        return features

    def train(self, X: np.ndarray, Y_train: np.ndarray) -> float:
        X = self._check_X(X)
        Y_train = np.asarray(Y_train, dtype=np.float64)

        if Y_train.ndim != 2 or Y_train.shape[0] != X.shape[0]:
            raise ValueError("Y_train must be a two-dimensional one-hot matrix.")
        if not np.all(np.isfinite(Y_train)):
            raise ValueError("Y_train contains NaN or infinite values.")

        self.n_classes_ = int(Y_train.shape[1])
        self._initialize_parameters(X)
        self.features = self._build_features(X)

        # Eq. (19): W = [Z^n, H^m]^+ Y.
        self.W = np.linalg.pinv(self.features) @ Y_train

        if not np.all(np.isfinite(self.W)):
            raise FloatingPointError("FBLS output weights contain NaN or Inf.")

        y_true = np.argmax(Y_train, axis=1)
        y_pred = np.argmax(self.features @ self.W, axis=1)
        return float(accuracy_score(y_true, y_pred))

    def predict_scores(self, X: np.ndarray) -> np.ndarray:
        if self.W is None:
            raise RuntimeError("The model must be trained before prediction.")
        X = self._check_X(X)
        self.features = self._build_features(X)
        scores = self.features @ self.W
        if not np.all(np.isfinite(scores)):
            raise FloatingPointError("FBLS predictions contain NaN or Inf.")
        return scores

    def predict(self, X: np.ndarray) -> np.ndarray:
        return np.argmax(self.predict_scores(X), axis=1)

    def test(self, X: np.ndarray, Y_test: np.ndarray) -> float:
        Y_test = np.asarray(Y_test)
        if Y_test.ndim != 2:
            raise ValueError("Y_test must be a two-dimensional one-hot matrix.")
        y_true = np.argmax(Y_test, axis=1)
        y_pred = self.predict(X)
        return float(accuracy_score(y_true, y_pred))
