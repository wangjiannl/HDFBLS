from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np


class ADHFBLSNumericalError(FloatingPointError):
    """Raised when the original ADHFBLS computation becomes non-finite."""


@dataclass(frozen=True)
class ADHFBLSConfig:
    """Model settings used by the official ADHFBLS implementation."""

    initial_rules: int = 2
    eta: int = 1
    task_type: int | str = 2
    shrinkage: float = 0.8
    regularization: float = 2.0 ** (-30)
    random_state: int = 200
    feature_chunk_size: int = 512

    def __post_init__(self) -> None:
        if self.initial_rules <= 0:
            raise ValueError("initial_rules must be positive.")
        if self.eta <= 0:
            raise ValueError("eta must be positive.")
        if self.shrinkage <= 0.0:
            raise ValueError("shrinkage must be positive.")
        if self.feature_chunk_size <= 0:
            raise ValueError("feature_chunk_size must be positive.")


class ADHFBLS:
    """
    Interval type-2 ADHFBLS baseline based on the supplied official code.

    The model-specific formulas and update order are retained:
      - random affine rule consequents with a bias term;
      - interval type-2 upper/lower membership functions;
      - fixed membership width 0.5;
      - upper and lower normalized firing strengths;
      - tanh enhancement nodes with shrinkage 0.8;
      - pseudoinverse output learning;
      - separate rule-node and enhancement-node increments.

    Numerical safeguards do not replace invalid values. A zero firing-strength
    denominator, NaN, Inf, or failed pseudoinverse raises
    ADHFBLSNumericalError so the experiment driver can record the fold and
    continue with subsequent folds and datasets.
    """

    def __init__(
        self,
        ruleNumber: int | None = None,
        eta: int | None = None,
        taskType: int | str | None = None,
        shrinkage: float = 0.8,
        regularization: float = 2.0 ** (-30),
        *,
        config: ADHFBLSConfig | None = None,
    ) -> None:
        if config is None:
            if ruleNumber is None:
                ruleNumber = 2
            if eta is None:
                eta = 1
            if taskType is None:
                taskType = 2
            config = ADHFBLSConfig(
                initial_rules=int(ruleNumber),
                eta=int(eta),
                task_type=taskType,
                shrinkage=float(shrinkage),
                regularization=float(regularization),
            )

        self.config = config
        self.ruleNumber = int(config.initial_rules)
        self.eta = int(config.eta)
        self.enhanceNodeNumber = self.ruleNumber * self.eta
        self.shrinkage = float(config.shrinkage)
        self.regularization = float(config.regularization)
        self.taskType = config.task_type
        self.feature_chunk_size = int(config.feature_chunk_size)
        self.rng = np.random.RandomState(config.random_state)

        self.centers: np.ndarray | None = None
        self.Alpha: np.ndarray | None = None
        self.enhanceWeight: np.ndarray | None = None
        self.enhanceShrinkage: float | None = None
        self.W: np.ndarray | None = None
        self.A_pinv: np.ndarray | None = None
        self.Z: np.ndarray | None = None
        self.A: np.ndarray | None = None
        self.sum_up: np.ndarray | None = None
        self.sum_down: np.ndarray | None = None

        self.ruleIncrementTime = 0
        self.enhanceIncrementTime = 0
        self.incrementList: list[int] = []
        self.AlphaIncrement: list[np.ndarray] = []
        self.centers_increment: list[np.ndarray] = []
        self.enhanceWeightIncrement_rule: list[np.ndarray] = []
        self.enhanceWeightIncrement_enhance: list[np.ndarray] = []
        self.addRuleNumber = 0
        self.addEnhanceNodeNumber = 0

        self.train_activation_diagnostics: dict[str, float] = {}
        self.last_activation_diagnostics: dict[str, float] = {}

    # ------------------------------------------------------------------
    # Utilities
    # ------------------------------------------------------------------
    def set_random_state(self, seed: int) -> None:
        """Reset only the RNG used by future incremental candidates."""
        self.rng = np.random.RandomState(int(seed))

    @staticmethod
    def _require_finite(name: str, values: np.ndarray) -> None:
        if not np.all(np.isfinite(values)):
            raise ADHFBLSNumericalError(f"{name} contains NaN or Inf.")

    @staticmethod
    def _pinv(values: np.ndarray, name: str) -> np.ndarray:
        try:
            result = np.linalg.pinv(values)
        except np.linalg.LinAlgError as exc:
            raise ADHFBLSNumericalError(
                f"Pseudoinverse failed for {name}: {exc}"
            ) from exc
        ADHFBLS._require_finite(f"pinv({name})", result)
        return result

    def _new_centers(self, n_rules: int, n_features: int) -> np.ndarray:
        c_down = self.rng.rand(n_rules, n_features) * 0.5
        c_up = self.rng.rand(n_rules, n_features) * 0.5 + 0.5
        return np.stack((c_down, c_up), axis=2)

    def effectCal(self, output: np.ndarray, train_y: np.ndarray) -> float:
        task = self.taskType
        if task == "regression" or task == 1:
            return float(np.sqrt(np.mean((output - train_y) ** 2)))
        if task == "classification" or task == 2:
            pred = np.argmax(output, axis=1)
            true = np.argmax(train_y, axis=1)
            return float(np.mean(pred == true))
        raise ValueError(f"Unsupported task type: {task!r}")

    def _firing_strengths(
        self,
        X: np.ndarray,
        centers: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
        """Compute the official upper/lower interval type-2 products."""
        X = np.asarray(X, dtype=np.float64)
        centers = np.asarray(centers, dtype=np.float64)
        self._require_finite("X", X)
        self._require_finite("centers", centers)

        n_samples, n_features = X.shape
        n_rules = centers.shape[0]
        if centers.shape != (n_rules, n_features, 2):
            raise ValueError(
                "centers must have shape (n_rules, n_features, 2)."
            )

        fs_up = np.ones((n_samples, n_rules), dtype=np.float64)
        fs_down = np.ones((n_samples, n_rules), dtype=np.float64)
        std = 0.5
        denominator = 2.0 * std * std

        with np.errstate(over="raise", invalid="raise", under="ignore"):
            for rule in range(n_rules):
                c_down_all = centers[rule, :, 0]
                c_up_all = centers[rule, :, 1]

                for start in range(0, n_features, self.feature_chunk_size):
                    stop = min(start + self.feature_chunk_size, n_features)
                    x_chunk = X[:, start:stop]
                    c_down = c_down_all[start:stop][None, :]
                    c_up = c_up_all[start:stop][None, :]
                    midpoint = 0.5 * (c_down + c_up)

                    upper = np.ones_like(x_chunk)
                    below = x_chunk < c_down
                    above = x_chunk >= c_up
                    upper[below] = np.exp(
                        -((x_chunk[below] - np.broadcast_to(c_down, x_chunk.shape)[below]) ** 2)
                        / denominator
                    )
                    upper[above] = np.exp(
                        -((x_chunk[above] - np.broadcast_to(c_up, x_chunk.shape)[above]) ** 2)
                        / denominator
                    )

                    lower = np.where(
                        x_chunk < midpoint,
                        np.exp(-((x_chunk - c_up) ** 2) / denominator),
                        np.exp(-((x_chunk - c_down) ** 2) / denominator),
                    )

                    fs_up[:, rule] *= np.prod(upper, axis=1)
                    fs_down[:, rule] *= np.prod(lower, axis=1)

        self._require_finite("upper raw firing strengths", fs_up)
        self._require_finite("lower raw firing strengths", fs_down)

        combined = np.hstack((fs_up, fs_down))
        diagnostics = {
            "zero_count": int(np.count_nonzero(combined == 0.0)),
            "element_count": int(combined.size),
            "row_has_positive": np.any(combined > 0.0, axis=1),
        }
        return fs_up, fs_down, diagnostics

    @staticmethod
    def _combine_diagnostics(
        blocks: list[dict[str, Any]],
    ) -> dict[str, float]:
        if not blocks:
            return {
                "raw_zero_ratio_percent": np.nan,
                "raw_all_zero_row_ratio_percent": np.nan,
            }
        zero_count = sum(int(block["zero_count"]) for block in blocks)
        element_count = sum(int(block["element_count"]) for block in blocks)
        row_has_positive = np.zeros_like(
            np.asarray(blocks[0]["row_has_positive"], dtype=bool)
        )
        for block in blocks:
            row_has_positive |= np.asarray(block["row_has_positive"], dtype=bool)
        return {
            "raw_zero_ratio_percent": 100.0 * zero_count / element_count,
            "raw_all_zero_row_ratio_percent": float(
                100.0 * np.mean(~row_has_positive)
            ),
        }

    @staticmethod
    def _normalize(
        raw: np.ndarray,
        denominator_values: np.ndarray,
        name: str,
    ) -> np.ndarray:
        denominator_values = np.asarray(denominator_values, dtype=np.float64)
        if np.any(~np.isfinite(denominator_values)):
            raise ADHFBLSNumericalError(
                f"{name} normalization denominator contains NaN or Inf."
            )
        if np.any(denominator_values == 0.0):
            count = int(np.count_nonzero(denominator_values == 0.0))
            raise ADHFBLSNumericalError(
                f"{name} normalization has {count} zero denominators "
                "caused by numerical underflow."
            )
        result = raw / denominator_values[:, None]
        ADHFBLS._require_finite(f"normalized {name} firing strengths", result)
        return result

    # ------------------------------------------------------------------
    # Initial training
    # ------------------------------------------------------------------
    def train(self, train_x: np.ndarray, train_y: np.ndarray) -> float:
        train_x = np.asarray(train_x, dtype=np.float64)
        train_y = np.asarray(train_y, dtype=np.float64)
        self._require_finite("train_x", train_x)
        self._require_finite("train_y", train_y)

        n_samples, n_features = train_x.shape
        self.Alpha = self.rng.rand(n_features + 1, self.ruleNumber)
        train_x_bias = np.hstack((train_x, np.ones((n_samples, 1))))
        consequent = train_x_bias @ self.Alpha

        self.centers = self._new_centers(self.ruleNumber, n_features)
        raw_up, raw_down, diag = self._firing_strengths(
            train_x, self.centers
        )
        self.sum_up = raw_up.sum(axis=1)
        self.sum_down = raw_down.sum(axis=1)
        fs_up = self._normalize(raw_up, self.sum_up, "upper")
        fs_down = self._normalize(raw_down, self.sum_down, "lower")

        a_up = fs_up * consequent
        a_down = fs_down * consequent
        self.Z = np.hstack((a_up, a_down))
        z_bias = np.hstack((self.Z, np.full((n_samples, 1), 0.1)))

        self.enhanceWeight = self.rng.rand(
            2 * self.ruleNumber + 1,
            self.enhanceNodeNumber,
        )
        h_linear = z_bias @ self.enhanceWeight
        max_value = float(np.max(h_linear))
        if not np.isfinite(max_value) or max_value == 0.0:
            raise ADHFBLSNumericalError(
                "Invalid enhancement shrinkage denominator."
            )
        self.enhanceShrinkage = self.shrinkage / max_value
        H = np.tanh(h_linear * self.enhanceShrinkage)
        self.A = np.hstack((self.Z, H))
        self._require_finite("initial design matrix", self.A)

        self.A_pinv = self._pinv(self.A, "initial design matrix")
        self.W = self.A_pinv @ train_y
        self._require_finite("initial output weights", self.W)

        self.train_activation_diagnostics = self._combine_diagnostics([diag])
        output = self.A @ self.W
        self._require_finite("initial training output", output)
        return self.effectCal(output, train_y)

    # ------------------------------------------------------------------
    # Full forward reconstruction for arbitrary data
    # ------------------------------------------------------------------
    def _transform_full(
        self,
        X: np.ndarray,
    ) -> tuple[np.ndarray, dict[str, float]]:
        if any(
            value is None
            for value in (
                self.Alpha,
                self.centers,
                self.enhanceWeight,
                self.enhanceShrinkage,
                self.W,
            )
        ):
            raise RuntimeError("The model has not been trained.")

        X = np.asarray(X, dtype=np.float64)
        self._require_finite("X", X)
        n_samples = X.shape[0]
        x_bias = np.hstack((X, np.ones((n_samples, 1))))

        consequent = x_bias @ self.Alpha
        raw_up, raw_down, base_diag = self._firing_strengths(X, self.centers)
        sum_up = raw_up.sum(axis=1)
        sum_down = raw_down.sum(axis=1)
        fs_up = self._normalize(raw_up, sum_up, "upper")
        fs_down = self._normalize(raw_down, sum_down, "lower")
        Z = np.hstack((fs_up * consequent, fs_down * consequent))

        z_bias = np.hstack((Z, np.full((n_samples, 1), 0.1)))
        H = np.tanh((z_bias @ self.enhanceWeight) * self.enhanceShrinkage)

        increment_data_Z = [Z]
        increment_data_H = [H]
        increment_data_H_enhance: list[np.ndarray] = []
        total_rules = self.ruleNumber
        rule_index = 0
        enhance_index = 0
        diag_blocks = [base_diag]

        for increment_type in self.incrementList:
            if increment_type == 0:
                z_bias = np.hstack((Z, np.full((n_samples, 1), 0.1)))
                weight = self.enhanceWeightIncrement_enhance[enhance_index]
                increment_data_H_enhance.append(
                    np.tanh((z_bias @ weight) * self.enhanceShrinkage)
                )
                enhance_index += 1
                continue

            alpha = self.AlphaIncrement[rule_index]
            centers = self.centers_increment[rule_index]
            n_new_rules = alpha.shape[1]
            consequent_new = x_bias @ alpha
            raw_up_new, raw_down_new, new_diag = self._firing_strengths(
                X, centers
            )
            diag_blocks.append(new_diag)

            sum_up = sum_up + raw_up_new.sum(axis=1)
            sum_down = sum_down + raw_down_new.sum(axis=1)
            fs_up_new = self._normalize(raw_up_new, sum_up, "upper")
            fs_down_new = self._normalize(raw_down_new, sum_down, "lower")
            z_temp = np.hstack(
                (fs_up_new * consequent_new, fs_down_new * consequent_new)
            )

            scale = total_rules / (total_rules + n_new_rules)
            for j in range(len(increment_data_Z)):
                increment_data_Z[j] *= scale
                increment_data_H[j] *= scale
            for j in range(len(increment_data_H_enhance)):
                increment_data_H_enhance[j] *= scale

            increment_data_Z.append(z_temp)
            z_bias = np.hstack((Z, np.full((n_samples, 1), 0.1)))
            weight = self.enhanceWeightIncrement_rule[rule_index]
            increment_data_H.append(
                np.tanh((z_bias @ weight) * self.enhanceShrinkage) * scale
            )
            Z = np.hstack((Z * scale, z_temp))
            total_rules += n_new_rules
            rule_index += 1

        A = np.hstack((increment_data_Z[0], increment_data_H[0]))
        rule_index = 1
        enhance_index = 0
        for increment_type in self.incrementList:
            if increment_type == 0:
                A = np.hstack((A, increment_data_H_enhance[enhance_index]))
                enhance_index += 1
            else:
                A = np.hstack(
                    (A, increment_data_Z[rule_index], increment_data_H[rule_index])
                )
                rule_index += 1

        self._require_finite("reconstructed design matrix", A)
        return A, self._combine_diagnostics(diag_blocks)

    def predict_scores(self, X: np.ndarray) -> np.ndarray:
        if self.W is None:
            raise RuntimeError("The model has not been trained.")
        A, diagnostics = self._transform_full(X)
        scores = A @ self.W
        self._require_finite("prediction scores", scores)
        self.last_activation_diagnostics = diagnostics
        return scores

    def predict(self, X: np.ndarray) -> np.ndarray:
        return np.argmax(self.predict_scores(X), axis=1)

    def test(self, test_x: np.ndarray, test_y: np.ndarray) -> float:
        scores = self.predict_scores(test_x)
        return self.effectCal(scores, np.asarray(test_y, dtype=np.float64))

    def refit_output(self, X: np.ndarray, Y: np.ndarray) -> float:
        """Refit only the linear output weights for the selected structure."""
        A, diagnostics = self._transform_full(X)
        self.A = A
        self.A_pinv = self._pinv(A, "refit design matrix")
        self.W = self.A_pinv @ np.asarray(Y, dtype=np.float64)
        self._require_finite("refitted output weights", self.W)
        self.train_activation_diagnostics = diagnostics
        output = A @ self.W
        self._require_finite("refit training output", output)
        return self.effectCal(output, np.asarray(Y, dtype=np.float64))

    # ------------------------------------------------------------------
    # Incremental learning (same update order as supplied code)
    # ------------------------------------------------------------------
    def enhanceNodeIncrement(
        self,
        incrementNumber: int,
        train_y: np.ndarray,
    ) -> float:
        if self.Z is None or self.A is None or self.A_pinv is None:
            raise RuntimeError("The model has not been trained.")

        n_new_nodes = int(incrementNumber) * self.eta
        self.incrementList.append(0)
        self.addEnhanceNodeNumber += n_new_nodes
        weight = self.rng.rand(self.Z.shape[1] + 1, n_new_nodes)
        self.enhanceWeightIncrement_enhance.append(weight)

        z_bias = np.hstack((self.Z, np.full((self.Z.shape[0], 1), 0.1)))
        A_increment = np.tanh(
            (z_bias @ weight) * float(self.enhanceShrinkage)
        )
        self._greville_append(A_increment)
        self.enhanceIncrementTime += 1

        self.W = self.A_pinv @ np.asarray(train_y, dtype=np.float64)
        self._require_finite("enhancement-increment output weights", self.W)
        output = self.A @ self.W
        return self.effectCal(output, np.asarray(train_y, dtype=np.float64))

    def ruleIncrement(
        self,
        incrementNumber: int,
        train_x: np.ndarray,
        train_y: np.ndarray,
    ) -> float:
        if any(
            value is None
            for value in (self.Z, self.A, self.A_pinv, self.sum_up, self.sum_down)
        ):
            raise RuntimeError("The model has not been trained.")

        incrementNumber = int(incrementNumber)
        train_x = np.asarray(train_x, dtype=np.float64)
        train_y = np.asarray(train_y, dtype=np.float64)
        n_samples, n_features = train_x.shape
        self.incrementList.append(1)

        alpha = self.rng.rand(n_features + 1, incrementNumber)
        centers = self._new_centers(incrementNumber, n_features)
        self.AlphaIncrement.append(alpha)
        self.centers_increment.append(centers)

        x_bias = np.hstack((train_x, np.ones((n_samples, 1))))
        consequent = x_bias @ alpha
        raw_up, raw_down, _ = self._firing_strengths(train_x, centers)
        self.sum_up = self.sum_up + raw_up.sum(axis=1)
        self.sum_down = self.sum_down + raw_down.sum(axis=1)
        fs_up = self._normalize(raw_up, self.sum_up, "upper")
        fs_down = self._normalize(raw_down, self.sum_down, "lower")

        old_rules = self.ruleNumber + self.addRuleNumber
        new_rules = old_rules + incrementNumber
        inverse_scale = new_rules / old_rules
        scale = old_rules / new_rules

        a_up_increment = fs_up * consequent * inverse_scale
        a_down_increment = fs_down * consequent * inverse_scale

        n_new_enhance = self.eta * incrementNumber
        weight = self.rng.rand(self.Z.shape[1] + 1, n_new_enhance)
        self.enhanceWeightIncrement_rule.append(weight)
        z_bias = np.hstack((self.Z, np.full((n_samples, 1), 0.1)))
        enhancement_increment = np.tanh(
            (z_bias @ weight) * float(self.enhanceShrinkage)
        )

        A_increment = np.hstack(
            (a_up_increment, a_down_increment, enhancement_increment)
        )
        self._greville_append(A_increment, post_scale=inverse_scale)
        self.A *= scale

        self.addEnhanceNodeNumber += n_new_enhance
        self.Z = np.hstack(
            (self.Z, a_up_increment, a_down_increment)
        ) * scale
        self.ruleIncrementTime += 1
        self.addRuleNumber += incrementNumber

        self.W = self.A_pinv @ train_y
        self._require_finite("rule-increment output weights", self.W)
        output = self.A @ self.W
        return self.effectCal(output, train_y)

    def _greville_append(
        self,
        A_increment: np.ndarray,
        post_scale: float = 1.0,
    ) -> None:
        if self.A is None or self.A_pinv is None:
            raise RuntimeError("The model has not been trained.")
        self._require_finite("incremental design block", A_increment)

        D = self.A_pinv @ A_increment
        C = A_increment - self.A @ D

        # This is the intended Greville zero-residual branch. The supplied
        # implementation used C.all()==0, which does not test whether C is
        # the zero matrix. Using allclose follows the stated algorithm.
        if np.allclose(C, 0.0, rtol=1e-12, atol=1e-14):
            middle = np.eye(D.shape[1]) - D.T @ D
            B = self._pinv(middle, "Greville middle matrix") @ D.T @ self.A_pinv
        else:
            B = self._pinv(C, "Greville residual")

        self.A_pinv = np.vstack((self.A_pinv - D @ B, B)) * post_scale
        self.A = np.hstack((self.A, A_increment))
        self._require_finite("updated pseudoinverse", self.A_pinv)
        self._require_finite("updated design matrix", self.A)

    def structure(self) -> dict[str, int]:
        final_rules = self.ruleNumber + self.addRuleNumber
        final_enhance = self.enhanceNodeNumber + self.addEnhanceNodeNumber
        return {
            "rules": int(final_rules),
            "upper_feature_nodes": int(final_rules),
            "lower_feature_nodes": int(final_rules),
            "feature_nodes": int(2 * final_rules),
            "enhancement_nodes": int(final_enhance),
            "total_nodes": int(2 * final_rules + final_enhance),
            "rule_increments": int(self.ruleIncrementTime),
            "enhancement_increments": int(self.enhanceIncrementTime),
        }
