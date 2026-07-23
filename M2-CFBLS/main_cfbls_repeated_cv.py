from __future__ import annotations

import csv
from collections import Counter
from itertools import product
from pathlib import Path
from time import perf_counter

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
)
from sklearn.model_selection import RepeatedStratifiedKFold, train_test_split
from sklearn.preprocessing import LabelEncoder

from CFBLS_checked import CFBLS, CFBLSConfig


# ============================================================
# Experimental protocol
# ============================================================
OUTER_SPLITS = 5
OUTER_REPEATS = 5
VALIDATION_RATIO = 0.20
RANDOM_SEED = 2026

DATASET_NAMES = [
    "PAGEB",          # D1
    "Thyroid",        # D2
    "SATELLITE",      # D3
    "TEXTURE",        # D4
    "spambase",       # D5
    "ORL",            # D6
    "semg",           # D7
    "warpAR10P (1)",  # D8
    "PIE",            # D9
    "handoutlines",   # D10
    "Giesste",        # D11
    "Drivace",        # D12
    "Leukemia",       # D13
    "CMHS",           # D14
    "GSAFM",          # D15
]

FUZZY_SET_LIST = [3, 5, 7]
RULE_LIST = [10, 20, 30]
ENHANCEMENT_LIST = [20, 40, 60, 80, 100]

SCRIPT_DIR = Path(__file__).resolve().parent


def locate_data_dir() -> Path:
    """Locate data placed beside the server main program."""
    candidates = [
        SCRIPT_DIR / "datasets_npz",
        SCRIPT_DIR,
        SCRIPT_DIR.parent / "datasets_npz",
    ]
    for candidate in candidates:
        if any((candidate / f"{name}.npz").exists() for name in DATASET_NAMES):
            return candidate
    return candidates[0]


DATA_DIR = locate_data_dir()
RESULT_DIR = SCRIPT_DIR / "results_cfbls_holdout_cv"
VALIDATION_FILE = RESULT_DIR / "cfbls_validation_settings.csv"
SELECTED_FILE = RESULT_DIR / "cfbls_selected_settings.csv"
DETAIL_FILE = RESULT_DIR / "cfbls_outer_cv_details.csv"
SUMMARY_FILE = RESULT_DIR / "cfbls_holdout_cv_summary.csv"
DATASET_TIME_FILE = RESULT_DIR / "cfbls_dataset_times.csv"


# ============================================================
# Utilities
# ============================================================
def minmax_fit(X_train: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    x_min = np.min(X_train, axis=0, keepdims=True)
    x_max = np.max(X_train, axis=0, keepdims=True)
    x_range = x_max - x_min
    x_range[x_range == 0.0] = 1.0
    return x_min, x_range


def minmax_transform(
    X: np.ndarray,
    x_min: np.ndarray,
    x_range: np.ndarray,
) -> np.ndarray:
    return np.clip((X - x_min) / x_range, 0.0, 1.0)


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def finite_mean_std(values: list[float]) -> tuple[float, float]:
    array = np.asarray(values, dtype=np.float64)
    array = array[np.isfinite(array)]
    if array.size == 0:
        return np.nan, np.nan
    mean = float(np.mean(array))
    std = float(np.std(array, ddof=1)) if array.size > 1 else 0.0
    return mean, std


def repeated_cv_mean_std(values: list[float]) -> tuple[float, float]:
    """Average outer folds within each repetition, then summarize repeats."""
    array = np.asarray(values, dtype=np.float64)
    expected = OUTER_SPLITS * OUTER_REPEATS
    if array.size != expected:
        raise ValueError(f"Expected {expected} values, received {array.size}.")
    if not np.all(np.isfinite(array)):
        return np.nan, np.nan
    repetition_means = np.mean(
        array.reshape(OUTER_REPEATS, OUTER_SPLITS),
        axis=1,
    )
    return finite_mean_std(repetition_means.tolist())


def repeated_cv_mean(values: list[float]) -> float:
    return repeated_cv_mean_std(values)[0]


def mode_int(values: list[int]) -> int:
    counts = Counter(values)
    return min(counts, key=lambda value: (-counts[value], value))


def safe_diag(diag: dict[str, float], key: str) -> float:
    value = diag.get(key, np.nan)
    return float(value) if value is not None else np.nan


def make_validation_split(
    y: np.ndarray,
    outer_split: int,
) -> tuple[np.ndarray, np.ndarray]:
    indices = np.arange(y.size)
    fit_idx, validation_idx = train_test_split(
        indices,
        test_size=VALIDATION_RATIO,
        shuffle=True,
        stratify=y,
        random_state=RANDOM_SEED + outer_split,
    )
    return fit_idx, validation_idx


def evaluate_configuration(
    X_train_raw: np.ndarray,
    y_train: np.ndarray,
    X_eval_raw: np.ndarray,
    y_eval: np.ndarray,
    n_classes: int,
    n_sets: int,
    n_rules: int,
    n_enhance: int,
    model_seed: int,
) -> dict:
    """Fit one CFBLS configuration and evaluate it once."""
    x_min, x_range = minmax_fit(X_train_raw)
    X_train = minmax_transform(X_train_raw, x_min, x_range)
    X_eval = minmax_transform(X_eval_raw, x_min, x_range)
    Y_train = np.eye(n_classes, dtype=np.float64)[y_train]

    model = CFBLS(
        input_dim=X_train.shape[1],
        config=CFBLSConfig(
            n_rules=n_rules,
            n_fuzzy_sets=n_sets,
            n_enhancement_nodes=n_enhance,
            random_state=model_seed,
        ),
    )

    fit_start = perf_counter()
    model.fit(X_train, Y_train)
    fit_time = perf_counter() - fit_start
    train_diag = dict(model.train_activation_diagnostics)

    predict_start = perf_counter()
    y_pred = model.predict(X_eval)
    predict_time = perf_counter() - predict_start
    eval_diag = dict(model.test_activation_diagnostics)

    accuracy = float(accuracy_score(y_eval, y_pred))
    balanced_accuracy = float(balanced_accuracy_score(y_eval, y_pred))
    macro_f1 = float(
        f1_score(y_eval, y_pred, average="macro", zero_division=0)
    )
    if not np.all(
        np.isfinite([accuracy, balanced_accuracy, macro_f1])
    ):
        raise FloatingPointError("Non-finite evaluation metric detected.")

    return {
        "accuracy": accuracy,
        "balanced_accuracy": balanced_accuracy,
        "macro_f1": macro_f1,
        "fit_time": fit_time,
        "predict_time": predict_time,
        "train_diag": train_diag,
        "eval_diag": eval_diag,
    }


# ============================================================
# Main experiment
# ============================================================
def main() -> None:
    RESULT_DIR.mkdir(parents=True, exist_ok=True)
    parameter_grid = list(
        product(FUZZY_SET_LIST, RULE_LIST, ENHANCEMENT_LIST)
    )

    validation_rows: list[dict] = []
    selected_rows: list[dict] = []
    detail_rows: list[dict] = []
    summary_rows: list[dict] = []
    dataset_time_rows: list[dict] = []

    for dataset_index, dataset_name in enumerate(DATASET_NAMES):
        dataset_start = perf_counter()
        npz_path = DATA_DIR / f"{dataset_name}.npz"
        if not npz_path.exists():
            dataset_time_rows.append({
                "dataset": dataset_name,
                "status": "missing",
                "dataset_time_seconds": 0.0,
                "message": f"Missing dataset file: {npz_path}",
            })
            write_csv(DATASET_TIME_FILE, dataset_time_rows)
            continue

        try:
            data = np.load(npz_path)
            X_all = np.asarray(data["sample"], dtype=np.float64)
            y_raw = np.asarray(data["target"]).reshape(-1)
            if not np.all(np.isfinite(X_all)):
                raise ValueError("Input data contain NaN or Inf.")
            y_all = LabelEncoder().fit_transform(y_raw)
            n_classes = int(np.unique(y_all).size)
            if np.min(np.bincount(y_all)) < OUTER_SPLITS:
                raise ValueError(
                    "The smallest class contains fewer samples than "
                    "OUTER_SPLITS."
                )
        except Exception as exc:
            dataset_time_rows.append({
                "dataset": dataset_name,
                "status": "load_failed",
                "dataset_time_seconds": perf_counter() - dataset_start,
                "message": f"{type(exc).__name__}: {exc}",
            })
            write_csv(DATASET_TIME_FILE, dataset_time_rows)
            continue

        outer_splitter = RepeatedStratifiedKFold(
            n_splits=OUTER_SPLITS,
            n_repeats=OUTER_REPEATS,
            random_state=RANDOM_SEED,
        )
        outer_folds = list(outer_splitter.split(X_all, y_all))

        outer_acc: list[float] = []
        outer_bacc: list[float] = []
        outer_f1: list[float] = []
        outer_fit_time: list[float] = []
        outer_predict_time: list[float] = []
        outer_total_time: list[float] = []
        outer_pipeline_time: list[float] = []
        selection_times: list[float] = []
        train_zero_ratios: list[float] = []
        train_all_zero_ratios: list[float] = []
        test_zero_ratios: list[float] = []
        test_all_zero_ratios: list[float] = []
        selected_sets: list[int] = []
        selected_rules: list[int] = []
        selected_enhance: list[int] = []
        selected_total_nodes: list[int] = []

        print(
            f"\nRunning {OUTER_REPEATS} repetitions of "
            f"{OUTER_SPLITS}-fold CV with an inner stratified "
            f"holdout on {dataset_name}..."
        )

        for outer_index, (outer_train_idx, outer_test_idx) in enumerate(
            outer_folds
        ):
            outer_split = outer_index + 1
            repeat_id = outer_index // OUTER_SPLITS + 1
            fold_id = outer_index % OUTER_SPLITS + 1
            X_outer_train = X_all[outer_train_idx]
            y_outer_train = y_all[outer_train_idx]
            X_outer_test = X_all[outer_test_idx]
            y_outer_test = y_all[outer_test_idx]
            fit_idx, validation_idx = make_validation_split(
                y_outer_train, outer_split
            )

            selection_start = perf_counter()
            setting_scores: list[dict] = []

            for parameter_id, (
                n_sets,
                n_rules,
                n_enhance,
            ) in enumerate(parameter_grid, start=1):
                validation_seed = (
                    RANDOM_SEED
                    + dataset_index * 1_000_000
                    + parameter_id * 10_000
                    + outer_split * 100
                    + 1
                )
                validation_result = {
                    "accuracy": np.nan,
                    "balanced_accuracy": np.nan,
                    "macro_f1": np.nan,
                }
                validation_status = "ok"
                validation_error = ""
                try:
                    validation_result = evaluate_configuration(
                        X_outer_train[fit_idx],
                        y_outer_train[fit_idx],
                        X_outer_train[validation_idx],
                        y_outer_train[validation_idx],
                        n_classes,
                        n_sets,
                        n_rules,
                        n_enhance,
                        validation_seed,
                    )
                except Exception as exc:
                    validation_status = "failed"
                    validation_error = f"{type(exc).__name__}: {exc}"

                setting = {
                    "dataset": dataset_name,
                    "outer_repeat": repeat_id,
                    "outer_fold": fold_id,
                    "outer_split": outer_split,
                    "parameter_id": parameter_id,
                    "n_fuzzy_sets": n_sets,
                    "partition_G": n_sets - 1,
                    "n_rules": n_rules,
                    "n_enhancement_nodes": n_enhance,
                    "validation_seed": validation_seed,
                    "validation_accuracy": validation_result["accuracy"],
                    "validation_balanced_accuracy": validation_result[
                        "balanced_accuracy"
                    ],
                    "validation_macro_f1": validation_result["macro_f1"],
                    "status": validation_status,
                    "error": validation_error,
                }
                setting_scores.append(setting)
                validation_rows.append(setting)

            valid_settings = [
                row
                for row in setting_scores
                if row["status"] == "ok"
                and np.isfinite(row["validation_accuracy"])
            ]
            if not valid_settings:
                selection_time = perf_counter() - selection_start
                detail_rows.append({
                    "dataset": dataset_name,
                    "repeat": repeat_id,
                    "fold": fold_id,
                    "split": outer_split,
                    "parameter_id": np.nan,
                    "n_fuzzy_sets": np.nan,
                    "partition_G": np.nan,
                    "n_rules": np.nan,
                    "n_enhancement_nodes": np.nan,
                    "validation_accuracy": np.nan,
                    "seed": np.nan,
                    "status": "all_validation_settings_failed",
                    "accuracy": np.nan,
                    "balanced_accuracy": np.nan,
                    "macro_f1": np.nan,
                    "selection_time_seconds": selection_time,
                    "fit_time": np.nan,
                    "predict_time": np.nan,
                    "final_model_time": np.nan,
                    "total_computation_time": np.nan,
                    "train_raw_zero_ratio_percent": np.nan,
                    "train_raw_all_zero_row_ratio_percent": np.nan,
                    "test_raw_zero_ratio_percent": np.nan,
                    "test_raw_all_zero_row_ratio_percent": np.nan,
                    "error": "No configuration succeeded on validation data.",
                })
                outer_acc.append(np.nan)
                outer_bacc.append(np.nan)
                outer_f1.append(np.nan)
                outer_fit_time.append(np.nan)
                outer_predict_time.append(np.nan)
                outer_total_time.append(np.nan)
                outer_pipeline_time.append(np.nan)
                selection_times.append(selection_time)
                train_zero_ratios.append(np.nan)
                train_all_zero_ratios.append(np.nan)
                test_zero_ratios.append(np.nan)
                test_all_zero_ratios.append(np.nan)
                write_csv(VALIDATION_FILE, validation_rows)
                write_csv(DETAIL_FILE, detail_rows)
                print(
                    f"  Split {outer_split:02d}/"
                    f"{OUTER_SPLITS * OUTER_REPEATS}: "
                    "all validation configurations failed"
                )
                continue

            best = max(
                valid_settings,
                key=lambda row: row["validation_accuracy"],
            )
            selection_time = perf_counter() - selection_start
            parameter_id = int(best["parameter_id"])
            n_sets = int(best["n_fuzzy_sets"])
            n_rules = int(best["n_rules"])
            n_enhance = int(best["n_enhancement_nodes"])
            final_seed = (
                RANDOM_SEED
                + dataset_index * 1_000_000
                + parameter_id * 10_000
                + outer_split
            )

            final_start = perf_counter()
            status = "ok"
            error_message = ""
            result: dict = {
                "accuracy": np.nan,
                "balanced_accuracy": np.nan,
                "macro_f1": np.nan,
                "fit_time": np.nan,
                "predict_time": np.nan,
                "train_diag": {},
                "eval_diag": {},
            }
            try:
                result = evaluate_configuration(
                    X_outer_train,
                    y_outer_train,
                    X_outer_test,
                    y_outer_test,
                    n_classes,
                    n_sets,
                    n_rules,
                    n_enhance,
                    final_seed,
                )
            except Exception as exc:
                status = "failed"
                error_message = f"{type(exc).__name__}: {exc}"
            final_time = perf_counter() - final_start
            total_computation_time = (
                selection_time + final_time if status == "ok" else np.nan
            )

            train_diag = result["train_diag"]
            test_diag = result["eval_diag"]
            selected_rows.append({
                **best,
                "selection_time_seconds": selection_time,
            })
            detail_rows.append({
                "dataset": dataset_name,
                "repeat": repeat_id,
                "fold": fold_id,
                "split": outer_split,
                "parameter_id": parameter_id,
                "n_fuzzy_sets": n_sets,
                "partition_G": n_sets - 1,
                "n_rules": n_rules,
                "n_enhancement_nodes": n_enhance,
                "validation_accuracy": best["validation_accuracy"],
                "seed": final_seed,
                "status": status,
                "accuracy": result["accuracy"],
                "balanced_accuracy": result["balanced_accuracy"],
                "macro_f1": result["macro_f1"],
                "selection_time_seconds": selection_time,
                "fit_time": result["fit_time"],
                "predict_time": result["predict_time"],
                "final_model_time": final_time,
                "total_computation_time": total_computation_time,
                "train_raw_zero_ratio_percent": safe_diag(
                    train_diag,
                    "raw_zero_ratio_percent",
                ),
                "train_raw_all_zero_row_ratio_percent": safe_diag(
                    train_diag,
                    "raw_all_zero_row_ratio_percent",
                ),
                "test_raw_zero_ratio_percent": safe_diag(
                    test_diag,
                    "raw_zero_ratio_percent",
                ),
                "test_raw_all_zero_row_ratio_percent": safe_diag(
                    test_diag,
                    "raw_all_zero_row_ratio_percent",
                ),
                "error": error_message,
            })

            outer_acc.append(result["accuracy"])
            outer_bacc.append(result["balanced_accuracy"])
            outer_f1.append(result["macro_f1"])
            outer_fit_time.append(result["fit_time"])
            outer_predict_time.append(result["predict_time"])
            outer_total_time.append(final_time)
            outer_pipeline_time.append(total_computation_time)
            selection_times.append(selection_time)
            train_zero_ratios.append(
                safe_diag(train_diag, "raw_zero_ratio_percent")
            )
            train_all_zero_ratios.append(
                safe_diag(train_diag, "raw_all_zero_row_ratio_percent")
            )
            test_zero_ratios.append(
                safe_diag(test_diag, "raw_zero_ratio_percent")
            )
            test_all_zero_ratios.append(
                safe_diag(test_diag, "raw_all_zero_row_ratio_percent")
            )
            selected_sets.append(n_sets)
            selected_rules.append(n_rules)
            selected_enhance.append(n_enhance)
            selected_total_nodes.append(n_rules + n_enhance)

            write_csv(VALIDATION_FILE, validation_rows)
            write_csv(SELECTED_FILE, selected_rows)
            write_csv(DETAIL_FILE, detail_rows)
            print(
                f"  Split {outer_split:02d}/"
                f"{OUTER_SPLITS * OUTER_REPEATS}: "
                f"Sets={n_sets}, Rules={n_rules}, Enh={n_enhance} | "
                f"ACC={result['accuracy']:.4f} | "
                f"selection={selection_time:.2f}s"
            )

        accuracy_mean, accuracy_std = repeated_cv_mean_std(outer_acc)
        bacc_mean, bacc_std = repeated_cv_mean_std(outer_bacc)
        f1_mean, f1_std = repeated_cv_mean_std(outer_f1)
        fit_mean, fit_std = repeated_cv_mean_std(outer_fit_time)
        predict_mean, predict_std = repeated_cv_mean_std(outer_predict_time)
        final_time_mean, final_time_std = repeated_cv_mean_std(
            outer_total_time
        )
        selection_mean, selection_std = repeated_cv_mean_std(selection_times)
        total_time_mean, total_time_std = repeated_cv_mean_std(
            outer_pipeline_time
        )

        summary_rows.append({
            "dataset": dataset_name,
            "accuracy_mean": accuracy_mean,
            "accuracy_std": accuracy_std,
            "balanced_accuracy_mean": bacc_mean,
            "balanced_accuracy_std": bacc_std,
            "macro_f1_mean": f1_mean,
            "macro_f1_std": f1_std,
            "selected_n_fuzzy_sets_mean": (
                float(np.mean(selected_sets)) if selected_sets else np.nan
            ),
            "selected_n_fuzzy_sets_mode": (
                mode_int(selected_sets) if selected_sets else np.nan
            ),
            "selected_n_rules_mean": (
                float(np.mean(selected_rules)) if selected_rules else np.nan
            ),
            "selected_n_rules_mode": (
                mode_int(selected_rules) if selected_rules else np.nan
            ),
            "selected_n_enhancement_nodes_mean": (
                float(np.mean(selected_enhance))
                if selected_enhance
                else np.nan
            ),
            "selected_n_enhancement_nodes_mode": (
                mode_int(selected_enhance) if selected_enhance else np.nan
            ),
            "selected_total_nodes_mean": (
                float(np.mean(selected_total_nodes))
                if selected_total_nodes
                else np.nan
            ),
            "fit_time_mean": fit_mean,
            "fit_time_std": fit_std,
            "predict_time_mean": predict_mean,
            "predict_time_std": predict_std,
            "final_model_time_mean": final_time_mean,
            "final_model_time_std": final_time_std,
            "selection_time_mean": selection_mean,
            "selection_time_std": selection_std,
            "total_computation_time_mean": total_time_mean,
            "total_computation_time_std": total_time_std,
            "train_raw_zero_ratio_percent_mean": repeated_cv_mean(
                train_zero_ratios
            ),
            "train_raw_all_zero_row_ratio_percent_mean": repeated_cv_mean(
                train_all_zero_ratios
            ),
            "test_raw_zero_ratio_percent_mean": repeated_cv_mean(
                test_zero_ratios
            ),
            "test_raw_all_zero_row_ratio_percent_mean": repeated_cv_mean(
                test_all_zero_ratios
            ),
            "valid_outer_folds": int(np.sum(np.isfinite(outer_acc))),
            "failed_outer_folds": int(np.sum(~np.isfinite(outer_acc))),
            "dataset_time_seconds": perf_counter() - dataset_start,
        })
        write_csv(SUMMARY_FILE, summary_rows)

        dataset_time_rows.append({
            "dataset": dataset_name,
            "status": "completed",
            "dataset_time_seconds": perf_counter() - dataset_start,
            "message": "",
        })
        write_csv(DATASET_TIME_FILE, dataset_time_rows)
        print(
            f"[{dataset_name}] ACC={accuracy_mean:.4f}±{accuracy_std:.4f} | "
            f"BACC={bacc_mean:.4f}±{bacc_std:.4f} | "
            f"Macro-F1={f1_mean:.4f}±{f1_std:.4f}"
        )

    print(f"\nValidation settings saved to: {VALIDATION_FILE}")
    print(f"Selected settings saved to: {SELECTED_FILE}")
    print(f"Outer-CV details saved to: {DETAIL_FILE}")
    print(f"Holdout-CV summary saved to: {SUMMARY_FILE}")
    print(f"Dataset times saved to: {DATASET_TIME_FILE}")


if __name__ == "__main__":
    main()
