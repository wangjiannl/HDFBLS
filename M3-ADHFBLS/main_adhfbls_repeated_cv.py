from __future__ import annotations

import copy
import csv
import json
from pathlib import Path
from time import perf_counter

import numpy as np
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score
from sklearn.model_selection import RepeatedStratifiedKFold, train_test_split
from sklearn.preprocessing import LabelEncoder

from ADHFBLS_checked import ADHFBLS, ADHFBLSConfig


# ============================================================
# Common outer evaluation protocol
# ============================================================
CV_SPLITS = 5
CV_REPEATS = 5
RANDOM_SEED = 2026

DATASET_NAMES = [
 #   "PAGEB",          # D1
 #   "Thyroid",        # D2
 #   "SATELLITE",      # D3
 #   "TEXTURE",        # D4
 #   "spambase",       # D5
    "ORL",            # D6
 #   "semg",           # D7
 #   "warpAR10P (1)",  # D8
 #   "PIE",            # D9
 #   "handoutlines",   # D10
 #   "Giesste",        # D11
 #   "Drivace",        # D12
 #   "Leukemia",       # D13
 #   "CMHS",           # D14
 #   "GSAFM",          # D15
]

DATA_DIR = Path("datasets_npz")
RESULT_DIR = Path("results_adhfbls_cv")
DETAIL_FILE = RESULT_DIR / "adhfbls_cv_details.csv"
SUMMARY_FILE = RESULT_DIR / "adhfbls_cv_summary.csv"
DATASET_TIME_FILE = RESULT_DIR / "adhfbls_cv_dataset_times.csv"


# ============================================================
# ADHFBLS settings from the manuscript / official source
# ============================================================
INITIAL_RULES = 2
ETA = 1
TASK_TYPE = 2
SHRINKAGE = 0.8
REGULARIZATION = 2.0 ** (-30)
SEARCH_REPETITIONS = 5       # tau
MAX_SEARCHES = 10            # t_max
INITIAL_DELTA = 1
INNER_VALIDATION_RATIO = 0.20


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
    X_scaled = (X - x_min) / x_range
    return np.clip(X_scaled, 0.0, 1.0)


def one_hot(y: np.ndarray, n_classes: int) -> np.ndarray:
    return np.eye(n_classes, dtype=np.float64)[y]


def finite_mean(values: list[float]) -> float:
    array = np.asarray(values, dtype=np.float64)
    array = array[np.isfinite(array)]
    return float(np.mean(array)) if array.size else np.nan


def finite_std(values: list[float]) -> float:
    array = np.asarray(values, dtype=np.float64)
    array = array[np.isfinite(array)]
    return float(np.std(array, ddof=1)) if array.size > 1 else 0.0


def repeated_cv_mean_std(values: list[float]) -> tuple[float, float]:
    """Average folds within each repetition, then summarize repetitions."""
    array = np.asarray(values, dtype=np.float64)
    expected = CV_SPLITS * CV_REPEATS
    if array.size != expected:
        raise ValueError(
            f"Expected {expected} fold values, but received {array.size}."
        )
    repetition_means = np.mean(
        array.reshape(CV_REPEATS, CV_SPLITS),
        axis=1,
    )
    values_by_repeat = repetition_means.tolist()
    return finite_mean(values_by_repeat), finite_std(values_by_repeat)


def repeated_cv_mean(values: list[float]) -> float:
    return repeated_cv_mean_std(values)[0]


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def metric_triplet(y_true: np.ndarray, y_pred: np.ndarray) -> tuple[float, float, float]:
    acc = float(accuracy_score(y_true, y_pred))
    bacc = float(balanced_accuracy_score(y_true, y_pred))
    macro_f1 = float(
        f1_score(y_true, y_pred, average="macro", zero_division=0)
    )
    values = np.asarray([acc, bacc, macro_f1], dtype=np.float64)
    if not np.all(np.isfinite(values)):
        raise FloatingPointError("A non-finite evaluation metric was produced.")
    return acc, bacc, macro_f1


def validation_accuracy(model: ADHFBLS, X_val: np.ndarray, y_val: np.ndarray) -> float:
    prediction = model.predict(X_val)
    return float(accuracy_score(y_val, prediction))


def select_structure(
    X_fit: np.ndarray,
    Y_fit: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    fold_seed: int,
) -> tuple[ADHFBLS, float, list[dict], int]:
    """
    Run the official adaptive search without touching the outer test fold.

    The official demonstration evaluates candidate increments on its held-out
    set. Here that role is assigned to an inner validation subset drawn only
    from the outer training fold. After selection, the output weights are
    refitted on the complete outer training fold by the caller.
    """
    model = ADHFBLS(
        config=ADHFBLSConfig(
            initial_rules=INITIAL_RULES,
            eta=ETA,
            task_type=TASK_TYPE,
            shrinkage=SHRINKAGE,
            regularization=REGULARIZATION,
            random_state=fold_seed,
        )
    )
    model.train(X_fit, Y_fit)
    current_acc = validation_accuracy(model, X_val, y_val)

    delta = INITIAL_DELTA
    trace: list[dict] = []
    candidate_failures = 0

    for search_step in range(1, MAX_SEARCHES + 1):
        best_rule_model: ADHFBLS | None = None
        best_rule_acc = -np.inf
        best_enhance_model: ADHFBLS | None = None
        best_enhance_acc = -np.inf

        for repetition in range(SEARCH_REPETITIONS):
            candidate_seed = (
                fold_seed + search_step * 10_000 + repetition * 100 + 1
            )
            try:
                candidate = copy.deepcopy(model)
                candidate.set_random_state(candidate_seed)
                candidate.ruleIncrement(delta, X_fit, Y_fit)
                score = validation_accuracy(candidate, X_val, y_val)
                if score > best_rule_acc:
                    best_rule_acc = score
                    best_rule_model = candidate
            except Exception:
                candidate_failures += 1

        for repetition in range(SEARCH_REPETITIONS):
            candidate_seed = (
                fold_seed + search_step * 10_000 + repetition * 100 + 2
            )
            try:
                candidate = copy.deepcopy(model)
                candidate.set_random_state(candidate_seed)
                candidate.enhanceNodeIncrement(delta, Y_fit)
                score = validation_accuracy(candidate, X_val, y_val)
                if score > best_enhance_acc:
                    best_enhance_acc = score
                    best_enhance_model = candidate
            except Exception:
                candidate_failures += 1

        selected = "none"
        accepted = False

        if (
            best_rule_model is not None
            and best_rule_acc > best_enhance_acc
            and best_rule_acc >= current_acc
        ):
            model = best_rule_model
            current_acc = best_rule_acc
            delta *= 2
            selected = "rule"
            accepted = True
        elif (
            best_enhance_model is not None
            and best_enhance_acc >= best_rule_acc
            and best_enhance_acc >= current_acc
        ):
            model = best_enhance_model
            current_acc = best_enhance_acc
            delta *= 2
            selected = "enhancement"
            accepted = True
        else:
            if delta == 1:
                trace.append({
                    "step": search_step,
                    "delta": delta,
                    "rule_validation_acc": (
                        best_rule_acc if np.isfinite(best_rule_acc) else None
                    ),
                    "enhancement_validation_acc": (
                        best_enhance_acc if np.isfinite(best_enhance_acc) else None
                    ),
                    "selected": selected,
                    "accepted": False,
                    **model.structure(),
                })
                break
            delta = 1

        trace.append({
            "step": search_step,
            "delta": delta,
            "rule_validation_acc": (
                best_rule_acc if np.isfinite(best_rule_acc) else None
            ),
            "enhancement_validation_acc": (
                best_enhance_acc if np.isfinite(best_enhance_acc) else None
            ),
            "selected": selected,
            "accepted": accepted,
            **model.structure(),
        })

    return model, current_acc, trace, candidate_failures


# ============================================================
# Main experiment
# ============================================================
def main() -> None:
    RESULT_DIR.mkdir(parents=True, exist_ok=True)
    detail_rows: list[dict] = []
    summary_rows: list[dict] = []
    dataset_time_rows: list[dict] = []

    for dataset_index, dataset_name in enumerate(DATASET_NAMES):
        dataset_start = perf_counter()
        npz_path = DATA_DIR / f"{dataset_name}.npz"

        if not npz_path.exists():
            message = f"Missing dataset file: {npz_path}"
            print(f"[Skipped] {message}")
            dataset_time_rows.append({
                "dataset": dataset_name,
                "status": "missing",
                "dataset_time": 0.0,
                "message": message,
            })
            write_csv(DATASET_TIME_FILE, dataset_time_rows)
            continue

        try:
            data = np.load(npz_path)
            X_all = np.asarray(data["sample"], dtype=np.float64)
            y_raw = np.asarray(data["target"]).reshape(-1)
        except Exception as exc:
            message = f"{type(exc).__name__}: {exc}"
            print(f"[Skipped] {dataset_name}: {message}")
            dataset_time_rows.append({
                "dataset": dataset_name,
                "status": "load_failed",
                "dataset_time": perf_counter() - dataset_start,
                "message": message,
            })
            write_csv(DATASET_TIME_FILE, dataset_time_rows)
            continue

        if not np.all(np.isfinite(X_all)):
            message = "Input data contain NaN or Inf."
            print(f"[Skipped] {dataset_name}: {message}")
            dataset_time_rows.append({
                "dataset": dataset_name,
                "status": "invalid_data",
                "dataset_time": perf_counter() - dataset_start,
                "message": message,
            })
            write_csv(DATASET_TIME_FILE, dataset_time_rows)
            continue

        label_encoder = LabelEncoder()
        y_all = label_encoder.fit_transform(y_raw)
        n_classes = int(np.unique(y_all).size)
        class_counts = np.bincount(y_all)

        if np.min(class_counts) < CV_SPLITS:
            message = (
                f"The smallest class has {np.min(class_counts)} samples, "
                f"fewer than CV_SPLITS={CV_SPLITS}."
            )
            print(f"[Skipped] {dataset_name}: {message}")
            dataset_time_rows.append({
                "dataset": dataset_name,
                "status": "insufficient_class_count",
                "dataset_time": perf_counter() - dataset_start,
                "message": message,
            })
            write_csv(DATASET_TIME_FILE, dataset_time_rows)
            continue

        splitter = RepeatedStratifiedKFold(
            n_splits=CV_SPLITS,
            n_repeats=CV_REPEATS,
            random_state=RANDOM_SEED,
        )
        splits = list(splitter.split(X_all, y_all))
        n_folds = len(splits)

        print(
            f"\nRunning {CV_SPLITS}-fold x {CV_REPEATS}-repeat "
            f"stratified CV on {dataset_name}..."
        )

        for fold_id, (outer_train_idx, test_idx) in enumerate(splits, start=1):
            fold_start = perf_counter()
            status = "ok"
            error_message = ""
            acc = bacc = macro_f1 = np.nan
            validation_acc = np.nan
            base_search_time = refit_time = predict_time = np.nan
            candidate_failures = 0
            trace: list[dict] = []
            structure = {
                "rules": np.nan,
                "upper_feature_nodes": np.nan,
                "lower_feature_nodes": np.nan,
                "feature_nodes": np.nan,
                "enhancement_nodes": np.nan,
                "total_nodes": np.nan,
                "rule_increments": np.nan,
                "enhancement_increments": np.nan,
            }
            train_diag = {
                "raw_zero_ratio_percent": np.nan,
                "raw_all_zero_row_ratio_percent": np.nan,
            }
            test_diag = dict(train_diag)

            fold_seed = RANDOM_SEED + dataset_index * 100_000 + fold_id

            try:
                X_outer_train_raw = X_all[outer_train_idx]
                X_test_raw = X_all[test_idx]
                y_outer_train = y_all[outer_train_idx]
                y_test = y_all[test_idx]

                x_min, x_range = minmax_fit(X_outer_train_raw)
                X_outer_train = minmax_transform(
                    X_outer_train_raw, x_min, x_range
                )
                X_test = minmax_transform(X_test_raw, x_min, x_range)
                Y_outer_train = one_hot(y_outer_train, n_classes)

                inner_indices = np.arange(X_outer_train.shape[0])
                fit_idx, val_idx = train_test_split(
                    inner_indices,
                    test_size=INNER_VALIDATION_RATIO,
                    shuffle=True,
                    stratify=y_outer_train,
                    random_state=fold_seed,
                )
                X_fit = X_outer_train[fit_idx]
                Y_fit = Y_outer_train[fit_idx]
                X_val = X_outer_train[val_idx]
                y_val = y_outer_train[val_idx]

                search_start = perf_counter()
                model, validation_acc, trace, candidate_failures = select_structure(
                    X_fit=X_fit,
                    Y_fit=Y_fit,
                    X_val=X_val,
                    y_val=y_val,
                    fold_seed=fold_seed,
                )
                base_search_time = perf_counter() - search_start

                refit_start = perf_counter()
                model.refit_output(X_outer_train, Y_outer_train)
                refit_time = perf_counter() - refit_start
                train_diag = dict(model.train_activation_diagnostics)

                predict_start = perf_counter()
                y_pred = model.predict(X_test)
                predict_time = perf_counter() - predict_start
                test_diag = dict(model.last_activation_diagnostics)

                acc, bacc, macro_f1 = metric_triplet(y_test, y_pred)
                structure = model.structure()

            except Exception as exc:
                status = "failed"
                error_message = f"{type(exc).__name__}: {exc}"

            fold_time = perf_counter() - fold_start
            detail_rows.append({
                "dataset": dataset_name,
                "fold": fold_id,
                "repeat": (fold_id - 1) // CV_SPLITS + 1,
                "fold_within_repeat": (fold_id - 1) % CV_SPLITS + 1,
                "seed": fold_seed,
                "status": status,
                "error_message": error_message,
                "accuracy": acc,
                "balanced_accuracy": bacc,
                "macro_f1": macro_f1,
                "validation_accuracy": validation_acc,
                **structure,
                "search_steps": len(trace),
                "candidate_failures": candidate_failures,
                "increment_trace": json.dumps(trace, ensure_ascii=False),
                "train_raw_zero_ratio_percent": train_diag.get(
                    "raw_zero_ratio_percent", np.nan
                ),
                "train_raw_all_zero_row_ratio_percent": train_diag.get(
                    "raw_all_zero_row_ratio_percent", np.nan
                ),
                "test_raw_zero_ratio_percent": test_diag.get(
                    "raw_zero_ratio_percent", np.nan
                ),
                "test_raw_all_zero_row_ratio_percent": test_diag.get(
                    "raw_all_zero_row_ratio_percent", np.nan
                ),
                "base_and_search_time": base_search_time,
                "refit_time": refit_time,
                "predict_time": predict_time,
                "fold_total_time": fold_time,
            })
            write_csv(DETAIL_FILE, detail_rows)

            print(
                f"  Fold {fold_id:02d}/{n_folds} | {status.upper()} | "
                f"ACC={acc:.4f} | BACC={bacc:.4f} | "
                f"Macro-F1={macro_f1:.4f} | "
                f"Rules={structure['rules']} | "
                f"Enh={structure['enhancement_nodes']} | "
                f"Time={fold_time:.2f}s"
            )

        selected = [row for row in detail_rows if row["dataset"] == dataset_name]
        successful = [row for row in selected if row["status"] == "ok"]
        failed_folds = len(selected) - len(successful)

        acc_mean, acc_std = repeated_cv_mean_std(
            [row["accuracy"] for row in selected]
        )
        bacc_mean, bacc_std = repeated_cv_mean_std(
            [row["balanced_accuracy"] for row in selected]
        )
        f1_mean, f1_std = repeated_cv_mean_std(
            [row["macro_f1"] for row in selected]
        )
        fold_time_mean, fold_time_std = repeated_cv_mean_std(
            [row["fold_total_time"] for row in selected]
        )

        summary = {
            "dataset": dataset_name,
            "successful_folds": len(successful),
            "failed_folds": failed_folds,
            "total_folds": n_folds,
            "accuracy_mean": acc_mean,
            "accuracy_std": acc_std,
            "balanced_accuracy_mean": bacc_mean,
            "balanced_accuracy_std": bacc_std,
            "macro_f1_mean": f1_mean,
            "macro_f1_std": f1_std,
            "rules_mean": repeated_cv_mean(
                [row["rules"] for row in selected]
            ),
            "upper_feature_nodes_mean": repeated_cv_mean(
                [row["upper_feature_nodes"] for row in selected]
            ),
            "lower_feature_nodes_mean": repeated_cv_mean(
                [row["lower_feature_nodes"] for row in selected]
            ),
            "enhancement_nodes_mean": repeated_cv_mean(
                [row["enhancement_nodes"] for row in selected]
            ),
            "total_nodes_mean": repeated_cv_mean(
                [row["total_nodes"] for row in selected]
            ),
            "search_steps_mean": repeated_cv_mean(
                [row["search_steps"] for row in selected]
            ),
            "candidate_failures_total": int(
                sum(row["candidate_failures"] for row in selected)
            ),
            "train_raw_zero_ratio_percent_mean": repeated_cv_mean(
                [row["train_raw_zero_ratio_percent"] for row in selected]
            ),
            "test_raw_zero_ratio_percent_mean": repeated_cv_mean(
                [row["test_raw_zero_ratio_percent"] for row in selected]
            ),
            "fold_total_time_mean": fold_time_mean,
            "fold_total_time_std": fold_time_std,
            "dataset_time": perf_counter() - dataset_start,
        }
        summary_rows.append(summary)
        write_csv(SUMMARY_FILE, summary_rows)

        dataset_time_rows.append({
            "dataset": dataset_name,
            "status": "completed" if successful else "all_failed",
            "dataset_time": summary["dataset_time"],
            "message": f"Failed folds: {failed_folds}/{n_folds}",
        })
        write_csv(DATASET_TIME_FILE, dataset_time_rows)

        print(
            f"[{dataset_name}] "
            f"ACC={summary['accuracy_mean']:.4f}±{summary['accuracy_std']:.4f} | "
            f"BACC={summary['balanced_accuracy_mean']:.4f}"
            f"±{summary['balanced_accuracy_std']:.4f} | "
            f"Macro-F1={summary['macro_f1_mean']:.4f}"
            f"±{summary['macro_f1_std']:.4f} | "
            f"Failed={failed_folds}/{n_folds} | "
            f"Time={summary['dataset_time']:.2f}s"
        )

    print(f"\nDetailed results saved to: {DETAIL_FILE}")
    print(f"Summary results saved to: {SUMMARY_FILE}")
    print(f"Dataset times saved to: {DATASET_TIME_FILE}")


if __name__ == "__main__":
    main()
