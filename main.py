from __future__ import annotations

import csv
from pathlib import Path
from time import perf_counter

import numpy as np
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.preprocessing import OneHotEncoder

from Feature_mapping_layer import HDFBLSFeatureMap
from Enhancement_node_layer import HDFBLSEnhanceNode, HDFBLSEnhanceConfig


# -------------------- Configuration --------------------
N_SPLITS = 5
REPEATS = 5
RANDOM_SEED = 2026

# ORL is included in the public repository as a runnable example. Add the
# remaining .npz files to datasets_npz and uncomment their names as needed.
DATASET_NAMES = [
    # "PAGEB",          # D1
    # "Thyroid",        # D2
    # "SATELLITE",      # D3
    # "TEXTURE",        # D4
    # "spambase",       # D5
    "ORL",              # D6
    # "semg",           # D7
    # "warpAR10P (1)",  # D8
    # "PIE",            # D9
    # "handoutlines",   # D10
    # "Giesste",        # D11
    # "Drivace",        # D12
    # "Leukemia",       # D13
    # "CMHS",           # D14
    # "GSAFM",          # D15
]

REPOSITORY_DIR = Path(__file__).resolve().parent
DATA_DIR = REPOSITORY_DIR / "datasets_npz"
RESULT_DIR = REPOSITORY_DIR / "results"
RESULT_DIR.mkdir(parents=True, exist_ok=True)

DETAIL_FILE = RESULT_DIR / "cv_details.csv"
SUMMARY_FILE = RESULT_DIR / "cv_summary.csv"


def make_onehot_encoder() -> OneHotEncoder:
    """Create a dense one-hot encoder across supported sklearn versions."""
    try:
        return OneHotEncoder(sparse_output=False, handle_unknown="error")
    except TypeError:
        return OneHotEncoder(sparse=False, handle_unknown="error")


def minmax_fit(X_train: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Fit feature-wise min-max scaling on the training fold only."""
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
    """Apply training-fold scaling and restrict inputs to [0, 1]."""
    return np.clip((X - x_min) / x_range, 0.0, 1.0)


def predict_labels(
    model: HDFBLSEnhanceNode,
    F_test: np.ndarray,
) -> np.ndarray:
    """Return class predictions using the selected HDFBLS structure."""
    if model.W_ is None:
        raise RuntimeError("The enhancement model has not been trained.")

    F_test = np.asarray(F_test, dtype=np.float64)
    Z_test = model._row_normalize(F_test)

    enhancement_columns = [
        model._evaluate_composite_rule(F_test, meta)
        for meta in model.blocks_meta
    ]
    if enhancement_columns:
        E_test = np.concatenate(enhancement_columns, axis=1)
        Z_test = np.concatenate([Z_test, E_test], axis=1)

    if Z_test.shape[1] != model.W_.shape[0]:
        raise ValueError(
            f"Shape mismatch: test matrix has {Z_test.shape[1]} columns, "
            f"but W_ expects {model.W_.shape[0]}."
        )

    return np.argmax(Z_test @ model.W_, axis=1)


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return

    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def sample_mean_std(values: list[float]) -> tuple[float, float]:
    """Return the mean and sample standard deviation of finite values."""
    array = np.asarray(values, dtype=np.float64)
    array = array[np.isfinite(array)]
    if array.size == 0:
        return float("nan"), float("nan")
    mean_value = float(np.mean(array))
    std_value = float(np.std(array, ddof=1)) if array.size > 1 else 0.0
    return mean_value, std_value


def repetition_means(fold_groups: list[list[float]]) -> list[float]:
    """Average the five folds within each cross-validation repetition."""
    means: list[float] = []
    for repeat_id, values in enumerate(fold_groups, start=1):
        if len(values) != N_SPLITS:
            raise RuntimeError(
                f"Repetition {repeat_id} contains {len(values)} folds; "
                f"expected {N_SPLITS}."
            )
        means.append(float(np.mean(np.asarray(values, dtype=np.float64))))
    return means


def main() -> None:
    detail_rows: list[dict] = []
    summary_rows: list[dict] = []

    for dataset_name in DATASET_NAMES:
        dataset_start = perf_counter()

        npz_path = DATA_DIR / f"{dataset_name}.npz"
        if not npz_path.exists():
            raise FileNotFoundError(f"Missing dataset file: {npz_path}")

        data = np.load(npz_path)
        X_all = np.asarray(data["sample"], dtype=np.float64)
        y_all = np.asarray(data["target"]).reshape(-1)

        _, class_counts = np.unique(y_all, return_counts=True)
        min_class_count = int(np.min(class_counts))
        if min_class_count < N_SPLITS:
            raise ValueError(
                f"{dataset_name}: the smallest class contains only "
                f"{min_class_count} samples, which is insufficient for "
                f"{N_SPLITS}-fold stratified cross-validation."
            )

        # Establish one fixed class-to-column mapping for all folds.
        encoder = make_onehot_encoder()
        encoder.fit(y_all.reshape(-1, 1))

        cv = RepeatedStratifiedKFold(
            n_splits=N_SPLITS,
            n_repeats=REPEATS,
            random_state=RANDOM_SEED,
        )

        repeat_accs = [[] for _ in range(REPEATS)]
        repeat_baccs = [[] for _ in range(REPEATS)]
        repeat_macro_f1s = [[] for _ in range(REPEATS)]
        repeat_n_rules = [[] for _ in range(REPEATS)]
        repeat_enh_depths = [[] for _ in range(REPEATS)]
        repeat_enh_total_nodes = [[] for _ in range(REPEATS)]
        repeat_enh_times = [[] for _ in range(REPEATS)]
        repeat_fold_total_times = [[] for _ in range(REPEATS)]

        print(
            f"\nRunning {N_SPLITS}-fold x {REPEATS}-repeat "
            f"stratified CV on {dataset_name}..."
        )

        for split_index, (train_index, test_index) in enumerate(
            cv.split(X_all, y_all),
            start=1,
        ):
            repeat_id = (split_index - 1) // N_SPLITS + 1
            fold_id = (split_index - 1) % N_SPLITS + 1
            repeat_index = repeat_id - 1

            # Preserve the random-seed sequence used in the reported results.
            # The 25 fold evaluations use seeds 2026, 2027, ..., 2050.
            seed = RANDOM_SEED + split_index - 1

            X_train_raw = X_all[train_index]
            X_test_raw = X_all[test_index]
            y_train = y_all[train_index]
            y_test = y_all[test_index]

            fold_start = perf_counter()

            x_min, x_range = minmax_fit(X_train_raw)
            X_train = minmax_transform(X_train_raw, x_min, x_range)
            X_test = minmax_transform(X_test_raw, x_min, x_range)

            Y_train = encoder.transform(
                y_train.reshape(-1, 1)
            ).astype(np.float64)
            y_test_idx = encoder.transform(
                y_test.reshape(-1, 1)
            ).argmax(axis=1)

            feature_layer = HDFBLSFeatureMap()
            F_train = feature_layer.train_F(
                X_train,
                Y_train=Y_train,
                eval_every=5,
                patience=5,
                min_rules=5,
            )
            F_test = feature_layer.test_F(X_test)

            enhance_layer = HDFBLSEnhanceNode(
                F_train.shape[1],
                config=HDFBLSEnhanceConfig(random_state=seed),
            )
            enhance_layer.train_enhancement_incremental(
                F_train,
                Y_train,
                20,
            )

            y_pred = predict_labels(enhance_layer, F_test)
            fold_total_time = perf_counter() - fold_start

            test_acc = float(accuracy_score(y_test_idx, y_pred))
            test_bacc = float(balanced_accuracy_score(y_test_idx, y_pred))
            test_macro_f1 = float(
                f1_score(
                    y_test_idx,
                    y_pred,
                    average="macro",
                    zero_division=0,
                )
            )

            n_rules = int(
                getattr(feature_layer, "n_centers_", F_train.shape[1])
            )
            enh_depth = float(
                getattr(enhance_layer, "best_node_depth", np.nan)
            )
            enh_total = float(
                getattr(enhance_layer, "best_node_total_nodes", np.nan)
            )
            enh_time = float(
                getattr(enhance_layer, "enhancement_time", np.nan)
            )

            repeat_accs[repeat_index].append(test_acc)
            repeat_baccs[repeat_index].append(test_bacc)
            repeat_macro_f1s[repeat_index].append(test_macro_f1)
            repeat_n_rules[repeat_index].append(float(n_rules))
            repeat_enh_depths[repeat_index].append(enh_depth)
            repeat_enh_total_nodes[repeat_index].append(enh_total)
            repeat_enh_times[repeat_index].append(enh_time)
            repeat_fold_total_times[repeat_index].append(float(fold_total_time))

            detail_rows.append({
                "dataset": dataset_name,
                "repeat": repeat_id,
                "fold": fold_id,
                "split": split_index,
                "seed": seed,
                "train_size": int(train_index.size),
                "test_size": int(test_index.size),
                "test_acc": test_acc,
                "test_bacc": test_bacc,
                "test_macro_f1": test_macro_f1,
                "n_rules": n_rules,
                "enh_depth": enh_depth,
                "enh_total_nodes": enh_total,
                "enh_time": enh_time,
                "fold_total_time": float(fold_total_time),
            })

            print(
                f"  Repeat {repeat_id}/{REPEATS}, "
                f"Fold {fold_id}/{N_SPLITS} | "
                f"ACC={test_acc:.4f} | "
                f"BACC={test_bacc:.4f} | "
                f"Macro-F1={test_macro_f1:.4f} | "
                f"TotalTime={fold_total_time:.4f}s"
            )

        repetition_acc_means = repetition_means(repeat_accs)
        repetition_bacc_means = repetition_means(repeat_baccs)
        repetition_macro_f1_means = repetition_means(repeat_macro_f1s)
        repetition_rule_means = repetition_means(repeat_n_rules)
        repetition_depth_means = repetition_means(repeat_enh_depths)
        repetition_node_means = repetition_means(repeat_enh_total_nodes)
        repetition_enh_time_means = repetition_means(repeat_enh_times)
        repetition_fold_time_means = repetition_means(
            repeat_fold_total_times
        )

        acc_mean, acc_std = sample_mean_std(repetition_acc_means)
        bacc_mean, bacc_std = sample_mean_std(repetition_bacc_means)
        f1_mean, f1_std = sample_mean_std(repetition_macro_f1_means)
        rules_mean, rules_std = sample_mean_std(repetition_rule_means)
        depth_mean, depth_std = sample_mean_std(repetition_depth_means)
        nodes_mean, nodes_std = sample_mean_std(repetition_node_means)
        enh_time_mean, enh_time_std = sample_mean_std(
            repetition_enh_time_means
        )
        fold_time_mean, fold_time_std = sample_mean_std(
            repetition_fold_time_means
        )

        dataset_time = perf_counter() - dataset_start

        summary_rows.append({
            "dataset": dataset_name,
            "cv_splits": N_SPLITS,
            "cv_repeats": REPEATS,
            "n_evaluations": N_SPLITS * REPEATS,
            "acc_mean": acc_mean,
            "acc_std": acc_std,
            "bacc_mean": bacc_mean,
            "bacc_std": bacc_std,
            "macro_f1_mean": f1_mean,
            "macro_f1_std": f1_std,
            "n_rules_mean": rules_mean,
            "n_rules_std": rules_std,
            "enh_depth_mean": depth_mean,
            "enh_depth_std": depth_std,
            "enh_total_nodes_mean": nodes_mean,
            "enh_total_nodes_std": nodes_std,
            "enh_time_mean": enh_time_mean,
            "enh_time_std": enh_time_std,
            "fold_total_time_mean": fold_time_mean,
            "fold_total_time_std": fold_time_std,
            "dataset_time": float(dataset_time),
        })

        print(
            f"[{dataset_name}] "
            f"ACC={acc_mean:.4f}±{acc_std:.4f} | "
            f"BACC={bacc_mean:.4f}±{bacc_std:.4f} | "
            f"Macro-F1={f1_mean:.4f}±{f1_std:.4f} | "
            f"Rules={rules_mean:.1f}±{rules_std:.1f} | "
            f"FoldTime={fold_time_mean:.4f}±{fold_time_std:.4f}s | "
            f"DatasetTime={dataset_time:.2f}s"
        )

    write_csv(DETAIL_FILE, detail_rows)
    write_csv(SUMMARY_FILE, summary_rows)

    print(f"\nDetailed results saved to: {DETAIL_FILE}")
    print(f"Summary results saved to: {SUMMARY_FILE}")


if __name__ == "__main__":
    main()
