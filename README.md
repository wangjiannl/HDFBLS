# HDFBLS

This repository provides the implementation used in the paper:

**High-Dimensional Rule-Based Fuzzy Broad Network with Adaptive Incremental Architecture**

## Repository structure

HDFBLS/

├── M1-FBLS/                  # Baseline: FBLS

├── M2-CFBLS/                 # Baseline: CFBLS

├── M3-ADHFBLS/               # Baseline: ADHFBLS

├── M4-IFBLS/                 # Baseline: IFBLS

├── M5-HDRBFNN/               # Baseline: HDRBFNN

├── M6-FSRE-AdaTSK/           # Baseline: FSRE-AdaTSK

├── datasets_npz/             # Datasets in .npz format

├── Enhancement_node_layer.py # DET-based enhancement layer

├── Feature_mapping_layer.py  # ARGS-based feature mapping layer

├── Unit.py                   # Utility functions

└── main.py                   # Main script


## Main components

The implementation contains two main modules of HDFBLS.

### Adaptive rule generation scheme

`Feature_mapping_layer.py` implements the ARGS-based feature mapping layer. ARGS generates zero-order fuzzy rules incrementally and introduces underflow-aware constraints on fuzzy antecedent parameters. This helps avoid numerical underflow when firing strengths are computed in high-dimensional spaces.

### Dynamic elite tree

`Enhancement_node_layer.py` implements the DET-based enhancement layer. Instead of generating random enhancement nodes, DET constructs composite fuzzy rules over the activation states of basic rules. It retains and reuses promising rule groups during the search, enabling competitive rule composition under bounded structure growth.

## Requirements

The HDFBLS implementation and the Python baselines require:

```bash
numpy
scipy
scikit-learn
pandas
```

You can install the required packages with:

```bash
pip install numpy scipy scikit-learn pandas
```

## Running HDFBLS

To run the main experiment, execute:

```bash
python main.py
```

The experimental settings, including dataset selection, number of repetitions, cross-validation protocol, and model parameters, can be configured in `main.py`.

In the revised experiments reported in the paper, the results are evaluated using five repetitions of 5-fold stratified cross-validation. The test fold is used only for final evaluation and remains unseen during parameter selection, model construction, and training.

Within each repetition, the five fold scores are averaged first. The reported mean and sample standard deviation are then calculated from the five repetition-level averages. For the stochastic components of HDFBLS, the 25 fold evaluations use the deterministic seed sequence 2026, 2027, ..., 2050.

The file `cv_details.csv` provides the fold-level results for all 15 datasets under five repetitions of 5-fold stratified cross-validation, including the random seed, classification metrics, network structure, and computation time for each fold.

The public repository includes ORL as a directly runnable example. To run the remaining datasets reported in the paper, place their `.npz` files in `datasets_npz` using the names listed in `main.py`, and then enable the corresponding entries in `DATASET_NAMES`.

## Running baseline methods

The compared methods are organized in separate folders:

- `M1-FBLS`
- `M2-CFBLS`
- `M3-ADHFBLS`
- `M4-IFBLS`
- `M5-HDRBFNN`
- `M6-FSRE-AdaTSK`

These folders contain the corresponding baseline implementations or reproduced code used for comparison. 

For M1, M2, and M4, the outer protocol consists of five repetitions of
5-fold stratified cross-validation. Within each outer training fold, a single
stratified 80/20 split is used for parameter selection: 80% of the samples are
used to fit each candidate configuration and 20% are used for validation. The
configuration with the highest validation accuracy is then refitted on the
complete outer training fold and evaluated once on the outer test fold. Thus,
the outer test fold is used only for final evaluation.

The corresponding entry points are:

```bash
python M1-FBLS/main_fbls_repeated_cv.py
python M2-CFBLS/main_cfbls_repeated_cv.py
```

For M4, open MATLAB with the repository as the current folder and run:

```matlab
addpath('M4-IFBLS');
run_ifbls_repeated_cv
```

M4 follows the original MATLAB implementation and also requires the legacy
helper `M4-IFBLS/ifbls_nested_fold.m`, which evaluates candidate models on a
given fitting/validation split; despite its filename, the current driver does
not perform inner cross-validation. Its datasets are read from
`datasets_HDFBLS` in MATLAB `.mat` format. The remaining Python scripts read
datasets from `datasets_npz`.

Each program saves the validation score for every candidate configuration,
the configuration selected in every outer fold, the final outer-fold results,
and the repetition-level summary in a method-specific results folder. The
reported total computation time includes parameter selection, refitting on the
complete outer training fold, and final evaluation.

## Evaluation metrics

The main classification metrics used in the paper are:

- Accuracy
- Balanced accuracy
- Macro-F1

These metrics are computed using the corresponding functions in the Python `scikit-learn` library.

## Reproducibility notes

To improve reproducibility, please check the following settings before running the experiments:

1. Use the same dataset files as those reported in the paper.
2. Keep the same cross-validation protocol and random seeds as specified in `main.py`.
3. Use the same parameter settings for HDFBLS and the compared methods.
4. Make sure that all methods are evaluated on identical data partitions.

Some conventional FBLS variants may fail on high-dimensional datasets because numerical underflow can produce zero firing strengths and NaN values during firing-strength normalization. This behavior is consistent with the numerical mechanism discussed in the paper.

## Contact

For questions about the code or experiments, please contact the authors through the corresponding author information provided in the paper.
