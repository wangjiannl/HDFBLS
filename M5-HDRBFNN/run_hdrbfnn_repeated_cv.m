function run_hdrbfnn_repeated_cv()
%RUN_HDRBFNN_REPEATED_CV
% Single-file HDRBFNN experiment under a unified evaluation protocol.
%
% Protocol
%   1) 5-fold repeated stratified cross-validation, repeated 5 times.
%   2) Min-max scaling is fitted on the training fold only.
%   3) Labels are encoded as 0/1 one-hot vectors for model training.
%   4) Accuracy, balanced accuracy, and macro-F1 are reported.
%   5) Numerical failures are recorded and do not stop later folds/datasets.
%   6) Preprocessing, training, testing, and total fold time are recorded.
%
% The HDRBFNN model equations and training procedure are preserved from the
% supplied MATLAB implementation:
%   - number of centers = round(N_train/2)
%   - centers initialized from training samples and refined by 100 k-means epochs
%   - adaptive width from RBFN_base
%   - output weights trained by JRMOCD (GD.m)
%
% Required toolbox functions used by the original implementation:
%   cvpartition, pdist2

clc;
format compact;

%% ======================== Configuration ========================
N_FOLDS = 5;
N_REPEATS = 5;
BASE_SEED = 2026;

I_VALUE = 0.01;       % Preserved from the supplied HDRBFNN main program
KMEANS_EPOCHS = 100;  % Preserved from the supplied code
UNDERFLOW_XI = 743;

DATA_DIR = 'datasets_HDFBLS';
RESULT_DIR = 'results_hdrbfnn_cv';
if ~exist(RESULT_DIR, 'dir')
    mkdir(RESULT_DIR);
end

DETAIL_FILE = fullfile(RESULT_DIR, 'hdrbfnn_cv_details.csv');
SUMMARY_FILE = fullfile(RESULT_DIR, 'hdrbfnn_cv_summary.csv');
DATASET_TIME_FILE = fullfile(RESULT_DIR, 'hdrbfnn_cv_dataset_times.csv');

% Canonical dataset names and accepted file-name aliases.
DATASETS = {
    'PAGEB',        {'PAGEB'};
    'Thyroid',      {'Thyroid'};
    'SATELLITE',    {'SATELLITE'};
    'TEXTURE',      {'TEXTURE'};
    'spambase',     {'spambase'};
    'ORL',          {'ORL'};
    'semg',         {'semg'};
    'warpAR10P',    {'warpAR10P', 'warpAR10P (1)'};
    'PIE',          {'PIE'};
    'handoutlines', {'handoutlines'};
    'Giesste',      {'Giesste', 'Gisette'};
    'Drivace',      {'Drivace'};
    'Leukemia',     {'Leukemia', 'LEUKEMIA'};
    'CMHS',         {'CMHS', 'CMHSCLA'};
    'GSAFM',        {'GSAFM', 'GSAFM4'};
};

DETAIL_HEADER = {
    'Dataset','Repeat','Fold','Seed','Status','ErrorMessage', ...
    'TrainSamples','TestSamples','Features','Classes', ...
    'NumCenters','MaxIter','IValue','LambdaInternal','CP','CM','Beta', ...
    'TrainACC','TrainBACC','TrainMacroF1', ...
    'TestACC','TestBACC','TestMacroF1', ...
    'TrainZeroActivationPct','TrainAllZeroRowPct', ...
    'TestZeroActivationPct','TestAllZeroRowPct', ...
    'PreprocessTime','TrainTime','TestTime','FoldTime'};

SUMMARY_HEADER = {
    'Dataset','SuccessfulFolds','FailedFolds', ...
    'ACCMean','ACCStd','BACCMean','BACCStd','MacroF1Mean','MacroF1Std', ...
    'NumCentersMean','TrainTimeMean','TrainTimeStd', ...
    'TestTimeMean','TestTimeStd','FoldTimeMean','FoldTimeStd','DatasetTime'};

TIME_HEADER = {'Dataset','DatasetTime','SuccessfulFolds','FailedFolds'};

detail_rows = cell(0, numel(DETAIL_HEADER));
summary_rows = cell(0, numel(SUMMARY_HEADER));
time_rows = cell(0, numel(TIME_HEADER));

write_cell_csv(DETAIL_FILE, DETAIL_HEADER, detail_rows);
write_cell_csv(SUMMARY_FILE, SUMMARY_HEADER, summary_rows);
write_cell_csv(DATASET_TIME_FILE, TIME_HEADER, time_rows);

%% ======================== Main experiment ========================
for d = 1:size(DATASETS, 1)
    dataset_name = DATASETS{d, 1};
    aliases = DATASETS{d, 2};
    dataset_timer = tic;

    fprintf('\n============================================================\n');
    fprintf('Dataset: %s\n', dataset_name);
    fprintf('============================================================\n');

    try
        [X_all, y_all] = load_dataset(DATA_DIR, aliases);
        X_all = double(X_all);
        y_all = encode_labels(y_all);

        if any(~isfinite(X_all(:)))
            error('Input data contain NaN or Inf before preprocessing.');
        end

        n_samples = size(X_all, 1);
        n_features = size(X_all, 2);
        n_classes = max(y_all);
        class_counts = accumarray(y_all, 1, [n_classes, 1]);

        if min(class_counts) < N_FOLDS
            error('At least one class has fewer than %d samples.', N_FOLDS);
        end
    catch ME
        dataset_time = toc(dataset_timer);
        fprintf('[Skipped] %s: %s\n', dataset_name, ME.message);
        summary_rows(end+1, :) = {dataset_name, 0, N_FOLDS*N_REPEATS, ...
            NaN,NaN,NaN,NaN,NaN,NaN,NaN,NaN,NaN,NaN,dataset_time};
        time_rows(end+1, :) = {dataset_name, dataset_time, 0, N_FOLDS*N_REPEATS};
        write_cell_csv(SUMMARY_FILE, SUMMARY_HEADER, summary_rows);
        write_cell_csv(DATASET_TIME_FILE, TIME_HEADER, time_rows);
        continue;
    end

    dataset_detail_start = size(detail_rows, 1) + 1;
    successful_folds = 0;
    failed_folds = 0;

    for rep = 1:N_REPEATS
        rng(BASE_SEED + rep - 1, 'twister');
        cvp = cvpartition(y_all, 'KFold', N_FOLDS);

        for fold = 1:N_FOLDS
            fold_timer = tic;
            seed = BASE_SEED + (rep - 1) * N_FOLDS + fold - 1;
            rng(seed, 'twister');

            status = 'success';
            error_message = '';

            % Default values retained for failed-fold logging.
            train_n = NaN; test_n = NaN;
            numcenters = NaN; maxiter = NaN;
            lambda_internal = NaN; CP = NaN; CM = NaN; beta = NaN;
            train_acc = NaN; train_bacc = NaN; train_macro_f1 = NaN;
            test_acc = NaN; test_bacc = NaN; test_macro_f1 = NaN;
            train_zero_pct = NaN; train_allzero_pct = NaN;
            test_zero_pct = NaN; test_allzero_pct = NaN;
            preprocess_time = NaN; train_time = NaN; test_time = NaN;

            try
                train_idx = training(cvp, fold);
                test_idx = test(cvp, fold);

                X_train_raw = X_all(train_idx, :);
                X_test_raw = X_all(test_idx, :);
                y_train_idx = y_all(train_idx);
                y_test_idx = y_all(test_idx);

                train_n = size(X_train_raw, 1);
                test_n = size(X_test_raw, 1);

                preprocess_timer = tic;
                [X_train, X_test] = minmax_train_test(X_train_raw, X_test_raw);
                Y_train = one_hot(y_train_idx, n_classes);
                preprocess_time = toc(preprocess_timer);

                if any(~isfinite(X_train(:))) || any(~isfinite(X_test(:)))
                    error('Non-finite value detected after min-max scaling.');
                end

                maxiter = round(max(2 * n_features, 2000));
                numcenters = round(train_n / 2);
                numcenters = max(1, min(numcenters, train_n));

                train_timer = tic;
                initial_indices = randperm(train_n, numcenters);
                center_init = X_train(initial_indices, :);
                Centers = k_means_local(X_train, center_init, KMEANS_EPOCHS);

                Distance = pdist2(X_train, Centers, 'squaredeuclidean');
                CP = mean(Distance(:));
                CM = max(Distance(:));

                if ~isfinite(CP) || ~isfinite(CM) || CM <= 0
                    error('Invalid CP or CM encountered when constructing HDRBFNN.');
                end

                lambda_internal = (CM - I_VALUE * CP) * UNDERFLOW_XI / CM;
                [beta, T_train] = rbfn_base_local( ...
                    numcenters, CP, X_train, Centers, ...
                    lambda_internal, Distance, UNDERFLOW_XI);

                assert_finite(T_train, 'Training activation matrix');
                [train_zero_pct, train_allzero_pct] = activation_statistics(T_train);

                WeightTopCD = gd_jrmocd_local(T_train, Y_train, numcenters, maxiter);
                assert_finite(WeightTopCD, 'Output-weight matrix');

                train_scores = T_train * WeightTopCD;
                assert_finite(train_scores, 'Training output scores');
                [~, y_train_pred] = max(train_scores, [], 2);
                [train_acc, train_bacc, train_macro_f1] = ...
                    classification_metrics(y_train_idx, y_train_pred, n_classes);
                train_time = toc(train_timer);

                test_timer = tic;
                T_test = rbf_activations_local(Centers, beta, X_test);
                assert_finite(T_test, 'Test activation matrix');
                [test_zero_pct, test_allzero_pct] = activation_statistics(T_test);

                test_scores = T_test * WeightTopCD;
                assert_finite(test_scores, 'Test output scores');
                [~, y_test_pred] = max(test_scores, [], 2);
                [test_acc, test_bacc, test_macro_f1] = ...
                    classification_metrics(y_test_idx, y_test_pred, n_classes);
                test_time = toc(test_timer);

                metric_vector = [train_acc, train_bacc, train_macro_f1, ...
                    test_acc, test_bacc, test_macro_f1];
                if any(~isfinite(metric_vector))
                    error('At least one evaluation metric is NaN or Inf.');
                end

                successful_folds = successful_folds + 1;

            catch ME
                status = 'failed';
                error_message = sprintf('%s: %s', class(ME), ME.message);
                failed_folds = failed_folds + 1;
                fprintf('[Failed] %s | Repeat %d | Fold %d | %s\n', ...
                    dataset_name, rep, fold, error_message);
            end

            fold_time = toc(fold_timer);

            detail_rows(end+1, :) = {
                dataset_name, rep, fold, seed, status, error_message, ...
                train_n, test_n, n_features, n_classes, ...
                numcenters, maxiter, I_VALUE, lambda_internal, CP, CM, beta, ...
                train_acc, train_bacc, train_macro_f1, ...
                test_acc, test_bacc, test_macro_f1, ...
                train_zero_pct, train_allzero_pct, ...
                test_zero_pct, test_allzero_pct, ...
                preprocess_time, train_time, test_time, fold_time};

            % Save immediately so that a server interruption does not erase
            % completed folds.
            write_cell_csv(DETAIL_FILE, DETAIL_HEADER, detail_rows);

            fprintf(['%s | Rep %d/%d | Fold %d/%d | ACC=%.4f | ', ...
                'BACC=%.4f | Macro-F1=%.4f | Centers=%g | Time=%.2fs\n'], ...
                dataset_name, rep, N_REPEATS, fold, N_FOLDS, ...
                test_acc, test_bacc, test_macro_f1, numcenters, fold_time);
        end
    end

    dataset_time = toc(dataset_timer);
    dataset_rows = detail_rows(dataset_detail_start:end, :);

    acc_values = cell_numeric_column(dataset_rows, 21);
    bacc_values = cell_numeric_column(dataset_rows, 22);
    f1_values = cell_numeric_column(dataset_rows, 23);
    center_values = cell_numeric_column(dataset_rows, 11);
    train_time_values = cell_numeric_column(dataset_rows, 29);
    test_time_values = cell_numeric_column(dataset_rows, 30);
    fold_time_values = cell_numeric_column(dataset_rows, 31);

    [acc_mean, acc_std] = repeated_cv_mean_std( ...
        acc_values, N_FOLDS, N_REPEATS);
    [bacc_mean, bacc_std] = repeated_cv_mean_std( ...
        bacc_values, N_FOLDS, N_REPEATS);
    [f1_mean, f1_std] = repeated_cv_mean_std( ...
        f1_values, N_FOLDS, N_REPEATS);
    [center_mean, ~] = repeated_cv_mean_std( ...
        center_values, N_FOLDS, N_REPEATS);
    [train_time_mean, train_time_std] = repeated_cv_mean_std( ...
        train_time_values, N_FOLDS, N_REPEATS);
    [test_time_mean, test_time_std] = repeated_cv_mean_std( ...
        test_time_values, N_FOLDS, N_REPEATS);
    [fold_time_mean, fold_time_std] = repeated_cv_mean_std( ...
        fold_time_values, N_FOLDS, N_REPEATS);

    summary_rows(end+1, :) = {
        dataset_name, successful_folds, failed_folds, ...
        acc_mean, acc_std, bacc_mean, bacc_std, f1_mean, f1_std, ...
        center_mean, train_time_mean, train_time_std, ...
        test_time_mean, test_time_std, fold_time_mean, fold_time_std, ...
        dataset_time};

    time_rows(end+1, :) = {dataset_name, dataset_time, successful_folds, failed_folds};

    write_cell_csv(SUMMARY_FILE, SUMMARY_HEADER, summary_rows);
    write_cell_csv(DATASET_TIME_FILE, TIME_HEADER, time_rows);

    fprintf('\n>>> %s SUMMARY\n', dataset_name);
    fprintf('ACC       = %.4f +/- %.4f\n', acc_mean, acc_std);
    fprintf('BACC      = %.4f +/- %.4f\n', bacc_mean, bacc_std);
    fprintf('Macro-F1  = %.4f +/- %.4f\n', f1_mean, f1_std);
    fprintf('Successful folds: %d | Failed folds: %d\n', successful_folds, failed_folds);
    fprintf('Dataset time: %.2f s\n', dataset_time);
end

fprintf('\nAll HDRBFNN experiments finished.\n');
fprintf('Details: %s\n', DETAIL_FILE);
fprintf('Summary: %s\n', SUMMARY_FILE);
fprintf('Dataset times: %s\n', DATASET_TIME_FILE);
end

%% ======================== Local functions ========================
function [X, y] = load_dataset(data_dir, aliases)
file_path = '';
for i = 1:numel(aliases)
    candidate = fullfile(data_dir, [aliases{i}, '.mat']);
    if exist(candidate, 'file')
        file_path = candidate;
        break;
    end
end

if isempty(file_path)
    error('No matching MAT file found in %s.', data_dir);
end

ws = load(file_path);

if isfield(ws, 'X') && isfield(ws, 'Y')
    X = ws.X;
    y = ws.Y;
elseif isfield(ws, 'data') && isfield(ws, 'label')
    X = ws.data;
    y = ws.label;
elseif isfield(ws, 'sample') && isfield(ws, 'target')
    X = ws.sample;
    y = ws.target;
else
    names = fieldnames(ws);
    if numel(names) == 1
        combined = ws.(names{1});
        if size(combined, 2) < 2
            error('Single variable does not contain both features and labels.');
        end
        X = combined(:, 1:end-1);
        y = combined(:, end);
    else
        error('Unrecognized MAT-file variable layout.');
    end
end

% Accept either a label vector or an N-by-C one-hot/score matrix.
if ~isvector(y) && size(y, 2) > 1
    if size(y, 1) == size(X, 1)
        [~, y] = max(y, [], 2);
    elseif size(y, 2) == size(X, 1)
        [~, y] = max(y, [], 1);
        y = y';
    else
        error('Label matrix dimensions do not match the feature matrix.');
    end
end

y = y(:);
if size(X, 1) ~= numel(y)
    if size(X, 2) == numel(y)
        X = X';
    else
        error('Feature and label sample counts do not match.');
    end
end
end

function y_idx = encode_labels(y)
if isnumeric(y) || islogical(y)
    [~, ~, y_idx] = unique(y(:));
elseif iscategorical(y)
    [~, ~, y_idx] = unique(cellstr(y(:)));
elseif isstring(y)
    [~, ~, y_idx] = unique(cellstr(y(:)));
elseif iscell(y)
    [~, ~, y_idx] = unique(y(:));
else
    error('Unsupported label data type.');
end
y_idx = double(y_idx(:));
end

function Y = one_hot(y_idx, n_classes)
n = numel(y_idx);
Y = zeros(n, n_classes);
linear_idx = sub2ind([n, n_classes], (1:n)', y_idx(:));
Y(linear_idx) = 1;
end

function [X_train, X_test] = minmax_train_test(X_train_raw, X_test_raw)
x_min = min(X_train_raw, [], 1);
x_max = max(X_train_raw, [], 1);
x_range = x_max - x_min;
x_range(x_range == 0) = 1;

X_train = bsxfun(@rdivide, bsxfun(@minus, X_train_raw, x_min), x_range);
X_test = bsxfun(@rdivide, bsxfun(@minus, X_test_raw, x_min), x_range);

% Match the HDFBLS preprocessing protocol.
X_train = min(max(X_train, 0), 1);
X_test = min(max(X_test, 0), 1);
end

function Centers = k_means_local(dataset, Centers, num_epochs)
num_centers = size(Centers, 1);
for epoch = 1:num_epochs
    dist = pdist2(dataset, Centers, 'squaredeuclidean');
    [~, assignment] = min(dist, [], 2);
    for j = 1:num_centers
        idx = assignment == j;
        if any(idx)
            Centers(j, :) = mean(dataset(idx, :), 1);
        end
    end
end
end

function [beta, T] = rbfn_base_local(numcenters, CP, train_x, Centers, I, Distance, xi)
n_features = size(train_x, 2);
denominator = 2 * (xi - I);
if denominator <= 0
    error('Invalid width denominator: 2*(xi-I) must be positive.');
end

Dmin = sqrt(CP / denominator);
center_distance = pdist2(Centers, Centers);
Cmax = max(center_distance(:));
beta11 = Cmax / sqrt(2 * numcenters);
dimension_factor = tanh(0.001 * n_features);
beta = (beta11 + dimension_factor * Dmin)^2;

if ~isfinite(beta) || beta <= 0
    error('The adaptive Gaussian width is non-positive or non-finite.');
end

T = exp(-Distance / (2 * beta));
end

function T = rbf_activations_local(Centers, beta, X)
Distance = pdist2(X, Centers, 'squaredeuclidean');
T = exp(-Distance / (2 * beta));
end

function WeightTopCD = gd_jrmocd_local(T, train_y, numcenters, maxiter)
n_outputs = size(train_y, 2);
WeightTopCD = 0.1 * rand(numcenters, n_outputs);

B = T' * T;
residual = train_y - T * WeightTopCD;
s = T' * residual;

assert_finite(B, 'JRMOCD Gram matrix');
assert_finite(s, 'JRMOCD initial residual correlation');

for iteration = 1:maxiter
    sj = sum(s.^2, 2);
    [~, j] = max(sj);
    denominator = B(j, j);

    if ~isfinite(denominator) || denominator <= eps(max(1, max(abs(diag(B)))))
        error('JRMOCD selected a zero or numerically singular diagonal entry.');
    end

    miu = s(j, :) / denominator;
    s = s - B(:, j) * miu;
    WeightTopCD(j, :) = WeightTopCD(j, :) + miu;

    if any(~isfinite(miu)) || any(~isfinite(s(:))) || any(~isfinite(WeightTopCD(:)))
        error('JRMOCD produced NaN or Inf at iteration %d.', iteration);
    end
end
end

function [acc, bacc, macro_f1] = classification_metrics(y_true, y_pred, n_classes)
y_true = y_true(:);
y_pred = y_pred(:);

if numel(y_true) ~= numel(y_pred)
    error('Prediction and target lengths do not match.');
end
if any(y_pred < 1) || any(y_pred > n_classes)
    error('Predicted class index is outside the valid class range.');
end

cm = accumarray([y_true, y_pred], 1, [n_classes, n_classes]);
acc = sum(diag(cm)) / sum(cm(:));

recall_den = sum(cm, 2);
precision_den = sum(cm, 1)';
recall = zeros(n_classes, 1);
precision = zeros(n_classes, 1);

valid_recall = recall_den > 0;
valid_precision = precision_den > 0;
recall(valid_recall) = diag(cm(valid_recall, valid_recall)) ./ recall_den(valid_recall);
precision(valid_precision) = diag(cm(valid_precision, valid_precision)) ./ precision_den(valid_precision);

f1 = zeros(n_classes, 1);
valid_f1 = (precision + recall) > 0;
f1(valid_f1) = 2 * precision(valid_f1) .* recall(valid_f1) ./ ...
    (precision(valid_f1) + recall(valid_f1));

% With stratified folds every class is present; averaging all classes then
% matches the intended balanced-accuracy and macro-F1 definitions.
bacc = mean(recall);
macro_f1 = mean(f1);
end

function [zero_pct, allzero_row_pct] = activation_statistics(T)
zero_pct = mean(T(:) == 0) * 100;
allzero_row_pct = mean(all(T == 0, 2)) * 100;
end

function assert_finite(A, name)
if any(~isfinite(A(:)))
    error('%s contains NaN or Inf.', name);
end
end

function values = cell_numeric_column(rows, column_index)
if isempty(rows)
    values = [];
    return;
end
values = NaN(size(rows, 1), 1);
for i = 1:size(rows, 1)
    value = rows{i, column_index};
    if isnumeric(value) && isscalar(value)
        values(i) = value;
    end
end
end

function value = finite_mean(values)
values = values(isfinite(values));
if isempty(values)
    value = NaN;
else
    value = mean(values);
end
end

function [mean_value, std_value] = finite_mean_std(values)
values = values(isfinite(values));
if isempty(values)
    mean_value = NaN;
    std_value = NaN;
elseif numel(values) == 1
    mean_value = values(1);
    std_value = 0;
else
    mean_value = mean(values);
    std_value = std(values, 0);
end
end

function [mean_value, std_value] = repeated_cv_mean_std( ...
    values, n_splits, n_repeats)
% Average folds within each repetition, then summarize repetitions.
values = values(:);
expected = n_splits * n_repeats;
if numel(values) ~= expected
    error('HDRBFNN:InvalidFoldCount', ...
        'Expected %d fold values, but received %d.', ...
        expected, numel(values));
end

fold_matrix = reshape(values, n_splits, n_repeats);
repetition_means = nan(n_repeats, 1);
for repeat_id = 1:n_repeats
    repeat_values = fold_matrix(:, repeat_id);
    if all(isfinite(repeat_values))
        repetition_means(repeat_id) = mean(repeat_values);
    end
end
[mean_value, std_value] = finite_mean_std(repetition_means);
end

function write_cell_csv(path, header, rows)
output = [header; rows];
writecell(output, path);
end
