function run_tsf_fsre_lse_repeated_cv()
% RUN_TSF_FSRE_LSE_REPEATED_CV
% Self-contained evaluation program for the TSFNN feature-selection and
% rule-extraction model. The final consequent parameters are estimated by
% the faster regularized LSE solver only; the alternative gradient-descent
% fine-tuning branch is intentionally omitted.
%
% Fair evaluation protocol:
%   1) 5-fold repeated stratified cross-validation, repeated 5 times;
%   2) min-max scaling fitted on each training fold only;
%   3) common 0/1 one-hot targets;
%   4) ACC, balanced accuracy, and macro-F1;
%   5) numerical failures are recorded and do not stop later folds/datasets;
%   6) per-fold and per-dataset running times are recorded.

clc;
format compact;

%% -------------------- Experiment configuration --------------------
CV_SPLITS = 5;
CV_REPEATS = 5;
RANDOM_SEED = 2026;

DATA_DIR = 'datasets_HDFBLS';
RESULT_DIR = 'results_tsf_fsre_lse_cv';
DETAIL_FILE = fullfile(RESULT_DIR, 'tsf_fsre_lse_details.csv');
SUMMARY_FILE = fullfile(RESULT_DIR, 'tsf_fsre_lse_summary.csv');
DATASET_TIME_FILE = fullfile(RESULT_DIR, 'tsf_fsre_lse_dataset_times.csv');

if ~exist(RESULT_DIR, 'dir')
    mkdir(RESULT_DIR);
end

DATASET_NAMES = { ...
    'PAGEB', 'Thyroid', 'SATELLITE', 'TEXTURE', 'spambase', ...
    'ORL', 'semg', 'warpAR10P (1)', 'PIE', 'handoutlines', ...
    'Giesste', 'Drivace', 'Leukemia', 'CMHS', 'GSAFM'};

% Set true for a quick server-side smoke test.
RUN_SMALL_ONLY = false;
SMALL_DATASETS = {'Leukemia', 'GSAFM'};
if RUN_SMALL_ONLY
    DATASET_NAMES = SMALL_DATASETS;
end

%% -------------------- Original model settings --------------------
ETA = 0.2;
MAX_ITER_FEATURE = 300;
MAX_ITER_RULE = 300;
CO_TAU_FEATURE = 0.5;
CO_TAU_RULE = 0.3;
GAMMA = 0.1;
NUM_MF_FEATURE = 10;
NUM_MF_RULE = 5;
SOLVER_NAME = 'LSE';

% Safety limit for the explicit rule-index matrix. This does not change a
% successful run. It only prevents the MATLAB process from being terminated
% by an impossible allocation; that fold is recorded as failed instead.
MAX_RULE_INDEX_ELEMENTS = 5e7;

DETAIL_HEADERS = { ...
    'dataset','repeat','fold','seed','solver','status','error_message', ...
    'n_train','n_test','n_classes','n_features_original', ...
    'n_features_selected','n_rules_extracted', ...
    'accuracy','balanced_accuracy','macro_f1', ...
    'feature_selection_time','rule_extraction_time','lse_time', ...
    'train_time','test_time','total_time', ...
    'train_raw_zero_percent','train_normalized_zero_percent', ...
    'train_raw_all_zero_rows_percent','train_normalized_all_zero_rows_percent', ...
    'test_raw_zero_percent','test_normalized_zero_percent', ...
    'test_raw_all_zero_rows_percent','test_normalized_all_zero_rows_percent'};

SUMMARY_HEADERS = { ...
    'dataset','valid_folds','failed_folds', ...
    'accuracy_mean','accuracy_std', ...
    'balanced_accuracy_mean','balanced_accuracy_std', ...
    'macro_f1_mean','macro_f1_std', ...
    'selected_features_mean','extracted_rules_mean', ...
    'feature_selection_time_mean','rule_extraction_time_mean', ...
    'lse_time_mean','train_time_mean','test_time_mean', ...
    'total_time_mean','total_time_std', ...
    'dataset_time'};

TIME_HEADERS = {'dataset','dataset_time','valid_folds','failed_folds'};

detail_rows = cell(0, numel(DETAIL_HEADERS));
summary_rows = cell(0, numel(SUMMARY_HEADERS));
time_rows = cell(0, numel(TIME_HEADERS));

%% -------------------- Main experiment loop --------------------
for dataset_idx = 1:numel(DATASET_NAMES)
    dataset_name = DATASET_NAMES{dataset_idx};
    dataset_clock = tic;
    fprintf('\n============================================================\n');
    fprintf('Dataset: %s (%d/%d)\n', dataset_name, dataset_idx, numel(DATASET_NAMES));
    fprintf('============================================================\n');

    try
        [X_all, y_raw, resolved_path] = load_dataset_robust(DATA_DIR, dataset_name);
        fprintf('Loaded: %s\n', resolved_path);

        X_all = double(X_all);
        if any(~isfinite(X_all(:)))
            error('Input data contain NaN or Inf.');
        end

        [y_all, class_values] = encode_labels(y_raw);
        n_classes = numel(class_values);
        class_counts = accumarray(y_all, 1, [n_classes, 1]);
        if min(class_counts) < CV_SPLITS
            error('The smallest class has only %d samples, fewer than CV_SPLITS=%d.', ...
                min(class_counts), CV_SPLITS);
        end
    catch ME
        dataset_time = toc(dataset_clock);
        fprintf('[Dataset skipped] %s: %s\n', dataset_name, ME.message);
        time_rows(end+1,:) = {dataset_name, dataset_time, 0, CV_SPLITS * CV_REPEATS}; %#ok<AGROW>
        write_cell_table(DATASET_TIME_FILE, TIME_HEADERS, time_rows);
        continue;
    end

    dataset_detail_start = size(detail_rows, 1) + 1;

    for repeat_id = 1:CV_REPEATS
        repeat_seed = RANDOM_SEED + repeat_id - 1;
        fold_id = stratified_fold_ids(y_all, CV_SPLITS, repeat_seed);

        for fold = 1:CV_SPLITS
            fold_seed = RANDOM_SEED + (repeat_id - 1) * CV_SPLITS + fold - 1;
            fold_clock = tic;

            status = 'success';
            error_message = '';
            acc = NaN;
            bacc = NaN;
            macro_f1 = NaN;
            n_selected = NaN;
            n_rules = NaN;
            feature_time = NaN;
            rule_time = NaN;
            lse_time = NaN;
            train_time = NaN;
            test_time = NaN;
            train_stats = nan(1,4);
            test_stats = nan(1,4);

            test_idx = (fold_id == fold);
            train_idx = ~test_idx;
            X_train_raw = X_all(train_idx, :);
            X_test_raw = X_all(test_idx, :);
            y_train = y_all(train_idx);
            y_test = y_all(test_idx);

            n_train = size(X_train_raw, 1);
            n_test = size(X_test_raw, 1);
            n_features_original = size(X_train_raw, 2);

            try
                rng(fold_seed, 'twister');

                [x_min, x_range] = fit_minmax(X_train_raw);
                X_train = transform_minmax(X_train_raw, x_min, x_range);
                X_test = transform_minmax(X_test_raw, x_min, x_range);
                X_train = min(max(X_train, 0), 1);
                X_test = min(max(X_test, 0), 1);

                Y_train = one_hot_indices(y_train, n_classes);

                train_clock = tic;

                %% Phase 1: feature selection
                phase_clock = tic;
                [Mean_FS, P_FS, IndFire_FS, lambda] = train_fsre( ...
                    X_train, Y_train, NUM_MF_FEATURE, ETA, ...
                    MAX_ITER_FEATURE, 1, MAX_RULE_INDEX_ELEMENTS);

                gate_feature = abs(gate_fun(lambda));
                tau_feature = max(gate_feature) - ...
                    (max(gate_feature) - min(gate_feature)) * CO_TAU_FEATURE;
                [X_train_selected, Mean_FS, P_FS, IndFire_FS, selected_features] = ...
                    prune_features(X_train, Mean_FS, P_FS, IndFire_FS, ...
                    lambda, tau_feature); %#ok<ASGLU>
                X_test_selected = X_test(:, selected_features);
                n_selected = numel(selected_features);
                if n_selected == 0
                    error('All features were removed during feature selection.');
                end
                feature_time = toc(phase_clock);

                %% Phase 2: rule extraction
                estimated_rule_rows = estimate_rule_count(n_selected, NUM_MF_RULE);
                estimated_rule_elements = estimated_rule_rows * n_selected;
                if estimated_rule_elements > MAX_RULE_INDEX_ELEMENTS
                    error(['Estimated rule-index matrix contains %.3e elements, ' ...
                        'exceeding the safety limit %.3e.'], ...
                        estimated_rule_elements, MAX_RULE_INDEX_ELEMENTS);
                end

                phase_clock = tic;
                [Mean_RE, P_RE, IndFire_RE, ~, theta] = train_fsre( ...
                    X_train_selected, Y_train, NUM_MF_RULE, ETA, ...
                    MAX_ITER_RULE, 2, MAX_RULE_INDEX_ELEMENTS);

                gate_rule = abs(gate_fun(theta));
                tau_rule = max(gate_rule) - ...
                    (max(gate_rule) - min(gate_rule)) * CO_TAU_RULE;
                [IndFire_RE, P_RE, extracted_rules] = prune_rules( ...
                    IndFire_RE, theta, P_RE, tau_rule);
                n_rules = numel(extracted_rules);
                if n_rules == 0
                    error('All rules were removed during rule extraction.');
                end
                rule_time = toc(phase_clock);

                %% Phase 3: fast LSE consequent estimation only
                phase_clock = tic;
                [Mean_final, P_final] = train_lse( ...
                    X_train_selected, Y_train, Mean_RE, IndFire_RE, GAMMA, ...
                    MAX_RULE_INDEX_ELEMENTS);
                lse_time = toc(phase_clock);
                train_time = toc(train_clock);

                if any(~isfinite(Mean_final(:))) || any(~isfinite(P_final(:)))
                    error('Non-finite final model parameters detected.');
                end

                %% Evaluation
                [train_scores, train_stats] = model_forward( ...
                    X_train_selected, n_classes, Mean_final, P_final, IndFire_RE);

                test_clock = tic;
                [test_scores, test_stats] = model_forward( ...
                    X_test_selected, n_classes, Mean_final, P_final, IndFire_RE);
                test_time = toc(test_clock);

                % Preserve the decision rule of the supplied ClaInd.m:
                % choose the class with the largest absolute output.
                [~, y_pred] = max(abs(test_scores), [], 2);
                [acc, bacc, macro_f1] = classification_metrics( ...
                    y_test, y_pred, n_classes);

                metric_values = [acc, bacc, macro_f1];
                if any(~isfinite(metric_values))
                    error('Non-finite evaluation metric detected.');
                end

                if any(~isfinite(train_scores(:))) || any(~isfinite(test_scores(:)))
                    error('Non-finite model outputs detected.');
                end

            catch ME
                status = 'failed';
                error_message = sprintf('%s: %s', class(ME), ME.message);
                fprintf('[Failed] %s | repeat=%d fold=%d | %s\n', ...
                    dataset_name, repeat_id, fold, error_message);
            end

            total_time = toc(fold_clock);

            detail_rows(end+1,:) = { ... %#ok<AGROW>
                dataset_name, repeat_id, fold, fold_seed, SOLVER_NAME, ...
                status, error_message, n_train, n_test, n_classes, ...
                n_features_original, n_selected, n_rules, ...
                acc, bacc, macro_f1, feature_time, rule_time, lse_time, ...
                train_time, test_time, total_time, ...
                train_stats(1), train_stats(2), train_stats(3), train_stats(4), ...
                test_stats(1), test_stats(2), test_stats(3), test_stats(4)};

            write_cell_table(DETAIL_FILE, DETAIL_HEADERS, detail_rows);

            if strcmp(status, 'success')
                fprintf(['[%s] repeat=%d/%d fold=%d/%d | ACC=%.4f | ' ...
                    'BACC=%.4f | Macro-F1=%.4f | Features=%d | Rules=%d | ' ...
                    'Time=%.2fs\n'], dataset_name, repeat_id, CV_REPEATS, ...
                    fold, CV_SPLITS, acc, bacc, macro_f1, ...
                    n_selected, n_rules, total_time);
            end
        end
    end

    dataset_time = toc(dataset_clock);
    dataset_rows = detail_rows(dataset_detail_start:end, :);
    summary = summarize_dataset_rows( ...
        dataset_name, dataset_rows, dataset_time, CV_SPLITS, CV_REPEATS);
    summary_rows(end+1,:) = summary; %#ok<AGROW>

    valid_folds = summary{2};
    failed_folds = summary{3};
    time_rows(end+1,:) = {dataset_name, dataset_time, valid_folds, failed_folds}; %#ok<AGROW>

    write_cell_table(SUMMARY_FILE, SUMMARY_HEADERS, summary_rows);
    write_cell_table(DATASET_TIME_FILE, TIME_HEADERS, time_rows);

    fprintf('\n>>> FINAL %s | ACC=%.4f+-%.4f | BACC=%.4f+-%.4f | ', ...
        dataset_name, summary{4}, summary{5}, summary{6}, summary{7});
    fprintf('Macro-F1=%.4f+-%.4f | Failed=%d/%d | Time=%.2fs\n', ...
        summary{8}, summary{9}, failed_folds, CV_SPLITS*CV_REPEATS, dataset_time);
end

fprintf('\nAll experiments finished.\n');
fprintf('Details: %s\n', DETAIL_FILE);
fprintf('Summary: %s\n', SUMMARY_FILE);
fprintf('Dataset times: %s\n', DATASET_TIME_FILE);
end

%% ========================================================================
%                              Data utilities
% ========================================================================
function [X, y, resolved_path] = load_dataset_robust(data_dir, canonical_name)
name_candidates = dataset_aliases(canonical_name);
resolved_path = '';
for i = 1:numel(name_candidates)
    candidate = fullfile(data_dir, [name_candidates{i}, '.mat']);
    if exist(candidate, 'file')
        resolved_path = candidate;
        break;
    end
end
if isempty(resolved_path)
    error('No matching MAT file was found for dataset %s.', canonical_name);
end

ws = load(resolved_path);
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
    vars = fieldnames(ws);
    if numel(vars) == 1
        combined = ws.(vars{1});
        X = combined(:, 1:end-1);
        y = combined(:, end);
    else
        error('Unrecognized variables in %s.', resolved_path);
    end
end

if size(y, 2) > 1 && size(y, 1) > 1
    [~, y] = max(y, [], 2);
else
    y = y(:);
end

if size(X,1) ~= numel(y)
    if size(X,2) == numel(y)
        X = X';
    else
        error('Sample count mismatch between X and y.');
    end
end
end

function names = dataset_aliases(name)
switch lower(name)
    case 'warpar10p (1)'
        names = {'warpAR10P (1)', 'warpAR10P'};
    case 'giesste'
        names = {'Giesste', 'Gisette'};
    case 'leukemia'
        names = {'Leukemia', 'LEUKEMIA'};
    case 'cmhs'
        names = {'CMHS', 'CMHSCLA'};
    case 'gsafm'
        names = {'GSAFM', 'GSAFM4'};
    otherwise
        names = {name};
end
end

function [y_idx, class_values] = encode_labels(y_raw)
y_raw = y_raw(:);
[class_values, ~, y_idx] = unique(y_raw, 'sorted');
y_idx = double(y_idx);
end

function fold_id = stratified_fold_ids(y, n_folds, seed)
rng(seed, 'twister');
fold_id = zeros(size(y));
classes = unique(y(:))';
for c = classes
    idx = find(y == c);
    idx = idx(randperm(numel(idx)));
    assignment = mod(0:numel(idx)-1, n_folds) + 1;
    fold_id(idx) = assignment(:);
end
end

function [x_min, x_range] = fit_minmax(X)
x_min = min(X, [], 1);
x_max = max(X, [], 1);
x_range = x_max - x_min;
x_range(x_range == 0) = 1;
end

function X_scaled = transform_minmax(X, x_min, x_range)
X_scaled = bsxfun(@rdivide, bsxfun(@minus, X, x_min), x_range);
end

function Y = one_hot_indices(y, n_classes)
Y = zeros(numel(y), n_classes);
linear_idx = sub2ind(size(Y), (1:numel(y))', y(:));
Y(linear_idx) = 1;
end

%% ========================================================================
%                 TSFNN feature/rule selection training
% ========================================================================
function [Mean, P, IndFire, lambda, theta] = train_fsre( ...
    X, Y, num_mf, eta, max_iter, phase, max_rule_index_elements)

C = size(Y, 2);
[N, D] = size(X);
Mean = zeros(num_mf, D);
for d = 1:D
    Mean(:, d) = linspace(min(X(:,d)), max(X(:,d)), num_mf);
end

if phase == 1
    IndFire = reshape(1:num_mf*D, num_mf, D);
elseif phase == 2
    estimated_elements = estimate_rule_count(D, num_mf) * D;
    if estimated_elements > max_rule_index_elements
        error('Rule-index matrix would contain %.3e elements.', estimated_elements);
    end
    IndFire = make_rule_index(D, num_mf);
else
    error('phase must be 1 or 2.');
end

R = size(IndFire, 1);
P = zeros(C*R, D+1);
lambda = 0.01 * ones(1, D);
theta = 0.01 * ones(R, 1);

for iter = 1:max_iter
    DelMean = zeros(size(Mean));
    DelP = zeros(size(P));
    if phase == 1
        DelGate = zeros(size(lambda));
    else
        DelGate = zeros(size(theta));
    end

    for k = 1:N
        Mu = exp(-(X(k,:) - Mean).^2);
        selected_mu = Mu(IndFire);
        min_mu = min(selected_mu, [], 2);
        q = get_q_approx(min_mu);
        FirStr = softmin_fun(selected_mu, q)';
        denom = sum(FirStr, 2);
        if ~isfinite(denom) || denom <= 0 || any(~isfinite(FirStr(:)))
            error('Invalid firing-strength normalization at iteration %d, sample %d.', iter, k);
        end
        FSbar = FirStr ./ denom;

        if phase == 1
            ysub = reshape(([1, gate_fun(lambda)] .* P) * [1, X(k,:)]', R, C)';
        else
            ysub = reshape(repmat(gate_fun(theta), C, 1) .* P * [1, X(k,:)]', R, C)';
        end
        yout = sum(ysub .* FSbar, 2)';

        for r = 1:R
            q_denom = sum(Mu(IndFire(r,:)).^q(r));
            if ~isfinite(q_denom) || q_denom == 0
                error('Invalid softmin derivative denominator.');
            end
            temp = FirStr(r) * Mu(IndFire(r,:)).^q(r) / q_denom .* ...
                (X(k,:) - Mean(IndFire(r,:))) ./ 0.5 .* ...
                sum((yout - Y(k,:)) .* (ysub(:,r)' - yout)) / denom;
            DelMean(IndFire(r,:)) = DelMean(IndFire(r,:)) + temp;
        end

        if phase == 1
            DelP = DelP + reshape(repmat(yout-Y(k,:), R, 1), R*C, 1) .* ...
                repmat(FSbar', C, 1) .* [1, gate_fun(lambda)] .* [1, X(k,:)];
            for d = 1:D
                DelGate(d) = DelGate(d) + X(k,d) * ...
                    (1-lambda(d)^2) * exp(1-lambda(d)^2)^0.5 * ...
                    sum(reshape(repmat(yout-Y(k,:),R,1),R*C,1) .* ...
                    repmat(FSbar',C,1) .* P(:,1+d));
            end
        else
            DelP = DelP + reshape(repmat(yout-Y(k,:), R, 1), R*C, 1) .* ...
                repmat(FSbar' .* gate_fun(theta), C, 1) .* [1, X(k,:)];
            for r = 1:R
                DelGate(r) = DelGate(r) + FSbar(r) * ...
                    (1-theta(r)^2) * exp(1-theta(r)^2)^0.5 * ...
                    sum((yout-Y(k,:))' .* (P(r:R:C*R,:) * [1,X(k,:)]'));
            end
        end
    end

    Mean = Mean - eta * (DelMean / N);
    P = P - eta * (DelP / N);
    if phase == 1
        lambda = lambda - eta * (DelGate / N);
    else
        theta = theta - eta * (DelGate / N);
    end

    if any(~isfinite(Mean(:))) || any(~isfinite(P(:))) || ...
            any(~isfinite(lambda(:))) || any(~isfinite(theta(:)))
        error('Non-finite parameters detected at iteration %d.', iter);
    end
end
end

function [X, Mean, P, IndFire, selected] = prune_features( ...
    X, Mean, P, IndFire, lambda, tau)
num_mf = size(Mean, 1);
bad = find(abs(gate_fun(lambda)) < tau);
selected = find(abs(gate_fun(lambda)) >= tau);

X(:, bad) = [];
Mean(:, bad) = [];
for i = 1:size(IndFire,2)
    IndFire(:,i) = IndFire(:,i) - (i-1)*num_mf;
end
IndFire(:, bad) = [];
for i = 1:size(IndFire,2)
    IndFire(:,i) = IndFire(:,i) + (i-1)*num_mf;
end
P(:, bad+1) = [];
end

function [IndFire, P, extracted] = prune_rules(IndFire, theta, P, tau)
R = size(IndFire, 1);
C = size(P, 1) / R;
pruned = find(abs(gate_fun(theta)) < tau);
extracted = find(abs(gate_fun(theta')) >= tau);

if numel(extracted) < C && numel(theta) > C
    [~, order] = sort(gate_fun(theta'), 'descend');
    pruned = order(C+1:end)';
    extracted = order(1:C);
end

IndFire(pruned,:) = [];
pruned_p = repmat(pruned', C, 1);
for c = 2:C
    pruned_p(c,:) = (c-1)*R + pruned_p(1,:);
end
P(pruned_p(:), :) = [];
end

%% ========================================================================
%                         Fast LSE final solver
% ========================================================================
function [Mean, P] = train_lse(X, Y, Mean, IndFire, gamma, max_design_elements)
[N, D] = size(X);
C = size(Y, 2);
R = size(IndFire, 1);
design_elements = N * R * (D + 1);
if design_elements > max_design_elements
    error('LSE design matrix would contain %.3e elements.', design_elements);
end
Phi = zeros(N, R*(D+1));

for k = 1:N
    Mu = exp(-(X(k,:) - Mean).^2);
    selected_mu = Mu(IndFire);
    q = get_q_approx(min(selected_mu, [], 2));
    FirStr = softmin_fun(selected_mu, q)';
    denom = sum(FirStr, 2);
    if ~isfinite(denom) || denom <= 0 || any(~isfinite(FirStr(:)))
        error('Invalid firing strengths in LSE construction, sample %d.', k);
    end
    FSbar = FirStr ./ denom;
    Phi(k,:) = reshape([1, X(k,:)]' * FSbar, 1, R*(D+1));
end

if any(~isfinite(Phi(:)))
    error('Non-finite LSE design matrix detected.');
end

p = size(Phi, 2);
if p <= N
    P_tilde = (Phi' * Phi + gamma * eye(p)) \ (Phi' * Y);
else
    % Algebraically equivalent dual ridge solution, used to avoid forming
    % an impractically large p-by-p matrix in high-dimensional settings.
    P_tilde = Phi' * ((Phi * Phi' + gamma * eye(N)) \ Y);
end

if any(~isfinite(P_tilde(:)))
    error('Non-finite LSE solution detected.');
end
P = reshape(P_tilde, D+1, R*C)';
end

%% ========================================================================
%                          Forward and statistics
% ========================================================================
function [Out, stats] = model_forward(X, n_classes, Mean, P, IndFire)
N = size(X, 1);
R = size(IndFire, 1);
Out = zeros(N, n_classes);
raw_zero_count = 0;
norm_zero_count = 0;
raw_all_zero_count = 0;
norm_all_zero_count = 0;

for k = 1:N
    Mu = exp(-(X(k,:) - Mean).^2);
    selected_mu = Mu(IndFire);
    q = get_q_approx(min(selected_mu, [], 2));
    FirStr = softmin_fun(selected_mu, q)';
    denom = sum(FirStr, 2);
    if ~isfinite(denom) || denom <= 0 || any(~isfinite(FirStr(:)))
        error('Invalid firing strengths during prediction, sample %d.', k);
    end
    FSbar = FirStr ./ denom;
    if any(~isfinite(FSbar(:)))
        error('Non-finite normalized firing strengths during prediction.');
    end

    raw_zero_count = raw_zero_count + sum(FirStr == 0);
    norm_zero_count = norm_zero_count + sum(FSbar == 0);
    raw_all_zero_count = raw_all_zero_count + all(FirStr == 0);
    norm_all_zero_count = norm_all_zero_count + all(FSbar == 0);

    ysub = reshape(P * [1, X(k,:)]', R, n_classes)';
    Out(k,:) = sum(ysub .* FSbar, 2)';
end

stats = [ ...
    100 * raw_zero_count / (N*R), ...
    100 * norm_zero_count / (N*R), ...
    100 * raw_all_zero_count / N, ...
    100 * norm_all_zero_count / N];
end

function [acc, bacc, macro_f1] = classification_metrics(y_true, y_pred, C)
acc = mean(y_true == y_pred);
recall = zeros(C,1);
f1 = zeros(C,1);
for c = 1:C
    tp = sum((y_true == c) & (y_pred == c));
    fn = sum((y_true == c) & (y_pred ~= c));
    fp = sum((y_true ~= c) & (y_pred == c));
    if tp + fn > 0
        recall(c) = tp / (tp + fn);
    else
        recall(c) = 0;
    end
    if tp + fp > 0
        precision = tp / (tp + fp);
    else
        precision = 0;
    end
    if precision + recall(c) > 0
        f1(c) = 2 * precision * recall(c) / (precision + recall(c));
    else
        f1(c) = 0;
    end
end
bacc = mean(recall);
macro_f1 = mean(f1);
end

%% ========================================================================
%                            Original helper logic
% ========================================================================
function y = softmin_fun(x, q)
y = (sum(x.^q, 2) / size(x,2)).^(1./q);
end

function q = get_q_approx(zn)
q = ceil(690 ./ (log(zn) - eps));
q(q < -1000) = -1000;
if any(~isfinite(q(:))) || any(q(:) == 0)
    error('Invalid adaptive softmin parameter q detected.');
end
end

function y = gate_fun(x)
y = x .* exp(1 - x.^2).^0.5;
end

function R = estimate_rule_count(D, num_mf)
if num_mf == 1 || D == 1
    R = num_mf;
elseif num_mf == 2 || D == 2
    R = (D + 1) * num_mf;
else
    R = (2*D + 1) * num_mf;
end
end

function IndFire = make_rule_index(D, num_mf)
if num_mf == 1 || D == 1
    IndFire = reshape(1:num_mf*D, num_mf, D);
    return;
end
I = eye(D);
if num_mf == 2 || D == 2
    num_group = D + 1;
else
    num_group = 2*D + 1;
end
IndFire = zeros(num_group*num_mf, D);
for i = 1:num_mf
    base = ones(1,D) * i;
    IndFire(num_group*(i-1)+1,:) = base;
    if i == 1
        left = base + I*(num_mf-1);
        right = base + I;
    elseif i == num_mf
        left = base - I;
        right = base - I*(num_mf-1);
    else
        left = base - I;
        right = base + I;
    end
    if num_mf == 2 || D == 2
        IndFire(num_group*(i-1)+2:num_group*i,:) = right;
    else
        IndFire(num_group*(i-1)+2:num_group*i,:) = [left; right];
    end
end
offset = (0:D-1) * num_mf;
IndFire = bsxfun(@plus, IndFire, offset);
end

%% ========================================================================
%                               Output helpers
% ========================================================================
function summary = summarize_dataset_rows( ...
    dataset_name, rows, dataset_time, n_splits, n_repeats)
status_col = 6;
valid_mask = strcmp(rows(:,status_col), 'success');
valid = rows(valid_mask,:);
valid_folds = size(valid,1);
failed_folds = size(rows,1) - valid_folds;

if valid_folds == 0
    means_stds = num2cell(nan(1,15));
    summary = [{dataset_name, valid_folds, failed_folds}, means_stds, {dataset_time}];
    return;
end

acc = cell2mat(rows(:,14));
bacc = cell2mat(rows(:,15));
f1 = cell2mat(rows(:,16));
selected = cell2mat(rows(:,12));
rules = cell2mat(rows(:,13));
feature_t = cell2mat(rows(:,17));
rule_t = cell2mat(rows(:,18));
lse_t = cell2mat(rows(:,19));
train_t = cell2mat(rows(:,20));
test_t = cell2mat(rows(:,21));
total_t = cell2mat(rows(:,22));

[acc_mean, acc_std] = repeated_cv_mean_std(acc, n_splits, n_repeats);
[bacc_mean, bacc_std] = repeated_cv_mean_std(bacc, n_splits, n_repeats);
[f1_mean, f1_std] = repeated_cv_mean_std(f1, n_splits, n_repeats);
[selected_mean, ~] = repeated_cv_mean_std(selected, n_splits, n_repeats);
[rules_mean, ~] = repeated_cv_mean_std(rules, n_splits, n_repeats);
[feature_mean, ~] = repeated_cv_mean_std(feature_t, n_splits, n_repeats);
[rule_mean, ~] = repeated_cv_mean_std(rule_t, n_splits, n_repeats);
[lse_mean, ~] = repeated_cv_mean_std(lse_t, n_splits, n_repeats);
[train_mean, ~] = repeated_cv_mean_std(train_t, n_splits, n_repeats);
[test_mean, ~] = repeated_cv_mean_std(test_t, n_splits, n_repeats);
[total_mean, total_std] = repeated_cv_mean_std( ...
    total_t, n_splits, n_repeats);

summary = {dataset_name, valid_folds, failed_folds, ...
    acc_mean, acc_std, bacc_mean, bacc_std, ...
    f1_mean, f1_std, selected_mean, rules_mean, ...
    feature_mean, rule_mean, lse_mean, train_mean, ...
    test_mean, total_mean, total_std, dataset_time};
end

function [mean_value, std_value] = repeated_cv_mean_std( ...
    values, n_splits, n_repeats)
% Average folds within each repetition, then summarize repetitions.
values = values(:);
expected = n_splits * n_repeats;
if numel(values) ~= expected
    error('FSREAdaTSK:InvalidFoldCount', ...
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

valid = repetition_means(isfinite(repetition_means));
if isempty(valid)
    mean_value = NaN;
    std_value = NaN;
elseif numel(valid) == 1
    mean_value = valid(1);
    std_value = 0;
else
    mean_value = mean(valid);
    std_value = std(valid, 0);
end
end

function write_cell_table(path, headers, rows)
if isempty(rows)
    return;
end
T = cell2table(rows, 'VariableNames', headers);
writetable(T, path);
end
