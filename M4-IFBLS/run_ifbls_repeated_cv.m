function run_ifbls_repeated_cv()
%RUN_IFBLS_REPEATED_CV Repeated CV with holdout selection for IF-BLS.
%
% The outer five-repetition 5-fold stratified cross-validation estimates
% generalization performance. Within each outer training fold, 80% of the
% samples fit candidate configurations and 20% form a stratified validation
% subset. The selected configuration is refitted on the complete outer
% training fold before the outer test fold is evaluated once.

clc;
format compact;

%% ======================== Configuration ========================
OUTER_SPLITS = 5;
OUTER_REPEATS = 5;
VALIDATION_RATIO = 0.20;
RANDOM_SEED = 2026;

RUN_SMALL_ONLY = false;
SMALL_DATASETS = {'Leukemia', 'GSAFM'};

DATASET_NAMES = {
    'PAGEB', ...          % D1
    'Thyroid', ...        % D2
    'SATELLITE', ...      % D3
    'TEXTURE', ...        % D4
    'spambase', ...       % D5
    'ORL', ...            % D6
    'semg', ...           % D7
    'warpAR10P (1)', ...  % D8
    'PIE', ...            % D9
    'handoutlines', ...   % D10
    'Giesste', ...        % D11
    'Drivace', ...        % D12
    'Leukemia', ...       % D13
    'CMHS', ...           % D14
    'GSAFM' ...           % D15
};

% Candidate ranges reported in the HDFBLS manuscript.
C_GRID = [1e-5, 1e5];
MU_GRID = 2.^[-5, 0, 5];
NFG_GRID = [3, 5, 7];
NL_GRID = [10, 20, 30];
NE_GRID = [20, 40, 60, 80, 100];

SCRIPT_FOLDER = fileparts(mfilename('fullpath'));
REPOSITORY_FOLDER = fileparts(SCRIPT_FOLDER);
addpath(SCRIPT_FOLDER);
DATA_FOLDER = locate_data_folder( ...
    SCRIPT_FOLDER, REPOSITORY_FOLDER, DATASET_NAMES);
RESULT_FOLDER = fullfile(SCRIPT_FOLDER, 'results_ifbls_holdout_cv');
if ~exist(RESULT_FOLDER, 'dir')
    mkdir(RESULT_FOLDER);
end

VALIDATION_FILE = fullfile(RESULT_FOLDER, 'ifbls_validation_settings.csv');
SELECTED_FILE = fullfile(RESULT_FOLDER, 'ifbls_selected_settings.csv');
DETAIL_FILE = fullfile(RESULT_FOLDER, 'ifbls_outer_cv_details.csv');
SUMMARY_FILE = fullfile(RESULT_FOLDER, 'ifbls_holdout_cv_summary.csv');
DATASET_TIME_FILE = fullfile(RESULT_FOLDER, 'ifbls_dataset_times.csv');

initialize_csv(VALIDATION_FILE, {
    'dataset','outer_repeat','outer_fold','outer_split','parameter_id', ...
    'C','mu','Nfg','Nl','Ne','validation_seed', ...
    'validation_accuracy','validation_balanced_accuracy', ...
    'validation_macro_f1','status','error_id','error_message'});

initialize_csv(SELECTED_FILE, {
    'dataset','outer_repeat','outer_fold','outer_split','parameter_id', ...
    'C','mu','Nfg','Nl','Ne','validation_accuracy', ...
    'selection_time_seconds'});

initialize_csv(DETAIL_FILE, {
    'dataset','repeat','fold','split','parameter_id','C','mu', ...
    'Nfg','Nl','Ne','feature_nodes','enhancement_nodes','total_nodes', ...
    'binary_models','effective_total_nodes','validation_accuracy', ...
    'model_seed','accuracy','balanced_accuracy','macro_f1', ...
    'selection_time_seconds','score_time_seconds','train_time_seconds', ...
    'predict_time_seconds','final_model_time_seconds', ...
    'total_computation_time_seconds','status', ...
    'error_id','error_message'});

initialize_csv(SUMMARY_FILE, {
    'dataset','accuracy_mean','accuracy_std','balanced_accuracy_mean', ...
    'balanced_accuracy_std','macro_f1_mean','macro_f1_std', ...
    'selected_C_mode','selected_mu_mode','selected_Nfg_mean', ...
    'selected_Nfg_mode','selected_Nl_mean','selected_Nl_mode', ...
    'selected_Ne_mean','selected_Ne_mode','feature_nodes_mean', ...
    'total_nodes_mean','final_model_time_mean', ...
    'final_model_time_std','selection_time_mean','selection_time_std', ...
    'total_computation_time_mean','total_computation_time_std', ...
    'dataset_time_seconds'});

initialize_csv(DATASET_TIME_FILE, {
    'dataset','n_samples','n_features','n_classes','outer_folds', ...
    'dataset_time_seconds','status','message'});

if RUN_SMALL_ONLY
    datasets = SMALL_DATASETS;
else
    datasets = DATASET_NAMES;
end

parameter_grid = build_parameter_grid( ...
    C_GRID, MU_GRID, NFG_GRID, NL_GRID, NE_GRID);
num_parameters = size(parameter_grid, 1);

%% ======================== Main experiment ========================
for dataset_id = 1:numel(datasets)
    dataset_name = datasets{dataset_id};
    dataset_clock = tic;

    fprintf('\n============================================================\n');
    fprintf('Dataset: %s\n', dataset_name);
    fprintf('============================================================\n');

    try
        [X_all, y_raw] = load_dataset(DATA_FOLDER, dataset_name);
        X_all = double(X_all);
        if any(~isfinite(X_all(:)))
            error('IFBLS:InvalidInput', 'Input data contain NaN or Inf.');
        end

        y_all = encode_labels(y_raw);
        n_samples = size(X_all, 1);
        n_features = size(X_all, 2);
        n_classes = max(y_all);
        class_counts = accumarray(y_all, 1, [n_classes, 1]);
        if min(class_counts) < OUTER_SPLITS
            error('IFBLS:InsufficientClassCount', ...
                'The smallest class has fewer than %d samples.', OUTER_SPLITS);
        end

        outer_folds = repeated_stratified_folds( ...
            y_all, OUTER_SPLITS, OUTER_REPEATS, RANDOM_SEED);
        total_outer_folds = numel(outer_folds);
        binary_models = max(1, n_classes * (n_classes > 2));

        outer_acc = nan(total_outer_folds, 1);
        outer_bacc = nan(total_outer_folds, 1);
        outer_f1 = nan(total_outer_folds, 1);
        outer_final_time = nan(total_outer_folds, 1);
        outer_total_time = nan(total_outer_folds, 1);
        selection_times = nan(total_outer_folds, 1);
        selected_parameters = nan(total_outer_folds, 5);
        selected_feature_nodes = nan(total_outer_folds, 1);
        selected_total_nodes = nan(total_outer_folds, 1);

        fprintf(['%d repetitions of %d-fold CV with a stratified ' ...
            'holdout (%d outer folds).\n'], ...
            OUTER_REPEATS, OUTER_SPLITS, total_outer_folds);

        for outer_id = 1:total_outer_folds
            outer_train_idx = outer_folds(outer_id).train_idx;
            outer_test_idx = outer_folds(outer_id).test_idx;
            repeat_id = outer_folds(outer_id).repeat_id;
            fold_in_repeat = outer_folds(outer_id).fold_in_repeat;

            X_outer_train = X_all(outer_train_idx, :);
            y_outer_train = y_all(outer_train_idx);
            X_outer_test = X_all(outer_test_idx, :);
            y_outer_test = y_all(outer_test_idx);

            validation_seed = RANDOM_SEED + outer_id;
            [fit_idx, validation_idx] = stratified_holdout( ...
                y_outer_train, VALIDATION_RATIO, validation_seed);

            selection_clock = tic;
            validation_seed_base = RANDOM_SEED ...
                + dataset_id * 100000000 ...
                + outer_id * 10000 ...
                + 100;
            validation_result = ifbls_nested_fold( ...
                X_outer_train(fit_idx, :), ...
                y_outer_train(fit_idx), ...
                X_outer_train(validation_idx, :), ...
                y_outer_train(validation_idx), ...
                n_classes, parameter_grid, MU_GRID, validation_seed_base);

            validation_acc = validation_result.accuracy;
            validation_bacc = validation_result.balanced_accuracy;
            validation_f1 = validation_result.macro_f1;
            valid_parameter = isfinite(validation_acc);

            candidate_ids = find(valid_parameter);
            if isempty(candidate_ids)
                error('IFBLS:AllValidationSettingsFailed', ...
                    'All settings failed on validation data in outer fold %d.', ...
                    outer_id);
            end

            % Select by validation accuracy. MATLAB max returns
            % the first grid entry when accuracies tie, matching the original
            % accuracy-based selection rule.
            [~, local_best] = max(validation_acc(candidate_ids));
            best_id = candidate_ids(local_best);
            best_parameters = parameter_grid(best_id, :);
            selection_time = toc(selection_clock);
            selection_times(outer_id) = selection_time;
            selected_parameters(outer_id, :) = best_parameters;

            for parameter_id = 1:num_parameters
                append_csv_row(VALIDATION_FILE, {
                    dataset_name, repeat_id, fold_in_repeat, outer_id, ...
                    parameter_id, parameter_grid(parameter_id, 1), ...
                    parameter_grid(parameter_id, 2), ...
                    parameter_grid(parameter_id, 3), ...
                    parameter_grid(parameter_id, 4), ...
                    parameter_grid(parameter_id, 5), ...
                    validation_seed_base + parameter_id * 100000, ...
                    validation_acc(parameter_id), ...
                    validation_bacc(parameter_id), ...
                    validation_f1(parameter_id), ...
                    validation_result.status{parameter_id}, ...
                    validation_result.error_id{parameter_id}, ...
                    validation_result.error_message{parameter_id}});
            end

            append_csv_row(SELECTED_FILE, {
                dataset_name, repeat_id, fold_in_repeat, outer_id, best_id, ...
                best_parameters(1), best_parameters(2), best_parameters(3), ...
                best_parameters(4), best_parameters(5), ...
                validation_acc(best_id), selection_time});

            final_seed_base = RANDOM_SEED ...
                + dataset_id * 100000000 ...
                + outer_id * 100;
            final_result = ifbls_nested_fold( ...
                X_outer_train, y_outer_train, X_outer_test, y_outer_test, ...
                n_classes, [best_parameters, best_id], best_parameters(2), ...
                final_seed_base);

            outer_acc(outer_id) = final_result.accuracy(1);
            outer_bacc(outer_id) = final_result.balanced_accuracy(1);
            outer_f1(outer_id) = final_result.macro_f1(1);
            outer_final_time(outer_id) = final_result.total_time(1);
            if isfinite(final_result.accuracy(1))
                outer_total_time(outer_id) = ...
                    selection_time + final_result.total_time(1);
            end

            feature_nodes = best_parameters(3) * best_parameters(4);
            total_nodes = feature_nodes + best_parameters(5);
            selected_feature_nodes(outer_id) = feature_nodes;
            selected_total_nodes(outer_id) = total_nodes;
            effective_total_nodes = binary_models * total_nodes;
            model_seed = final_seed_base + best_id * 100000;

            append_csv_row(DETAIL_FILE, {
                dataset_name, repeat_id, fold_in_repeat, outer_id, best_id, ...
                best_parameters(1), best_parameters(2), best_parameters(3), ...
                best_parameters(4), best_parameters(5), feature_nodes, ...
                best_parameters(5), total_nodes, binary_models, ...
                effective_total_nodes, validation_acc(best_id), model_seed, ...
                final_result.accuracy(1), ...
                final_result.balanced_accuracy(1), ...
                final_result.macro_f1(1), selection_time, ...
                final_result.score_time(1), final_result.train_time(1), ...
                final_result.predict_time(1), final_result.total_time(1), ...
                outer_total_time(outer_id), ...
                final_result.status{1}, final_result.error_id{1}, ...
                final_result.error_message{1}});

            fprintf(['  Outer %02d/%02d | C=%.1e, mu=%.5g, ' ...
                'Nfg=%d, Nl=%d, Ne=%d | ACC=%.4f | select=%.2fs\n'], ...
                outer_id, total_outer_folds, best_parameters(1), ...
                best_parameters(2), best_parameters(3), ...
                best_parameters(4), best_parameters(5), ...
                final_result.accuracy(1), selection_time);
        end

        [acc_mean, acc_std] = repeated_cv_mean_std( ...
            outer_acc, OUTER_SPLITS, OUTER_REPEATS);
        [bacc_mean, bacc_std] = repeated_cv_mean_std( ...
            outer_bacc, OUTER_SPLITS, OUTER_REPEATS);
        [f1_mean, f1_std] = repeated_cv_mean_std( ...
            outer_f1, OUTER_SPLITS, OUTER_REPEATS);
        [final_time_mean, final_time_std] = repeated_cv_mean_std( ...
            outer_final_time, OUTER_SPLITS, OUTER_REPEATS);
        [selection_time_mean, selection_time_std] = repeated_cv_mean_std( ...
            selection_times, OUTER_SPLITS, OUTER_REPEATS);
        [total_time_mean, total_time_std] = repeated_cv_mean_std( ...
            outer_total_time, OUTER_SPLITS, OUTER_REPEATS);

        dataset_time = toc(dataset_clock);
        append_csv_row(SUMMARY_FILE, {
            dataset_name, acc_mean, acc_std, bacc_mean, bacc_std, ...
            f1_mean, f1_std, mode(selected_parameters(:, 1)), ...
            mode(selected_parameters(:, 2)), ...
            mean(selected_parameters(:, 3)), mode(selected_parameters(:, 3)), ...
            mean(selected_parameters(:, 4)), mode(selected_parameters(:, 4)), ...
            mean(selected_parameters(:, 5)), mode(selected_parameters(:, 5)), ...
            mean(selected_feature_nodes), mean(selected_total_nodes), ...
            final_time_mean, final_time_std, selection_time_mean, ...
            selection_time_std, total_time_mean, total_time_std, dataset_time});

        append_csv_row(DATASET_TIME_FILE, {
            dataset_name, n_samples, n_features, n_classes, ...
            total_outer_folds, dataset_time, 'ok', ''});

        fprintf(['[%s] ACC=%.4f+-%.4f | BACC=%.4f+-%.4f | ' ...
            'Macro-F1=%.4f+-%.4f\n'], dataset_name, acc_mean, acc_std, ...
            bacc_mean, bacc_std, f1_mean, f1_std);

    catch ME
        dataset_time = toc(dataset_clock);
        fprintf('[Skipped/Failed] %s: %s\n', dataset_name, ME.message);
        append_csv_row(DATASET_TIME_FILE, {
            dataset_name, nan, nan, nan, nan, dataset_time, ...
            'failed', [ME.identifier, ': ', ME.message]});
    end
end

fprintf('\nAll holdout-validation IF-BLS experiments finished.\n');
fprintf('Results folder: %s\n', RESULT_FOLDER);
end

%% ======================== Data utilities ========================
function data_folder = locate_data_folder( ...
    script_folder, repository_folder, dataset_names)
candidates = {
    fullfile(script_folder, 'datasets_HDFBLS'), ...
    script_folder, ...
    fullfile(repository_folder, 'datasets_HDFBLS')};

data_folder = candidates{1};
for folder_id = 1:numel(candidates)
    candidate = candidates{folder_id};
    for dataset_id = 1:numel(dataset_names)
        file_path = fullfile(candidate, [dataset_names{dataset_id}, '.mat']);
        if exist(file_path, 'file')
            data_folder = candidate;
            return;
        end
    end
end
end

function [X, y] = load_dataset(data_folder, dataset_name)
file_path = fullfile(data_folder, [dataset_name, '.mat']);
if ~exist(file_path, 'file')
    error('IFBLS:MissingDataset', 'Missing dataset file: %s', file_path);
end

ws = load(file_path);
X = [];
y = [];
if isfield(ws, 'data') && isfield(ws, 'label')
    X = ws.data; y = ws.label;
elseif isfield(ws, 'X') && isfield(ws, 'Y')
    X = ws.X; y = ws.Y;
elseif isfield(ws, 'sample') && isfield(ws, 'target')
    X = ws.sample; y = ws.target;
elseif isfield(ws, dataset_name)
    combined = ws.(dataset_name);
    X = combined(:, 1:end-1); y = combined(:, end);
else
    fields = fieldnames(ws);
    if numel(fields) ~= 1
        error('IFBLS:UnrecognizedDataset', ...
            'Cannot identify data and labels in %s.', file_path);
    end
    combined = ws.(fields{1});
    X = combined(:, 1:end-1); y = combined(:, end);
end

if ismatrix(y) && size(y, 2) > 1 && size(y, 1) == size(X, 1)
    [~, y] = max(y, [], 2);
end
y = y(:);
if size(X, 1) ~= numel(y)
    if size(X, 2) == numel(y)
        X = X.';
    else
        error('IFBLS:DimensionMismatch', ...
            'Samples and labels have inconsistent dimensions.');
    end
end
end

function [fit_idx, validation_idx] = stratified_holdout( ...
    y, validation_ratio, seed)
rng(seed, 'twister');
partition = cvpartition(y, 'HoldOut', validation_ratio);
fit_idx = find(training(partition));
validation_idx = find(test(partition));
if isempty(fit_idx) || isempty(validation_idx)
    error('IFBLS:InvalidValidationSplit', ...
        'The stratified holdout produced an empty subset.');
end
end

function y = encode_labels(y_raw)
if isnumeric(y_raw) || islogical(y_raw)
    [~, ~, y] = unique(y_raw(:), 'stable');
else
    [~, ~, y] = unique(string(y_raw(:)), 'stable');
end
y = double(y(:));
end

function folds = repeated_stratified_folds(y, n_splits, n_repeats, seed)
n = numel(y);
n_classes = max(y);
folds = repmat(struct('train_idx', [], 'test_idx', [], ...
    'repeat_id', 0, 'fold_in_repeat', 0), n_splits * n_repeats, 1);
entry = 0;
for repeat_id = 1:n_repeats
    rng(seed + repeat_id - 1, 'twister');
    fold_assignment = zeros(n, 1);
    for class_id = 1:n_classes
        class_idx = find(y == class_id);
        class_idx = class_idx(randperm(numel(class_idx)));
        class_fold = mod((0:numel(class_idx)-1), n_splits) + 1;
        fold_assignment(class_idx) = class_fold(:);
    end
    for fold_id = 1:n_splits
        entry = entry + 1;
        folds(entry).test_idx = find(fold_assignment == fold_id);
        folds(entry).train_idx = find(fold_assignment ~= fold_id);
        folds(entry).repeat_id = repeat_id;
        folds(entry).fold_in_repeat = fold_id;
    end
end
end

function grid = build_parameter_grid(c_grid, mu_grid, nfg_grid, nl_grid, ne_grid)
num_rows = numel(c_grid) * numel(mu_grid) * numel(nfg_grid) ...
    * numel(nl_grid) * numel(ne_grid);
grid = zeros(num_rows, 5);
row = 0;
for c_value = c_grid
    for mu_value = mu_grid
        for nfg = nfg_grid
            for nl = nl_grid
                for ne = ne_grid
                    row = row + 1;
                    grid(row, :) = [c_value, mu_value, nfg, nl, ne];
                end
            end
        end
    end
end
end

function [mean_value, std_value, count] = finite_mean_std(values)
values = values(isfinite(values));
count = numel(values);
if count == 0
    mean_value = nan; std_value = nan;
elseif count == 1
    mean_value = values(1); std_value = 0;
else
    mean_value = mean(values); std_value = std(values, 0);
end
end

function [mean_value, std_value, count] = repeated_cv_mean_std( ...
    values, n_splits, n_repeats)
values = values(:);
if numel(values) ~= n_splits * n_repeats
    error('IFBLS:UnexpectedFoldCount', ...
        'Expected %d values but received %d.', ...
        n_splits * n_repeats, numel(values));
end
if any(~isfinite(values))
    mean_value = nan;
    std_value = nan;
    count = sum(isfinite(values));
    return;
end
matrix = reshape(values, n_splits, n_repeats);
repeat_means = mean(matrix, 1, 'omitnan');
[mean_value, std_value, count] = finite_mean_std(repeat_means(:));
end

%% ======================== CSV utilities ========================
function initialize_csv(file_path, headers)
fid = fopen(file_path, 'w');
if fid < 0
    error('IFBLS:CannotOpenCSV', 'Cannot open %s.', file_path);
end
cleanup = onCleanup(@() fclose(fid)); %#ok<NASGU>
fprintf(fid, '%s', headers{1});
for i = 2:numel(headers)
    fprintf(fid, ',%s', headers{i});
end
fprintf(fid, '\n');
end

function append_csv_row(file_path, values)
fid = fopen(file_path, 'a');
if fid < 0
    error('IFBLS:CannotOpenCSV', 'Cannot open %s.', file_path);
end
cleanup = onCleanup(@() fclose(fid)); %#ok<NASGU>
for i = 1:numel(values)
    if i > 1
        fprintf(fid, ',');
    end
    fprintf(fid, '%s', csv_escape(value_to_csv(values{i})));
end
fprintf(fid, '\n');
end

function text = value_to_csv(value)
if ischar(value)
    text = value;
elseif isstring(value)
    text = char(value);
elseif isnumeric(value) || islogical(value)
    if isempty(value)
        text = '';
    elseif isscalar(value)
        text = sprintf('%.17g', double(value));
    else
        text = mat2str(value);
    end
else
    text = char(string(value));
end
end

function text = csv_escape(text)
if contains(text, '"')
    text = strrep(text, '"', '""');
end
if contains(text, ',') || contains(text, '"') || contains(text, newline)
    text = ['"', text, '"'];
end
end
