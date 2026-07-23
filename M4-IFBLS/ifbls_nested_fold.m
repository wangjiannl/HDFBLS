function result = ifbls_nested_fold( ...
    X_train_raw, y_train, X_eval_raw, y_eval, n_classes, ...
    parameter_grid, mu_grid, seed_base)
%IFBLS_NESTED_FOLD Fit and evaluate one or more IF-BLS configurations.
% The legacy filename is retained for compatibility. The current driver uses
% one stratified holdout split for selection, rather than inner cross-validation.
%
% Min-max preprocessing and intuitionistic fuzzy scores are fitted using
% only the supplied training data. Scores are cached by mu and reused by
% configurations evaluated on the same fold.

[x_min, x_range] = minmax_fit(X_train_raw);
X_train = minmax_apply(X_train_raw, x_min, x_range);
X_eval = minmax_apply(X_eval_raw, x_min, x_range);

num_parameters = size(parameter_grid, 1);
result.accuracy = nan(1, num_parameters);
result.balanced_accuracy = nan(1, num_parameters);
result.macro_f1 = nan(1, num_parameters);
result.score_time = nan(1, num_parameters);
result.train_time = nan(1, num_parameters);
result.predict_time = nan(1, num_parameters);
result.total_time = nan(1, num_parameters);
result.status = repmat({'failed'}, 1, num_parameters);
result.error_id = repmat({''}, 1, num_parameters);
result.error_message = repmat({''}, 1, num_parameters);

score_cache = cell(numel(mu_grid), 1);
score_cache_time = nan(numel(mu_grid), 1);
score_cache_status = false(numel(mu_grid), 1);
score_cache_error_id = repmat({''}, numel(mu_grid), 1);
score_cache_error_message = repmat({''}, numel(mu_grid), 1);

for mu_id = 1:numel(mu_grid)
    score_clock = tic;
    try
        score_cache{mu_id} = if_score_all_classes( ...
            X_train, y_train, n_classes, mu_grid(mu_id));
        score_cache_status(mu_id) = true;
    catch ME
        score_cache_error_id{mu_id} = ME.identifier;
        score_cache_error_message{mu_id} = ME.message;
    end
    score_cache_time(mu_id) = toc(score_clock);
end

for parameter_row = 1:num_parameters
    c_value = parameter_grid(parameter_row, 1);
    mu_value = parameter_grid(parameter_row, 2);
    nfg = parameter_grid(parameter_row, 3);
    nl = parameter_grid(parameter_row, 4);
    ne = parameter_grid(parameter_row, 5);
    if size(parameter_grid, 2) >= 6
        parameter_id = parameter_grid(parameter_row, 6);
    else
        parameter_id = parameter_row;
    end

    mu_id = find( ...
        abs(mu_grid - mu_value) < eps(max(1, abs(mu_value))), 1);
    if isempty(mu_id)
        error('IFBLS:UnknownMu', ...
            'Parameter mu=%.17g is absent from the score cache.', mu_value);
    end

    result.score_time(parameter_row) = score_cache_time(mu_id);
    option.c = c_value;
    option.mu = mu_value;
    option.n1 = nl;
    option.n2 = nfg;
    option.n3 = ne;

    try
        if ~score_cache_status(mu_id)
            error(score_cache_error_id{mu_id}, '%s', ...
                score_cache_error_message{mu_id});
        end

        model_seed = seed_base + parameter_id * 100000;
        train_clock = tic;
        if n_classes == 2
            y_binary = double(y_train == 2);
            rng(model_seed, 'twister');
            model = ifbls_train_binary( ...
                X_train, y_binary, score_cache{mu_id}{1}, option);
            models = {model};
        else
            models = cell(n_classes, 1);
            for class_id = 1:n_classes
                y_binary = double(y_train == class_id);
                rng(model_seed + class_id, 'twister');
                models{class_id} = ifbls_train_binary( ...
                    X_train, y_binary, ...
                    score_cache{mu_id}{class_id}, option);
            end
        end
        result.train_time(parameter_row) = toc(train_clock);

        predict_clock = tic;
        if n_classes == 2
            [y_pred, ~] = ifbls_predict_binary(models{1}, X_eval);
            y_pred = y_pred + 1;
        else
            class_scores = zeros(size(X_eval, 1), n_classes);
            for class_id = 1:n_classes
                [~, positive_probability] = ...
                    ifbls_predict_binary(models{class_id}, X_eval);
                class_scores(:, class_id) = positive_probability;
            end
            [~, y_pred] = max(class_scores, [], 2);
        end
        result.predict_time(parameter_row) = toc(predict_clock);

        [acc, bacc, macro_f1] = classification_metrics( ...
            y_eval, y_pred, n_classes);
        metric_vector = [acc, bacc, macro_f1, ...
            result.score_time(parameter_row), ...
            result.train_time(parameter_row), ...
            result.predict_time(parameter_row)];
        if any(~isfinite(metric_vector))
            error('IFBLS:NonFiniteResult', ...
                'A metric or timing value is NaN or Inf.');
        end

        result.accuracy(parameter_row) = acc;
        result.balanced_accuracy(parameter_row) = bacc;
        result.macro_f1(parameter_row) = macro_f1;
        result.total_time(parameter_row) = ...
            result.score_time(parameter_row) ...
            + result.train_time(parameter_row) ...
            + result.predict_time(parameter_row);
        result.status{parameter_row} = 'ok';

    catch ME
        result.error_id{parameter_row} = ME.identifier;
        result.error_message{parameter_row} = ME.message;
    end
end
end

%% ======================== Preprocessing ========================
function [x_min, x_range] = minmax_fit(X_train)
x_min = min(X_train, [], 1);
x_max = max(X_train, [], 1);
x_range = x_max - x_min;
x_range(x_range == 0) = 1;
end

function X_scaled = minmax_apply(X, x_min, x_range)
X_scaled = bsxfun(@rdivide, bsxfun(@minus, X, x_min), x_range);
X_scaled = min(max(X_scaled, 0), 1);
if any(~isfinite(X_scaled(:)))
    error('IFBLS:NonFiniteScaledData', ...
        'Min-max scaling produced NaN or Inf.');
end
end

%% ======================== IF score ========================
function score_cells = if_score_all_classes(X, y, n_classes, mu)
if mu <= 0 || ~isfinite(mu)
    error('IFBLS:InvalidMu', 'Kernel parameter mu must be positive.');
end

squared_norm = sum(X.^2, 2);
distance_squared = bsxfun(@plus, squared_norm, squared_norm.') ...
    - 2 * (X * X.');
distance_squared = max(distance_squared, 0);
K = exp(-distance_squared / (mu^2));
if any(~isfinite(K(:)))
    error('IFBLS:NonFiniteKernel', 'Kernel matrix contains NaN or Inf.');
end

DD = sqrt(max(2 * (1 - K), 0));
if n_classes == 2
    score_cells = {if_score_from_kernel(K, DD, double(y == 2))};
else
    score_cells = cell(n_classes, 1);
    for class_id = 1:n_classes
        score_cells{class_id} = if_score_from_kernel( ...
            K, DD, double(y == class_id));
    end
end
end

function S = if_score_from_kernel(K, DD, binary_label)
n = numel(binary_label);
pos_idx = binary_label == 1;
neg_idx = ~pos_idx;
if ~any(pos_idx) || ~any(neg_idx)
    error('IFBLS:InvalidBinarySplit', ...
        'A one-vs-rest problem has an empty positive or negative class.');
end

K1 = K(pos_idx, pos_idx);
K2 = K(neg_idx, neg_idx);
radius_pos_term = 1 - 2 * mean(K1, 2) + mean(K1(:));
radius_neg_term = 1 - 2 * mean(K2, 2) + mean(K2(:));
radius_pos = sqrt(max(radius_pos_term, 0));
radius_neg = sqrt(max(radius_neg_term, 0));
radius_max_pos = max(radius_pos);
radius_max_neg = max(radius_neg);
alpha_d = max(radius_max_pos, radius_max_neg);

membership = zeros(n, 1);
membership(pos_idx) = 1 - radius_pos / (radius_max_pos + 1e-4);
membership(neg_idx) = 1 - radius_neg / (radius_max_neg + 1e-4);

rho = zeros(n, 1);
for i = 1:n
    neighbor_mask = DD(i, :).' < alpha_d;
    neighbor_count = sum(neighbor_mask);
    if neighbor_count == 0
        error('IFBLS:EmptyNeighborhood', ...
            'No sample lies inside the IF-score neighborhood.');
    end
    rho(i) = sum(binary_label(i) ~= binary_label(neighbor_mask)) ...
        / neighbor_count;
end

v = (1 - membership) .* rho;
S = zeros(n, 1);
for i = 1:n
    if v(i) == 0
        S(i) = membership(i);
    elseif membership(i) <= v(i)
        S(i) = 0;
    else
        denominator = 2 - membership(i) - v(i);
        if denominator == 0
            error('IFBLS:ZeroScoreDenominator', ...
                'The IF-score denominator is zero.');
        end
        S(i) = (1 - v(i)) / denominator;
    end
end
if any(~isfinite(S))
    error('IFBLS:NonFiniteIFScore', ...
        'The intuitionistic fuzzy score contains NaN or Inf.');
end
end

%% ======================== Binary IF-BLS core ========================
function model = ifbls_train_binary(X, y_binary, score_vector, option)
Y = [1 - y_binary(:), y_binary(:)];
Nsample = size(X, 1);
N1 = option.n1;
N2 = option.n2;
N3 = option.n3;
C = option.c;
if C <= 0 || ~isfinite(C)
    error('IFBLS:InvalidRegularization', ...
        'Regularization parameter C must be positive.');
end

H1 = [X, 0.1 * ones(size(X, 1), 1)];
Z = [];
We = cell(N2, 1);
for i = 1:N2
    We{i} = 2 * rand(size(X, 2) + 1, N1) - 1;
    A1 = H1 * We{i};
    A1 = local_mapminmax(A1);
    Z = [Z, A1]; %#ok<AGROW>
end

H2 = [Z, 0.1 * ones(size(X, 1), 1)];
if N1 * N2 >= N3
    wh = orth(2 * rand(N2 * N1 + 1, N3) - 1);
else
    random_matrix = 2 * rand(N2 * N1 + 1, N3) - 1;
    wh = orth(random_matrix.').';
end
if size(wh, 2) ~= N3
    error('IFBLS:EnhancementRankFailure', ...
        'Orthogonal initialization produced %d rather than %d nodes.', ...
        size(wh, 2), N3);
end

H = tanh(H2 * wh);
A = [Z, H];
if any(~isfinite(A(:)))
    error('IFBLS:NonFiniteFeatureMatrix', ...
        'The IF-BLS feature matrix contains NaN or Inf.');
end

s = score_vector(:);
if numel(s) ~= Nsample
    error('IFBLS:ScoreSizeMismatch', ...
        'The IF-score vector length does not match the sample count.');
end

if size(A, 2) < Nsample
    weighted_A = bsxfun(@times, A, s);
    weighted_Y = bsxfun(@times, Y, s);
    left_matrix = eye(size(A, 2)) / C + weighted_A.' * weighted_A;
    right_matrix = weighted_A.' * weighted_Y;
    W = left_matrix \ right_matrix;
else
    lambda = 1e-4;
    diagonal_term = (1 ./ (s + lambda).^2) / C;
    left_matrix = A * A.' + diag(diagonal_term);
    W = A.' * (left_matrix \ Y);
end
if any(~isfinite(W(:)))
    error('IFBLS:NonFiniteOutputWeights', ...
        'The IF-BLS output weights contain NaN or Inf.');
end

model.W = W;
model.Wh = {wh};
model.We = We;
model.n1 = N1;
model.n2 = N2;
model.n3 = N3;
end

function [predicted_binary_label, positive_probability] = ...
    ifbls_predict_binary(model, X)
T1 = [X, 0.1 * ones(size(X, 1), 1)];
Z_test = [];
for i = 1:model.n2
    T2 = T1 * model.We{i};
    T2 = local_mapminmax(T2);
    Z_test = [Z_test, T2]; %#ok<AGROW>
end
I2 = [Z_test, 0.1 * ones(size(X, 1), 1)];
H_test = tanh(I2 * model.Wh{1});
output = [Z_test, H_test] * model.W;
if any(~isfinite(output(:)))
    error('IFBLS:NonFinitePrediction', ...
        'The IF-BLS prediction matrix contains NaN or Inf.');
end
probabilities = stable_softmax(output);
[~, class_index] = max(probabilities, [], 2);
predicted_binary_label = class_index - 1;
positive_probability = probabilities(:, 2);
end

function Y = local_mapminmax(X)
row_min = min(X, [], 2);
row_max = max(X, [], 2);
row_range = row_max - row_min;
row_range(row_range == 0) = 1;
Y = 2 * bsxfun(@rdivide, bsxfun(@minus, X, row_min), row_range) - 1;
if any(~isfinite(Y(:)))
    error('IFBLS:NonFiniteMapMinMax', ...
        'Row-wise mapminmax produced NaN or Inf.');
end
end

function probabilities = stable_softmax(scores)
shifted = bsxfun(@minus, scores, max(scores, [], 2));
exponentials = exp(shifted);
denominator = sum(exponentials, 2);
if any(~isfinite(denominator)) || any(denominator <= 0)
    error('IFBLS:InvalidSoftmax', ...
        'Softmax normalization denominator is invalid.');
end
probabilities = bsxfun(@rdivide, exponentials, denominator);
end

%% ======================== Metrics ========================
function [accuracy, balanced_accuracy, macro_f1] = ...
    classification_metrics(y_true, y_pred, n_classes)
y_true = y_true(:);
y_pred = y_pred(:);
accuracy = mean(y_true == y_pred);
recall = zeros(n_classes, 1);
class_f1 = zeros(n_classes, 1);
for class_id = 1:n_classes
    true_positive = sum(y_true == class_id & y_pred == class_id);
    false_positive = sum(y_true ~= class_id & y_pred == class_id);
    false_negative = sum(y_true == class_id & y_pred ~= class_id);
    recall_denominator = true_positive + false_negative;
    precision_denominator = true_positive + false_positive;
    if recall_denominator > 0
        recall(class_id) = true_positive / recall_denominator;
    end
    if precision_denominator > 0
        precision = true_positive / precision_denominator;
    else
        precision = 0;
    end
    if precision + recall(class_id) > 0
        class_f1(class_id) = ...
            2 * precision * recall(class_id) ...
            / (precision + recall(class_id));
    end
end
balanced_accuracy = mean(recall);
macro_f1 = mean(class_f1);
end
