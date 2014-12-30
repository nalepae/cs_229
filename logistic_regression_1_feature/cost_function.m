%%cost_function: Cost function
function [J, grad] = cost_function(theta, X, y, lambda)
    % Number of training examples
    m = length(y);

    % Number of features
    n = length(theta);

    % Regularization matrix
    R = diag([0 ; ones(n - 1, 1)]);

    % Regularization vector
    r = R * theta;

    % Compute one time for all X * theta
    Xtheta = X * theta;

    % Hypothesis vector
    hypothesis = sigmoid(Xtheta);

    % Probabilistic vectorial part
    prob_vec_part = y .* log(hypothesis) + (1 - y) .* log(1 - hypothesis);

    % Probabilistic part
    probabilistic_part = - sum(prob_vec_part);

    % Regularization part
    regularization_part = lambda / 2 * theta' * r;

    J = 1 / m * (probabilistic_part + regularization_part);

    % Gradient vector
    grad = 1 / m * (X' * (hypothesis - y) + lambda * r);
end
