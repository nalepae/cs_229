%%cost_function: Cost function
function [J, grad] = cost_function(theta, X, y)
    % Number of training examples
    m = length(y);

    % Compute one time for all X * theta
    Xtheta = X * theta;

    % Hypothesis vector
    hypothesis = sigmoid(Xtheta);

    % Cost vector
    J_vector = y .* log(hypothesis) + (1 - y) .* log(1 - hypothesis);

    % Cost
    J = sum(J_vector) / -m;

    % Gradient vector
    grad = X' * (hypothesis - y);
end
