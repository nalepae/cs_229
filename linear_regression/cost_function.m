%%cost_function: Cost function
function [J, grad] = cost_function(theta, X, y)
    % Number of training examples
    m = length(y);

    % Compute X * theta
    XTheta = X * theta;

    J = 1 / (2 * m) * (XTheta - y)' * (XTheta - y);

    % Gradient vector
    grad = 1 / m * X' * (XTheta - y);
end