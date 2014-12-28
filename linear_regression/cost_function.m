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

    % Compute X * theta
    XTheta = X * theta;

    % Least square part
    least_square_part = (XTheta - y)' * (XTheta - y);
    
    regularization_part = lambda * (theta)' * r;

	J = 1 / (2 * m) * (least_square_part + regularization_part);     

    % Gradient vector
    grad = 1 / m * (X' * (XTheta - y) + lambda * r);
end