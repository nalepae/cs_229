%% gradient_descent: Gradient descent function
function [theta, J_history] = gradient_descent(X, y, theta, alpha, num_iters)
    % Number of training examples
    m = length(y);

    % History of cost
    J_history = zeros(num_iters, 1);

    for iter = 1:num_iters
        [J_history(iter), grad] = cost_function(theta, X, y);
        theta -= alpha * grad;
    end
end
