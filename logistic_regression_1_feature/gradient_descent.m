%% gradient_descent: Gradient descent function
function [theta, theta_history, J_history] = gradient_descent(X, y, theta,
                                                              alpha,
                                                              num_iters,
                                                              lambda)
    % Number of training examples
    m = length(y);

    % History of cost
    J_history = zeros(num_iters, 1);

    % History of theta
    theta_history = theta;

    for iter = 1:num_iters
        [J_history(iter), grad] = cost_function(theta, X, y, lambda);
        theta -= alpha * grad;

        theta_history = [theta_history theta];
    end
end
