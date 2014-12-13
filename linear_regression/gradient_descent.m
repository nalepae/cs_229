%% gradient_descent: Gradient descent function
function [theta, J] = gradient_descent(X, theta, y, alpha)
	n = length(theta);

	for i = 1:n
		theta(i) += alpha * (y - X * theta)' * X(:, i);
	end

	J = linear_cost_function(X, theta, y);