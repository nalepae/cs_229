%% gradient_descent_log: Gradient descent logarithm function
function [theta, J] = gradient_descent_log(X, theta, y, alpha)
	n = length(theta);
	m = length(y);

	for j = 1 : n
		partial_derivative_J = 0;
		for i = 1 : m
			h = hypothesis(X(i, :)', theta);
			partial_derivative_J += (h - y(i)) * X(i, j);
		end

		theta(j) -= alpha * partial_derivative_J;
	end

	J = cost_function_log(X, theta, y);