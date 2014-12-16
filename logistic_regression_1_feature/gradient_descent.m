%% gradient_descent: Gradient descent function
function [theta, J] = gradient_descent(X, theta, y, alpha)
	n = length(theta);
	m = length(y);

	for j = 1 : n
		partial_derivative_J = 0;
		for i = 1 : m
			g = hypothesis(X(i, :)', theta);
			partial_derivative_J += (g - y(i)) * g * (1 - g) * X(i, j);
		end

		theta(j) -= alpha * partial_derivative_J;
	end

	J = cost_function(X, theta, y);