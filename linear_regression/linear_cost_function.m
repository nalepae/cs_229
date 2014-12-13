%% linear_cost_function: Linear cost function
function J = linear_cost_function(X, theta, y)
	m = length(y);
	
	J = 1 / (2 * m) * (X * theta - y)' * (X * theta - y);
