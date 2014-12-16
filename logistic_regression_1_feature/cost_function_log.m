%% linear_cost_function_log: Linear cost function
function J = cost_function_log(X, theta, y)
	m = length(y);
	
	J = 0;

	for i = 1 : m
		h = hypothesis(theta, X(i, :)');
		J += y(i) * log(h) * (1 - y(i) * log(1 - h));
	end

	J /= -m;