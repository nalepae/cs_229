%% linear_cost_function: Linear cost function
function J = cost_function(X, theta, y)
	m = length(y);
	
	J = 0;

	for i = 1 : m
		J += (hypothesis(theta, X(i, :)') - y(i)) ** 2;
	end

	J /= 2;