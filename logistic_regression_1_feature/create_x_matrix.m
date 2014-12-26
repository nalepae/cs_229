%% create_x_matrix: Create X matrix function for 2 (two) features and degree n
function X = create_x_matrix(x, degree)
	% Example :
	% - n = 3
	% X = [x^0, x^1, x^2, x^3]
	
	% Number of examples
	m = length(x);

	% Create column 1 for x^0 * y^0
	X = ones(m, 1);

	% Create X for others degrees
	for i = 1 : degree
			X = [X  x .^i];
	end
end

