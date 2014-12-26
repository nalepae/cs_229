%% create_x_matrix: Create X matrix function for 2 (two) features and degree n
function X = create_x_matrix(x1, x2, degree)
	% Example :
	% - n = 3
	% - x feature 1
	% - y feature 2
	% X = [x^0 * y^0,
	%      x^0 * y^1, x^1 * y^0,
	%      x^0 * y^2, x^1 * y^1, x^2 * y^0,
	%      x^0 * y^3, x^1 * y^2, x^2 * y^1, x^3 * y^0]
	
	% Number of examples
	m = length(x1);

	% Create column 1 for x^0 * y^0
	X = ones(m, 1);

	% Create X for others degrees
	for i = 1 : degree
		for j = 0 : i
			X = [X  x1 .^ j .* x2 .^ (i - j)];
		end
	end
end

