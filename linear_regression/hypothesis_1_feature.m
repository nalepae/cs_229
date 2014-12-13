%% hypothesis_1_feature: Hypothesis for 1 feature polynomial function
function h = hypothesis_1_feature(x, theta)
	n = length(theta);
	h = 0;

	for i = 1 : n
		x_vec(i) = x .** (i - 1);
	end

	h = x_vec * theta;
