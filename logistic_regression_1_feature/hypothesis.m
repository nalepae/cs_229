%% hypothesis: Hypothesis function
function h = hypothesis(x, theta)
	h = 1 / (1 + exp(-theta' * x));
