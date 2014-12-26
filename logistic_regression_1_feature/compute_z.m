%% compute_z: z computation function for matrix x1 and x2
function z = compute_z(x1, x2, theta)
	z = arrayfun(@(x, y) compute_z_scalar(x, y, theta), x1, x2);