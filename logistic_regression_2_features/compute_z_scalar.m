%% compute_z_scalar: z computation function for scalars x1 and x2
function z = compute_z_scalar(x1, x2, theta)
	n = length(theta);

	% From the lenght of theta, compute the degree max of pylonomial
	degree_max = (sqrt(1 + 8 * n) - 1) / 2 - 1;

	% vec_x index
	k = 1;

	% Compute vec_x
	for i = 0:degree_max
		for j = 0:i
			vec_x(k) = x1 ^ j * x2 ^ (i - j);
			k++;
		end
	end 

	z = vec_x * theta;
end
