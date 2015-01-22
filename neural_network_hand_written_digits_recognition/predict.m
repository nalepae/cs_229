%%predict: Predict the label of the hand written number in X matrix
%% thanks to all_theta learning parameters
function [max_prob, p] = predict(theta_1, theta_2, X)
	% Number of digits to find label
	m = size(X, 1);

	% Compute the transposed of X
	X_t = X';

	% Number of labels
	num_labels = size(theta_2, 1);

	% Compute the activation of the input layer
	a_1 = X_t;

	% Compute the activation of the hidden layer
	z_2 = theta_1 * a_1;
	a_2 = [ones(1, size(z_2, 2)) ; sigmoid(z_2)];

	% Compute the activation of the output layer
	z_3 = theta_2 * a_2;
	h = sigmoid(z_3);

	% Determine which label is the most probable
	[max_prob, p_plus_1] = max(h', [], 2);

	% Arrays in octave cannot begin with 0.
	% So label i is in the (i+1)th column
	p = p_plus_1 - 1;
end
