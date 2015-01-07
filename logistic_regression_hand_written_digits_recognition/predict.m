%%predict: Predict the label of the hand written number in X matrix
%% thanks to all_theta learning parameters
function [max_prob, p] = predict(all_theta, X)
	% Number of digits to find label
	m = size(X, 1);

	% Number of labels
	num_labels = size(all_theta, 1);

	% The [row i - column j] of matrix 'sig_X_all_theta' contains
	% probability for the example j to behave to the class j.
	sig_X_all_theta = sigmoid(X * all_theta');
	
	% Determine which label is the most probable
	[max_prob, p_plus_1] = max(sig_X_all_theta, [], 2);

	% Arrays in octave cannot begin with 0.
	% So label i is in the (i+1)th column
	p = p_plus_1 - 1;
end
