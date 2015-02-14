%% cost_function: Compute the cost function and its gradient
function [J grad] = cost_function(params, n,
                                  hidden_layer_size, num_labels,
                                  X, y, lambda, gradient_checking)

	% params contains the weights matrices (between the input layer and the
	% hidden layer and between the hidden layer and the output layer) in one
    % linear vector.
	%
	%This part reshape this one linear vector in two matrices
	theta_1 = reshape(params(1 : hidden_layer_size * n),
		              hidden_layer_size, n);

	theta_2 = reshape(params(1 + hidden_layer_size * n : end),
		              num_labels, hidden_layer_size + 1);

	% Compute the number of examples to train
	m = size(X, 1);

	% Compute the transposed of X
	X_t = X';

	% Initialize probabilistic part of cost
	J_probablistic_part = 0;

	% Compute some matrices size
	size_theta_1 = size(theta_1);
	size_theta_2 = size(theta_2);

	% Initialize the probabilistic part of the two layers dependant gradient
	% matrices (grad_1 for input layer to hidden layer and grad_2 for hidden
	% layer to output layer)
	grad_1_probabilistic_part = zeros(size_theta_1);
	grad_2_probabilistic_part = zeros(size_theta_2);

	% For each example, compute the forward and the back propagation
	for i = 1 : m
		% Compute y_i vector
		% For example, if num_label = 10 and
		% - if y(i) = 2, then y_i = [0 ; 0 ; 1 ; 0 ; 0 ; 0 ; 0 ; 0 ; 0]
		% - if y(i) = 5, then y_i = [0 ; 0 ; 0 ; 0 ; 0 ; 1 ; 0 ; 0 ; 0]
		% - if y(i) = 0, then y_i = [1 ; 0 ; 0 ; 0 ; 0 ; 0 ; 0 ; 0 ; 0]
		y_i = zeros(num_labels, 1);
		y_i(y(i) + 1) = 1;

		%%%%%%%%%%%%%%%%%%%%%%%%%%%%
		% Forward propagation part %
		%%%%%%%%%%%%%%%%%%%%%%%%%%%%

		% Compute the activation of the input layer
		a_1 = X_t(:, i);

		% Compute the activation of the hidden layer
		z_2 = theta_1 * a_1;
		a_2 = [1 ; sigmoid(z_2)];

		% Compute the activation of the output layer
		z_3 = theta_2 * a_2;
		h = sigmoid(z_3);

		% Compute the probabilistic part of the cost
		J_probablistic_part -= y_i' * log(h) + (1 - y_i)' * log(1 - h);

		%%%%%%%%%%%%%%%%%%%%%%%%%
		% Back propagation part %
		%%%%%%%%%%%%%%%%%%%%%%%%%

		% Compute delta_3
		delta_3 = h - y_i;

		% Compute delta_2
		delta_2 = [theta_2' * delta_3 .* [1 ; sigmoid_derivative(z_2)]](2 : end);

		% Compute grad_2_probabilistic_part
		grad_2_probabilistic_part += delta_3 * a_2';

		% Compute grad_1_probabilistic_part
		grad_1_probabilistic_part += delta_2 * a_1';
	end

	% Compute the regularization part of the two layers dependant gradient
	% matrices (grad_1 for input layer to hidden layer and grad_2 for hidden
	% layer to output layer)
	grad_2_regularization_part = lambda * [zeros(size_theta_2, 1) ...
	                                       theta_2(:, 2 : end)];

	grad_1_regularization_part = lambda * [zeros(size_theta_1, 1) ...
		                                   theta_1(:, 2 : end)];

	% Compute the two layers dependent gradient matrices (grad_1 for input
	% layer to hidden layer and grad_2 for hidden layer to output layer)
	grad_1 = 1 / m * (grad_1_probabilistic_part + grad_1_regularization_part);
	grad_2 = 1 / m * (grad_2_probabilistic_part + grad_2_regularization_part);

	% Compute the regularization part of the cost
	partial_regularization_part = 0;

	for i = 1 : hidden_layer_size
		partial_regularization_part += theta_1(i, 2 : end) * ...
		                               theta_1(i, 2 : end)';
    end

    for i = 1 : num_labels
    	partial_regularization_part += theta_2(i, 2 : end) * ...
    	                               theta_2(i, 2 : end)';
    end

	J_regularization_part = lambda / 2 * partial_regularization_part;

	% Compute the cost
	J = 1 / m * (J_probablistic_part + J_regularization_part);

	% Unroll the gradient
	grad = [grad_1(:) ; grad_2(:)];

	% Persistent variable for the iteration number
	persistent iteration_number = 1

	printf('Iteration : %d - Cost : %f\r', iteration_number, J);

	% Incrementation of iteration number
	iteration_number += 0.5;
end
