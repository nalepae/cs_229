%% random_initialize_weights: Function which initialize randomly a weight matrix
%% l_in  : number of neurons of the input layer
%% l_out : number of neurons of the output layer
%% weight_matrix : [l_out * l_in] matrix with random numbers
%% Each element of W is randmoly set between [- epsilon ; + epsilon], where
%% epsilon = sqrt(6 / (l_in + l_out)) 

function weights_matrix = random_initialize_weights(l_in, l_out)
	epsilon = sqrt(6 / (l_in + l_out));
	weights_matrix = (2 * epsilon) * rand(l_out, l_in) - epsilon;
end