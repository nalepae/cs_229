% This files aims to compute hand written numbers with a neural network.
% The used neural network has exactly 3 (three) layers :
% - 1 input layer
% - 1 hidden layer
% - 1 output layer
%
% The input layer has 401 neurons :
% - Each image of hand written digit is a 20x20 = 400 array of pixels
%   (the value of each pixel is between 0 and 1)
% - An offset neuron is also added
%
% The user can choose the number of neurons of the hidden layer
%
% TO DO : Implement gradient checking to be sure that there is
% no bug in back propagation.

%%%%%%%%%%%%%%
% PARAMETERS %
%%%%%%%%%%%%%%

% Data file
DATA_FILE = 'datas.mat';

% Regularization parameter
LAMBDA = 0.1;

% Size of the hidden layer
HIDDEN_LAYER_SIZE = 25;

% Number of digits to guess
NUM_DIGIT_TO_GUESS = 25;

% Max number of iterations
ITERATIONS_MAX_NUMBER = 50;

% Gradient checking
% 0 : No
% 1 : Yes
GRADIENT_CHECKING = 0;

%%%%%%%%%%%%%%%%%%%%%
% END OF PARAMETERS %
%%%%%%%%%%%%%%%%%%%%%

% Load workspace
%
% digits_matrix is a (m x n) matrix where each row represents
% a hand written digit and each column of that row represents a pixel
% of this hand written digit.
% Here, m = 5000 and n = 400. That means there is 5000 hand written digits
% each composed of 400 pixels.
% (Each hand written digit is a picture of 40 * 40 pixels)
%
% digits_values is a vector corresponding the the matrix 'digits_matrix'.
% Each element of this vector represents the value of the corresponding row
% of the matrix 'digits_matrix'.
% For example, if the row i of 'digit_matrix' is a hand written digit
% respresenting the number "4", then digits_values(i) = 4.
load('datas.mat')

% Number of different possibles labels
% (Labels are 0, 1, 2, 3, 4, 5, 6, 7, 8 and 9)
num_labels = 10;

% Number of digits
num_digits = size(digits_matrix, 1);

% Randomly select NUM_DIGIT_TO_GUESS to display
rand_indices = randperm(num_digits);

% Digits the program has to guess the value
digits_to_guess = digits_matrix(rand_indices(1:NUM_DIGIT_TO_GUESS), :);

% Digits used for training
training_digits = digits_matrix(rand_indices(NUM_DIGIT_TO_GUESS+1 : ...
                                             num_digits), :);

% Number of training examples
m = size(training_digits, 1);

% Create X matrix and y vector from training digits
X = [ones(m, 1) training_digits];
y = digits_values(rand_indices(NUM_DIGIT_TO_GUESS+1:num_digits));

% Compute size of layers
input_layer_size = size(training_digits);

% Compute number of features
n = size(X, 2);

% Initialize the weights
initial_theta_1 = random_initialize_weights(HIDDEN_LAYER_SIZE, n);

initial_theta_2 = random_initialize_weights(num_labels,
	                                        HIDDEN_LAYER_SIZE + 1);

% Unroll initial parameters
initial_params = [initial_theta_1(:) ; initial_theta_2(:)];

% Set options for fmincg
options = optimset('GradObj', 'on', 'MaxIter', ITERATIONS_MAX_NUMBER);

% Train the neural network
[params cost] = fminunc(@(p) cost_function(p, n, HIDDEN_LAYER_SIZE, num_labels,
                                           X, y, LAMBDA, GRADIENT_CHECKING),
                        initial_params, options);
printf('\n');

% Convert theta_1 and theta_2 from vectorial shape to matrices shapes
theta_1 = reshape(params(1 : HIDDEN_LAYER_SIZE * n), HIDDEN_LAYER_SIZE, n);
theta_2 = reshape(params(HIDDEN_LAYER_SIZE * n + 1 : end), num_labels,
	                                                       HIDDEN_LAYER_SIZE + 1);

% Create X_predict matrix for digits to find label
X_predict = [ones(NUM_DIGIT_TO_GUESS, 1) digits_to_guess];

[max_prob, p] = predict(theta_1, theta_2, X_predict);

% Print p under the same matrix shape than the hand written digits
len_p = length(p);
display_rows = floor(sqrt(len_p));
display_cols = ceil(len_p / display_rows);

mat_max_prob = round(reshape(max_prob, display_cols, display_rows)' * 100);
mat_p = reshape(p, display_cols, display_rows)';

printf('\n');
mat_p

% Uncomment if you want to print the matrix of probabilities
%mat_max_prob

% Display the digits to guess
display_data(digits_to_guess);

% Wait the user to press a key to exit
printf('Press any key to exit ...\n');
pause();
