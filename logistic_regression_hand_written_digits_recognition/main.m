%%%%%%%%%%%%%%
% PARAMETERS %
%%%%%%%%%%%%%%

% Data file
DATA_FILE = 'datas.mat';

% Regularization parameter
LAMBDA = 0.1;

% Number of digits to guess
NUM_DIGIT_TO_GUESS = 25;

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
training_digits = digits_matrix(rand_indices(NUM_DIGIT_TO_GUESS+1:num_digits), 
	                            :);

% Number of training examples
m = size(training_digits, 1);

% Create X matrix and y vector from training digits
X = [ones(m, 1) training_digits];
y = digits_values(rand_indices(NUM_DIGIT_TO_GUESS+1:num_digits));

% Compute logistic regression for each label
all_theta = one_vs_all(X, y, num_labels, LAMBDA);

% Create X_predict matrix for digits to find label
X_predict = [ones(NUM_DIGIT_TO_GUESS, 1) digits_to_guess];

% Compute the labels
[max_prob, p] = predict(all_theta, X_predict);

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
input('Press any key to exit ...');
