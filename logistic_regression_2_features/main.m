%%%%%%%%%%%%%%
% PARAMATERS %
%%%%%%%%%%%%%%

% Data file
DATA_FILE  = 'datas.txt';

% Learning rate
ALPHA = 1 * 10^-9;

% Degree

% Example, if degree = 3,
% z = theta_O * x^0 * y^0 + 
%     theta_1 * x^1 * y^0 + theta_2 * x^0 + y^1 +
%     theta_3 * x^2 * y^0 + theta_4 * x^1 * y^1 + theta_5 * x^1 * y^2 +
%     theta_6 * x^3 * y^1 + theta_7 * x^2 * y^1 + theta_8 * x^1 * y^2 +
%                                                 theta_9 * x^0 * y^3
DEGREE = 2;

% Number of iterations
LAST_ITERATION = 1 * 10^5;

%%%%%%%%%%%%%%%%%%%%%
% END OF PARAMETERS %
%%%%%%%%%%%%%%%%%%%%%

% Load data file
datas = load(DATA_FILE);

% Number of examples
m = size(datas)(1);

% Create matrix X and vector y
x1 = datas(:, 1);
x2 = datas(:, 2);

X = create_x_matrix(x1, x2, DEGREE);
y = datas(:, 3);

% Initialize theta with 0 (sum of arithmetic sequence)
theta_size = (DEGREE + 1) * (DEGREE + 2) / 2;
theta = zeros(theta_size, 1);

% Compute gradient descent
[theta, J_history] = gradient_descent(X, y, theta, ALPHA, LAST_ITERATION);

%%%%%%%%
% Plot %
%%%%%%%%
figure;

% General computations
%%%%%%%%%%%%%%%%%%%%%%

% Computation for ploting 3D curve z = f(features)
min_x1 = min(X(:, 2));
max_x1 = max(X(:, 2));

min_x2 = min(X(:, 3));
max_x2 = max(X(:, 3));

x1_lin = linspace(min_x1, max_x1, 100);
x2_lin = linspace(min_x2, max_x2, 100);

[mesh_x1, mesh_x2] = meshgrid(x1_lin, x2_lin);
z = compute_z(mesh_x1, mesh_x2, theta);

% Plot of cost
%%%%%%%%%%%%%%
subplot(2, 2, 1);

hold on;
grid on;

xlabel('Iteration number');
ylabel('Cost J');
plot(J_history);

hold off;

% Plot of training examples and decision boundary
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
subplot(2, 2, 2);

hold on;
grid on;

xlabel('Feature 1');
ylabel('Feature 2');

% Find indices of positive and negative examples
pos = find(y == 1);
neg = find(y == 0);

% Plot examples
plot(X(pos, 2), X(pos, 3), 'k+', 'LineWidth', 2, 'MarkerSize', 7);
plot(X(neg, 2), X(neg, 3), 'ko', 'MarkerFaceColor', 'y', 'MarkerSize', 7);

% Plot decision boundary
contour(x1_lin, x2_lin, z, [0, 0], 'LineWidth', 2);

% Legend
legend('Positive examples', 'Negative examples', 'Decision boundary');

hold off;

% Plot z = f(x1, x2)
%%%%%%%%%%%%%%%%%%%%
subplot(2, 2, 3);

hold on;
grid on;

xlabel('Feature 1');
ylabel('Feature 2');
zlabel('z');

mesh(x1_lin, x2_lin, z);

hold off;

% Wait the user to press a key to exit
input('Press any key to exit ...');