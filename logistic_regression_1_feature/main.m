%%%%%%%%%%%%%%
% PARAMETERS %
%%%%%%%%%%%%%%

% Data file
DATA_FILE = 'datas.txt';

% Learning rate
ALPHA = 1 * 10^-2;

% Degree

% Example, if degree = 3
% z = theta_O * x^0 +
%     theta_1 * x^1 +
%     theta_2 * x^2 +
%     theta_3 * x^3
DEGREE = 1;

% Number of iterations
LAST_ITERATION = 1 * 10^4;

%%%%%%%%%%%%%%%%%%%%%
% END OF PARAMETERS %
%%%%%%%%%%%%%%%%%%%%%

%%%%%%%%%%%%%%%%%%%
% PLOT PARAMETERS %
%%%%%%%%%%%%%%%%%%%

% For plotting cost = f(theta_x, theta_y),
% the choice of theta is needed.
NUM_THETA_X = 0;
NUM_THETA_Y = 1;

%%%%%%%%%%%%%%%%%%%%%%%%%%
% END OF PLOT PARAMETERS %
%%%%%%%%%%%%%%%%%%%%%%%%%%

% Load data file
datas = load(DATA_FILE);

% Number of examples
m = size(datas)(1);

% Create matrix X and vector y
x = datas(:, 1);

X = create_x_matrix(x, DEGREE);
y = datas(:, 2);

% Initialise theta with 0
theta = zeros(DEGREE + 1, 1);

% Compute gradient descent
[theta, theta_history, J_history] = gradient_descent(X, y, theta, ALPHA, LAST_ITERATION);

%%%%%%%%
% Plot %
%%%%%%%%
figure;

% General computations
%%%%%%%%%%%%%%%%%%%%%%

% Computation for plotting curve h = f(x)
min_x = min(X(:, 2));
max_x = max(X(:, 2));

x_lin = linspace(min_x, max_x, 100)';
x_mat = create_x_matrix(x_lin, DEGREE);
h_lin = sigmoid(x_mat * theta);

[theta_x_lin, theta_y_lin, J_mesh] = compute_mesh_cost(NUM_THETA_X, NUM_THETA_Y,
                                                       X, y,
                                                       theta_history);
% Plot of cost
%%%%%%%%%%%%%%
subplot(2, 2, 1);

hold on;
grid on;

xlabel('Itetation number');
ylabel('Cost J');
plot(J_history);

hold off;

% Plot of training examples and hypothesis function
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
subplot(2, 2, 2);

hold on;
grid on;

xlabel('Feature');
ylabel('Hypothesis and training examples');

% Find indices of positive and negative examples
pos = find(y == 1);
neg = find(y == 0);

% Plot examples
plot(X(pos, 2), y(pos), 'k+', 'LineWidth', 2, 'MarkerSize', 7);
plot(X(neg, 2), y(neg), 'ko', 'MarkerFaceColor', 'y', 'MarkerSize', 7);

% Plot hypothesis
plot(x_lin, h_lin);

hold off;

% Plot Cost = f(theta_O, theta_1)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
subplot(2, 2, 3);

hold on;
grid on;

xlabel('\theta_0');
ylabel('\theta_1');
zlabel('Cost J');

mesh(theta_x_lin, theta_y_lin, J_mesh);

hold off;

% Plot of cost contour
%%%%%%%%%%%%%%%%%%%%%%
subplot(2, 2, 4);

hold on;
grid on;

title('Contour plot of cost and history of theta');
xlabel('\theta_0');
ylabel('\theta_1');

contour(theta_x_lin, theta_y_lin, J_mesh, 30);
plot(theta_history(NUM_THETA_X + 1, :), theta_history(NUM_THETA_Y + 1, :), 
     'LineWidth', 2);

hold off;

% Wait the user to press a key to exit
input('Press any key to exit ...');