%%%%%%%%%%%%%%
% PARAMATERS %
%%%%%%%%%%%%%%

% Data file
% Available files  :
% - datas_1.txt
% - datas_2.txt
DATA_FILE  = 'datas_1.txt';

% Minimum searsh algorithm
% 0 : Gradient descent (all used functions are implemented from a to z),
%     but this method can be very slow to converge, specially when degree > 1.
%     Furthermore, you have to choose the learning rate. (Not to high
%     otherwise the cost can diverge, not to low otherwise the convergence
%     will be too slow.)
%
% 1 : Use of 'fminunc' builtin function. Less explicit than gradient descent
%     but very quick to converge
ALGORITHM = 0;

% Degree

% Example, if degree = 3,
% z = theta_O * x^0 * y^0 + 
%     theta_1 * x^1 * y^0 + theta_2 * x^0 + y^1 +
%     theta_3 * x^2 * y^0 + theta_4 * x^1 * y^1 + theta_5 * x^1 * y^2 +
%     theta_6 * x^3 * y^1 + theta_7 * x^2 * y^1 + theta_8 * x^1 * y^2 +
%                                                 theta_9 * x^0 * y^3
DEGREE = 2;

% Learning rate (No effect if ALGORITHM = 1)
% With DEGREE = 2
% For datas_1.txt : Ideal 1 * 10^-7
% For datas_2.txt : Ideal 5 * 10^3
ALPHA = 1 * 10^-7;

% Number of iterations (no effect if ALGORITHM = 1)
% For datas_1.txt : Ideal 5 * 10^6
% For datas_2.txt : Ideal 5 * 10^3
LAST_ITERATION = 5 * 10^6;

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
theta_init = zeros(theta_size, 1);

% Compute gradient descent
if (ALGORITHM == 0)
    [theta, J_history] = gradient_descent(X, y, theta_init, ALPHA,
                                          LAST_ITERATION);
else
    options = optimset('GradObj', 'on');
    [theta, J] = fminunc(@(t)(cost_function(t, X, y)), theta_init, options);
end

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
if (ALGORITHM == 0)
    subplot(2, 2, 1);

    hold on;
    grid on;

    xlabel('Iteration number');
    ylabel('Cost J');
    plot(J_history);

    hold off;
end

% Plot of training examples and decision boundary
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
if (ALGORITHM == 0)
    subplot(2, 2, 2);
else
    subplot(1, 2, 2);
end

hold on;
grid on;

title(' + : Positive examples, o : Negative examples, Green : Decision boundary');
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

hold off;

% Plot z = f(x1, x2)
%%%%%%%%%%%%%%%%%%%%
if (ALGORITHM == 0)
    subplot(2, 2, 3);
else
    subplot(1, 2, 1);
end

hold on;
grid on;

xlabel('Feature 1');
ylabel('Feature 2');
zlabel('z');

mesh(x1_lin, x2_lin, z);

hold off;

% Wait the user to press a key to exit
input('Press any key to exit ...');