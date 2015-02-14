%%%%%%%%%%%%%%
% PARAMETERS %
%%%%%%%%%%%%%%

% Data file
DATA_FILE = 'datas.txt';

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

% Learning rate (No effect if ALGORITHM = 1)
ALPHA = 1 * 10^-2;

% Degree

% Example, if degree = 3
% z = theta_O * x^0 +
%     theta_1 * x^1 +
%     theta_2 * x^2 +
%     theta_3 * x^3
DEGREE = 1;

% Number of iterations( No effect if ALGORITHM = 1)
LAST_ITERATION = 1 * 10^4;

% Regularization parameter
LAMBDA = 0;

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

x = datas(:, 1);

X = create_x_matrix(x, DEGREE);
y = datas(:, 2);

% Initialise theta with 0
theta_init = zeros(DEGREE + 1, 1);

% Compute gradient descent
if (ALGORITHM == 0)
    [theta, theta_history, J_history] = gradient_descent(X, y, theta_init,
                                                         ALPHA,
                                                         LAST_ITERATION,
                                                         LAMBDA);
else
    options = optimset('GradObj', 'on');
    theta = fminunc(@(t)(cost_function(t, X, y, LAMBDA)), theta_init, options);
end


% Compute theta with normal equation
n = length(theta);

R = diag([0 ; ones(n - 1, 1)]);

theta_normal_eq = inv(X' * X + LAMBDA * R) * X' * y;

%%%%%%%%
% Plot %
%%%%%%%%
figure ;

% General computations
%%%%%%%%%%%%%%%%%%%%%%

% Computation for plotting curve h = f(x)
min_x = min(X(:, 2));
max_x = max(X(:, 2));

x_lin = linspace(min_x, max_x, 100)';
X_lin = create_x_matrix(x_lin, DEGREE);

y_lin = X_lin * theta;
y_lin_normal_eq = X_lin * theta_normal_eq;

% Computation for plotting curve Cost = f(theta_0, theta_1)
[theta_x_lin, theta_y_lin, J_mesh] = compute_mesh_cost(NUM_THETA_X,
                                                       NUM_THETA_Y, X, y,
                                                       theta_init, theta,
                                                       LAMBDA);

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

% Plot of training samples and hypothesis function
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
if (ALGORITHM == 0)
    subplot(2, 2, 2);
else
    subplot(1, 2, 2);
end

hold on;
grid on;

title('+ : Training examples - Blue : Linear regression - Green : Normal equation');
xlabel('Feature');
ylabel('Hypothesis and training examples');

% PLot examples
plot(x, y, 'k+', 'LineWidth', 2, 'Markersize', 7);

% Plot hypothesis
plot(x_lin, y_lin, 'LineWidth', 2);
plot(x_lin, y_lin_normal_eq, 'LineWidth', 2, 'g');

hold off;

% Plot of Cost = f(theta_0, theta_1)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
if (ALGORITHM == 0)
    subplot(2, 2, 3);
else
    subplot(1, 2, 1);
end

hold on;
grid on;

xlabel('\theta_0');
ylabel('\theta_1');
zlabel('Cost');

mesh(theta_x_lin, theta_y_lin, J_mesh);

hold off;

% Plot of cost contour
%%%%%%%%%%%%%%%%%%%%%%
if (ALGORITHM == 0)
    subplot(2, 2, 4);
    hold on;
    grid on;

    title('Contour plot of cost and history of theta');
    xlabel('\theta_0');
    ylabel('\theta_1');

    contour(theta_x_lin, theta_y_lin, J_mesh, 20);
    plot(theta_history(NUM_THETA_X + 1, :), theta_history(NUM_THETA_Y + 1, :), 'LineWidth', 2);

    hold off;
end

% Wait the user to press a key to exit
printf('Press any key to exit ...\n');
pause();
