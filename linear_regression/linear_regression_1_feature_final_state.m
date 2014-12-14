%%%%%%%%%%%%%%
% PARAMETERS %
%%%%%%%%%%%%%%

% Learning rate
ALPHA = 0.01;

% Initial theta
INITIAL_THETA = [4 ; 1];

% Number of iterations
NB_ITERATION = 200;

%%%%%%%%%%%%%%%%%%%%%
% END OF PARAMETERS %
%%%%%%%%%%%%%%%%%%%%%

% Polynomial degree
degree = 1;

% Initial theta
theta = INITIAL_THETA;

% Data files
datas = load('datas_1_feature_linear.txt');

x = datas(:, 1);
y = datas(:, 2);

X = ones(length(x), 1);

for i = 1 : degree
	X = [X x.^i];
end

% Determine minum x and maximum x
x_min = min(x);
x_max = max(x);

% History of J
J_hist(1) = linear_cost_function(X, theta, y);

% History of theta
theta_0_hist(1) = theta(1);
theta_1_hist(1) = theta(2);

% Gradient descent
for i = 1 : NB_ITERATION
	% Compute new theta and J
	[theta, J] = gradient_descent (X, theta, y, ALPHA);
	J_hist(i + 1) = J;

	theta_0_hist(i + 1) = theta(1);
	theta_1_hist(i + 1) = theta(2);
end

% Open figures

% Plot J
figure('name', 'Linear regression', 'NumberTitle', 'off');
subplot(2, 2, 1);
xlabel('Iteration number');
ylabel('Cost');
hold on;
grid on;
plot(1:length(J_hist), J_hist, 'o');

% Plot data set and regression curve
subplot(2, 2, 2);
grid on;
hold on;
x_vec = x_min : 0.01 : x_max;

for j = 1 : length(x_vec)
	y_vec(j) = hypothesis_1_feature(x_vec(j), theta);
end

plot(x, y, 'o', 'markerfacecolor', 'r', 'markersize', 10);
hold on;
plot(x_vec, y_vec, 'linewidth', 2);
legend('Data set', 'Linear regression curve');

% Mesh cost function in terms on theta(1) and theta(2)
subplot(2, 2, 3);
theta_1 = linspace(0, 4, 100)';
theta_2 = linspace(-3, 1, 100)';
[mesh_theta_1, mesh_theta_2] = meshgrid(theta_1, theta_2);

for i = 1 : size(mesh_theta_1)(1)
	for j = 1 : size(mesh_theta_1)(2)
		mesh_theta = [mesh_theta_1(i, j) ; mesh_theta_2(i, j)];
		mesh_J(i, j) = linear_cost_function(X, mesh_theta, y);
	end
end

mesh(theta_1, theta_2, mesh_J);
xlabel('\theta_0');
ylabel('\theta_1');
zlabel('Cost');

% Plot iso-cost curves
subplot(2, 2, 4);
contour(mesh_theta_1, mesh_theta_2, mesh_J, 40);
hold on;
grid on;
xlabel('\theta_0');
ylabel('\theta_1');
title('Iso-cost curves');

plot(theta_0_hist, theta_1_hist, 'o');

input('Press any key to exit ...');

