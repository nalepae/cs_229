%%%%%%%%%%%%%%
% PARAMETERS %
%%%%%%%%%%%%%%

% Learning rate
ALPHA = 0.5;

% Initial theta
%INITIAL_THETA = [-5 ; -1];
INITIAL_THETA = [0 ; -2];

% Number of iterations
NB_ITERATION = 2000;

%%%%%%%%%%%%%%%%%%%%%
% END OF PARAMETERS %
%%%%%%%%%%%%%%%%%%%%%

% Initial theta
theta = INITIAL_THETA;

% Data files
datas = load('datas_1_feature.txt');

x = datas(:, 1);
y = datas(:, 2);

X = [ones(length(x), 1) x];

% Determine minum x and maximum x
x_min = min(x);
x_max = max(x);

% Open figures
figure('name', 'Logistic regression', 'NumberTitle', 'off');
subplot(2, 2, 1);
xlabel('Iteration number');
ylabel('Cost')
grid on;

% % Mesh cost function in terms on theta(1) and theta(2)
subplot(2, 2, 3);
theta_1 = linspace(-10, 10, 100)';
theta_2 = linspace(-10, 10, 100)';
[mesh_theta_1, mesh_theta_2] = meshgrid(theta_1, theta_2);

for i = 1 : size(mesh_theta_1)(1)
	for j = 1 : size(mesh_theta_1)(2)
		mesh_theta = [mesh_theta_1(i, j) ; mesh_theta_2(i, j)];
		mesh_J(i, j) = cost_function(X, mesh_theta, y);
	end
end

mesh(theta_1, theta_2, mesh_J);
xlabel('\theta_0');
ylabel('\theta_1');
zlabel('Cost');

% Plot iso-cost curves
subplot(2, 2, 4);
contour(mesh_theta_1, mesh_theta_2, mesh_J);
hold on;
grid on;
xlabel('\theta_0');
ylabel('\theta_1');
title('Iso-cost curves');

% Gradient descent
J = cost_function(X, theta, y);
J_history(1) = J;

for i = 1 : NB_ITERATION
	% Plot J
	subplot(2, 2, 1);
	plot(1:i, J_history, 'o');
	xlabel('Iteration number');
	ylabel('Cost')
	grid on;

	% Plot data set and regression curve
	subplot(2, 2, 2);
	hold off;
	plot(x, y, 'o', 'markerfacecolor', 'r', 'markersize', 10);
	hold on;
	grid on;

	x_vec = x_min : 0.01 : x_max;

	for j = 1 : length(x_vec)
		y_vec(j) = hypothesis([1 ;x_vec(j)], theta);
	end

	plot(x_vec, y_vec, 'linewidth', 2);
	legend('Data set', 'Linear regression curve');

	% Plot theta in iso-cost curve
	subplot(2, 2, 4);
	plot(theta(1), theta(2));

	% Compute new theta and J
	[theta, J] = gradient_descent (X, theta, y, ALPHA);
	J_history(i + 1) = J;
	
	sleep(0.01);
end

input('Press any key to exit ...');

