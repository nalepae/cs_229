% Learning rate
ALPHA = 0.2;

% Polynomial degree
DEGREE = 3;

% Number of iterations
NB_ITERATION = 100000;

% Initial theta
theta = zeros(DEGREE + 1, 1);

% Data files
file_x = fopen('ex5Linx.dat', 'r');
file_y = fopen('ex5Liny.dat', 'r');

x = dlmread(file_x);
y = dlmread(file_y);

X = ones(length(x), 1);

for i = 1 : DEGREE
	X = [X x.^i];
end

% Determine minum x and maximum x
x_min = min(x);
x_max = max(x);

% Open figures
figure('name', 'Linear regression', 'NumberTitle', 'off');
subplot(2, 1, 1);
xlabel('Iteration number');
ylabel('Cost')

subplot(2, 1, 2);

J = linear_cost_function(X, theta, y);

% Gradient descent
for i = 1 : NB_ITERATION
	% Plot data set and regression curve
	subplot(2, 1, 2);
	hold off;
	plot(x, y, 'o', 'markerfacecolor', 'r', 'markersize', 10);
	hold on;

	x_vec = x_min : 0.01 : x_max;

	for j = 1 : length(x_vec)
		y_vec(j) = hypothesis_1_feature(x_vec(j), theta);
	end

	plot(x_vec, y_vec, 'linewidth', 2);
	legend('Data set', 'Linear regression curve');

	% Plot J
	subplot(2, 1, 1);
	hold on;
	plot(i, J);

	% Compute new theta and J
	[theta, J] = gradient_descent (X, theta, y, ALPHA);
	
	sleep(0.01);
end

input('Press any key to exit ...');

