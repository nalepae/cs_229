% Learning rate
ALPHA = 0.01;

% Polynomial degree
DEGREE = 3;

% Number of iterations
NB_ITERATION = 2000;

% Initial theta
theta = zeros(DEGREE + 1, 1);

% Data files
datas = load('datas_1_feature.txt');

x = datas(:, 1);
y = datas(:, 2);

X = ones(length(x), 1);

for i = 1 : DEGREE
	X = [X x.^i];
end

% Determine minum x and maximum x
x_min = min(x);
x_max = max(x);

% History of J
J_hist(1) = linear_cost_function(X, theta, y);

% Gradient descent
for i = 1 : NB_ITERATION
	% Compute new theta and J
	[theta, J] = gradient_descent (X, theta, y, ALPHA);
	J_hist(i + 1) = J;
end

% Open figures

% Plot J
figure('name', 'Linear regression', 'NumberTitle', 'off');
subplot(2, 1, 1);
xlabel('Iteration number');
ylabel('Cost');
plot(1:length(J_hist), J_hist, 'o');

% Plot data set and regression curve
subplot(2, 1, 2);

x_vec = x_min : 0.01 : x_max;

for j = 1 : length(x_vec)
	y_vec(j) = hypothesis_1_feature(x_vec(j), theta);
end

plot(x, y, 'o', 'markerfacecolor', 'r', 'markersize', 10);
hold on;
plot(x_vec, y_vec, 'linewidth', 2);
legend('Data set', 'Linear regression curve');

input('Press any key to exit ...');

