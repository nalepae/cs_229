% Polynomial degree
DEGREE = 3;

% Data files
datas = load('datas_1_feature.txt');

x = datas(:, 1);
y = datas(:, 2);

X = ones(length(x), 1);

X = ones(length(x), 1);

for i = 1 : DEGREE
	X = [X x.^i];
end

% Determine minum x and maximum x
x_min = min(x);
x_max = max(x);

% Normal equation
theta = inv(X' * X) * X' * y;

% Open figure
x_vec = x_min : 0.01 : x_max;

for j = 1 : length(x_vec)
	y_vec(j) = hypothesis_1_feature(x_vec(j), theta);
end

figure('name', 'Normal equation', 'NumberTitle', 'off');
plot(x, y, 'o', 'markerfacecolor', 'r', 'markersize', 10);
hold on;
plot(x_vec, y_vec, 'linewidth', 2);
legend('Data set', 'Linear regression curve');

input('Press any key to exit ...');

