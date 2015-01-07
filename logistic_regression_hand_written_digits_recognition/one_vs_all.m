%%one_vs_all: Compute linear regrassion for each of the 'num_labels' class
%% all_theta is a matrix where each row represents the learning parameters
%% for the each label
function all_theta = one_vs_all(X, y, num_labels, lambda)
    % Some useful variables
    n = size(X, 2);

    % You need to return the following variables correctly 
    all_theta = zeros(num_labels, n);

    % Set Initial theta 
    initial_theta = zeros(n, 1);

    % Set options for fmincg
    options = optimset('GradObj', 'on', 'MaxIter', 50);

    for c = 0:num_labels - 1
        printf('Training class %d ... ', c);
        % Compute gradient descent
        theta = fminunc(@(t)(cost_function(t, X, y == c, lambda)),
                                           initial_theta, options);

        % Add theta vector to the all_theta matrix
        all_theta(c + 1, :) = theta';
        printf('OK\n');
    end
end