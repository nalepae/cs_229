%% cost_function: Cost function
function J = cost_function(y, theta, m, hidden_layer_size, n, num_labels, X_t,
                           lambda)
    % Transform theta vector in two matrices
    [theta_1, theta_2] = reshape_theta(theta, hidden_layer_size, n,
                                       num_labels);

    % Initialize probabilistic part of cost
    J_probablistic_part = 0;

    % Compute probabilistic part of cost
    for i = 1 : m
        % Compute y_i vector
        % For example, if num_label = 10 and
        % - if y(i) = 2, then y_i = [0 ; 0 ; 1 ; 0 ; 0 ; 0 ; 0 ; 0 ; 0]
        % - if y(i) = 5, then y_i = [0 ; 0 ; 0 ; 0 ; 0 ; 1 ; 0 ; 0 ; 0]
        % - if y(i) = 0, then y_i = [1 ; 0 ; 0 ; 0 ; 0 ; 0 ; 0 ; 0 ; 0]
        y_i = zeros(num_labels, 1);
        y_i(y(i) + 1) = 1;

        %%%%%%%%%%%%%%%%%%%%%%%%%%%%
        % Forward propagation part %
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%
        [a_1, z_2, a_2, z_3, h] = forward_prop_one_sample(i, X_t, theta_1,
                                                          theta_2);

        % Compute the probabilistic part of the cost
        J_probablistic_part -= y_i' * log(h) + (1 - y_i)' * log(1 - h);
    end

    % Compute the regularization part of the cost
    partial_regularization_part = 0;

    for i = 1 : hidden_layer_size
        partial_regularization_part += theta_1(i, 2 : end) * ...
                                       theta_1(i, 2 : end)';
    end

    for i = 1 : num_labels
        partial_regularization_part += theta_2(i, 2 : end) * ...
                                       theta_2(i, 2 : end)';
    end

    J_regularization_part = lambda / 2 * partial_regularization_part;

    % Compute the cost
    J = 1 / m * (J_probablistic_part + J_regularization_part);
