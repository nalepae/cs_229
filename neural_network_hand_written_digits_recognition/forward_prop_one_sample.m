%% forward_prop_one_sample: Forward propagation for one sample
function [a_1, z_2, a_2, z_3, h] = forward_prop_one_sample (i, X_t, theta_1,
                                                            theta_2)
    % Compute the activation of the input layer
    a_1 = X_t(:, i);

    % Compute the activation of the hidden layer
    z_2 = theta_1 * a_1;
    a_2 = [1 ; sigmoid(z_2)];

    % Compute the activation of the output layer
    z_3 = theta_2 * a_2;
    h = sigmoid(z_3);
