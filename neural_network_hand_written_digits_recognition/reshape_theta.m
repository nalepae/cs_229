%% reshape_theta: Theta Reshape function
function [theta_1, theta_2] = reshape_theta(theta, hidden_layer_size, n,
                                            num_labels)
    theta_1 = reshape(theta(1 : hidden_layer_size * n),
                      hidden_layer_size, n);

    theta_2 = reshape(theta(1 + hidden_layer_size * n : end),
                      num_labels, hidden_layer_size + 1);
