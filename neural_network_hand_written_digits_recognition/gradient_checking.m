%% gradient_checking: Gradient Checking
function gradient_checking(y, theta, m, hidden_layer_size, n, num_labels,
                           EPSILON, X_t, lambda, grad, iteration_number)
    for i = 1 : length(theta)
        theta_plus = theta;
        theta_plus(i) += EPSILON;

        theta_minus = theta;
        theta_minus(i) -= EPSILON;

        J_theta_plus = cost_function(y, theta_plus, m, hidden_layer_size, n,
                                     num_labels, X_t, lambda);

        J_theta_minus = cost_function(y, theta_minus, m, hidden_layer_size, n,
                                      num_labels, X_t, lambda);

        grad_approx(i) = (J_theta_plus - J_theta_minus) / (2 * EPSILON);

        printf('Iteration number : %d - grad(%d) : %f - grad_approx(%d) : %f\n',
               iteration_number, i, grad(i), i, grad_approx(i));
    end
end
