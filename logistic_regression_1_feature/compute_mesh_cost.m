%% compute_mesh_cost: Mesh cost func
function [theta_x_lin, theta_y_lin, J_mesh] = compute_mesh_cost(num_theta_x,
                                                                num_theta_y,
                                                                X, y,
                                                                theta_init,
                                                                theta_final)
    % Inputs :
    % --------
    % num_theta_x : Number of the feature to be plotted on the x axis
    % num_theta_y : Number of the feature to be plotted on the y axis
    % X : Matrix of examples
    % y : Vector of results
    % theta_init : Initial theta
    % theta_final : Final theta

    % Outputs :
    % ---------
    % theta_x_lin : Linear space of x
    % theta_y_lin : Linear space of y
    % J_mesh : Mesh for J = f(theta_x_lin, theta_y_lin);

    % Number of discrete samples
    N = 100;

    % Size of theta
    n = length(theta_init);

    % Reduced X with only two features
    X_reduced = [X(:, num_theta_x + 1) ; X(:, num_theta_y + 1)];

    % Retrieve theta history
    theta_x_init = theta_init(num_theta_x + 1);
    theta_x_final = theta_final(num_theta_x + 1);
    theta_y_init = theta_init(num_theta_y + 1);
    theta_y_final = theta_final(num_theta_y + 1);

    % Compute min and max for theta_x and theta_y
    if (theta_x_init < theta_x_final)
        minimum = theta_x_init;
        maximum = theta_x_final;

        theta_x_min = minimum;
        theta_x_max = 2 * maximum - minimum;
    else
        minimum = theta_x_final;
        maximum = theta_x_init;

        theta_x_min = 2 * minimum - maximum;
        theta_x_max = maximum;
    end

    if (theta_y_init < theta_y_final)
        minimum = theta_y_init;
        maximum = theta_y_final;

        theta_y_min = minimum;
        theta_y_max = 2 * maximum - minimum;
    else
        minimum = theta_y_final;
        maximum = theta_y_init;

        theta_y_min = 2 * minimum - maximum;
        theta_y_max = maximum;
    end

    % Linear space for theta_x and theta_y
    theta_x_range = theta_x_max - theta_x_min;
    theta_y_range = theta_y_max - theta_y_min;

    theta_x_lin = linspace(theta_x_min - 0.5 * theta_x_range,
                           theta_x_max + 0.5 * theta_x_range, N);

    theta_y_lin = linspace(theta_y_min - 0.5 * theta_y_range,
                           theta_y_max + 0.5 * theta_y_range, N);

    % Mesh grid for theta_x and theta_y
    [theta_x_mesh, theta_y_mesh] = meshgrid(theta_x_lin, theta_y_lin);

    % Initialize theta
    theta = zeros(n, 1);

    % Compute cost for each value of theta_x and theta_y
    for i = 1:N
        for j = 1:N
            theta(num_theta_x + 1) = theta_x_mesh(i, j);
            theta(num_theta_y + 1) = theta_y_mesh(i, j);

            J_mesh(i, j) = cost_function(theta, X, y);
        end
    end
end
