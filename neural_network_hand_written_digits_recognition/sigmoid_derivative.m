%% sigmoid_derivatie : Funcion which compute the derivative of sigmoid function
function derivative = sigmoid_derivative(z)
	sig_z = sigmoid(z);
	derivative = sig_z .* (1 - sig_z);
end
