function [J grad] = nnCostFunction(nn_params, ...
                                   input_layer_size, ...
                                   hidden_layer_size, ...
                                   num_labels, ...
                                   X, y, lambda)
%NNCOSTFUNCTION Implements the neural network cost function for a two layer
%neural network which performs classification
%   [J grad] = NNCOSTFUNCTON(nn_params, hidden_layer_size, num_labels, ...
%   X, y, lambda) computes the cost and gradient of the neural network. The
%   parameters for the neural network are "unrolled" into the vector
%   nn_params and need to be converted back into the weight matrices. 
% 
%   The returned parameter grad should be a "unrolled" vector of the
%   partial derivatives of the neural network.
%

% Reshape nn_params back into the parameters Theta1 and Theta2, the weight matrices
% for our 2 layer neural network
Theta1 = reshape(nn_params(1:hidden_layer_size * (input_layer_size + 1)), ...
                 hidden_layer_size, (input_layer_size + 1));

Theta2 = reshape(nn_params((1 + (hidden_layer_size * (input_layer_size + 1))):end), ...
                 num_labels, (hidden_layer_size + 1));

% Setup some useful variables
m = size(X, 1);
         
% You need to return the following variables correctly 
J = 0;
Theta1_grad = zeros(size(Theta1));
Theta2_grad = zeros(size(Theta2));

% ====================== YOUR CODE HERE ======================
% Instructions: You should complete the code by working through the
%               following parts.
%
% Part 1: Feedforward the neural network and return the cost in the
%         variable J. After implementing Part 1, you can verify that your
%         cost function computation is correct by verifying the cost
%         computed in ex4.m
%
% Part 2: Implement the backpropagation algorithm to compute the gradients
%         Theta1_grad and Theta2_grad. You should return the partial derivatives of
%         the cost function with respect to Theta1 and Theta2 in Theta1_grad and
%         Theta2_grad, respectively. After implementing Part 2, you can check
%         that your implementation is correct by running checkNNGradients
%
%         Note: The vector y passed into the function is a vector of labels
%               containing values from 1..K. You need to map this vector into a 
%               binary vector of 1's and 0's to be used with the neural network
%               cost function.
%
%         Hint: We recommend implementing backpropagation using a for-loop
%               over the training examples if you are implementing it for the 
%               first time.
%
% Part 3: Implement regularization with the cost function and gradients.
%
%         Hint: You can implement this around the code for
%               backpropagation. That is, you can compute the gradients for
%               the regularization separately and then add them to Theta1_grad
%               and Theta2_grad from Part 2.
%

% Personal note :
% m = nb of training examples
% K = nb of labels
% L = nb of layers
% I = nb of unit on layer
% J = nb of unit on next layer
% J- = nb of unit on previous layer

% Change matrix y (size : m * 1)
Y = zeros(size(y,1),num_labels);

for i = 1:size(y,1)
Y(i, y(i)) = 1; % m * K 
end

% Feedforward => compute h_theta
a1 = [ones(size(X,1),1) X];
z2 = a1 * Theta1'; % m * J = m * (I + 1) * (J * (I + 1))' ... I is a ref to layer 1 here
a2 = sigmoid(z2); % m * J

a2 = [ones(size(X,1),1) a2];
z3 = a2 * Theta2';
a3 = sigmoid(z3);

% Compute cost 
J = - 1 / m * sum(sum(Y .* log(a3) + (1 - Y) .* log(1 - a3), 2)) + lambda / 2 / m * (sum(sum(Theta1(:, 2:end) .^ 2, 2)) + sum(sum(Theta2(:, 2:end) .^ 2, 2)))

% Deltas initialization
Delta_1 = zeros(size(Theta1));
Delta_2 = zeros(size(Theta2));

% back propagation
for t = 1:m
% initialization
delta3 = a3(t,:)' - Y(t,:)';

% small delta computaion
tmp_2 = [ones(m, 1) z2];
delta2 = Theta2' * delta3 .* sigmoidGradient(tmp_2(t,:))'; % I + 1 = (J * (I + 1))' * (J * 1) .* (1 * (I + 1))'
delta2 = delta2(2:end); % first unit is biaised

% Delta sums on all training sets
Delta_2 = Delta_2 + delta3 * a2(t,:); % J * (I + 1) = ... + (J * 1) * (1 * (I + 1))
Delta_1 = Delta_1 + delta2 * a1(t,:);
end

%regularization of the theta gradient
Theta1_grad(:, 1) = Delta_1(:, 1) / m;
Theta2_grad(:, 1) = Delta_2(:, 1) / m;

Theta1_grad(:, 2:end) = Delta_1(:, 2:end) / m + lambda / m * Theta1(:, 2:end);
Theta2_grad(:, 2:end) = Delta_2(:, 2:end) / m + lambda / m * Theta2(:, 2:end);



% -------------------------------------------------------------

% =========================================================================

% Unroll gradients
grad = [Theta1_grad(:) ; Theta2_grad(:)];


end
