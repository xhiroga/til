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

%Part1


%fprintf ('Theta1 column = ');
%size(Theta1,1);
%
%fprintf ('Theta1 row = ');
%size(Theta1,2);
%
%fprintf ('Theta2 column = ');
%size(Theta2,1);
%
%fprintf ('Theta2 row = ');
%size(Theta2,2);
%
%fprintf ('X column = ')
%size(X,1)
%fprintf ('X row = ')
%size(X,2)

%fprintf ('y column = ')
%size(y,1)
%fprintf ('y row = ')
%size(y,2)


%processing X for calculation

Xtemp = [ones(size(X,1),1) X];
%fprintf ('Xtemp = ')
%size(Xtemp)

%calculation Z as layer 1st
Z1 = Xtemp * Theta1';
%fprintf ('Z1 = ')
%size(Z1)

%calculation A of layer 1st
A2 = sigmoid(Z1);
%fprintf ('A2 = ')
%size(A2)

%expand A2
A2temp = [ones(size(A2,1),1) A2];
%fprintf ('A2temp = ')
%size(A2temp)

%calculation A3 from A2
Z2 = A2temp * Theta2';
A3 = sigmoid(Z2);
%fprintf ('A3 = ')
%size(A3)

ytemp = zeros(size(y,1),num_labels);

for i=1:1:size(y,1)
  ytemp(i,y(i,1)) = 1;
end

temp = sum((-ytemp).*log(A3) - (1-ytemp).*log(1-A3),2);

Theta1sq = Theta1(:,2:end) .^2;
Theta2sq = Theta2(:,2:end) .^2;
reg = (lambda/(2*m)) *(sum(Theta1sq(:)) + sum(Theta2sq(:)) );

J = (1/m)*sum(temp,1) + reg;

%compute grad

sigma3 = A3-ytemp;

sigma2 = (sigma3*Theta2).*sigmoidGradient([ones(size(Z1,1),1) Z1]);

sigma2 = sigma2(:,2:end);

delta1 = sigma2' * Xtemp;
delta2 = sigma3' * [ones(size(A2,1),1) A2];

p1 = (lambda/m).*[zeros(size(Theta1, 1), 1) Theta1(:, 2:end)];
p2 = (lambda/m).*[zeros(size(Theta2, 1), 1) Theta2(:, 2:end)];

Theta1_grad = delta1./m +p1;
Theta2_grad = delta2./m +p2;














% -------------------------------------------------------------

% =========================================================================

% Unroll gradients
grad = [Theta1_grad(:) ; Theta2_grad(:)];


end
