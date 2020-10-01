function [J, grad] = costFunction(theta, X, y)
%COSTFUNCTION Compute cost and gradient for logistic regression
%   J = COSTFUNCTION(theta, X, y) computes the cost of using theta as the
%   parameter for logistic regression and the gradient of the cost
%   w.r.t. to the parameters.

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
J = 0;
grad = zeros(size(theta));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost of a particular choice of theta.
%               You should set J to the cost.
%               Compute the partial derivatives and set grad to the partial
%               derivatives of the cost w.r.t. each parameter in theta
%
% Note: grad should have the same dimensions as theta
%

z = X*theta;
plot (z);

J = (1/m) * sum( -(y.* log(sigmoid(z)))  .-((1.-y) .*log(1.-sigmoid(z))) );

grad1 = (1/m) * sum ( ( sigmoid(z) .- y) .* X(:,1));
grad2 = (1/m) * sum ( ( sigmoid(z) .- y) .* X(:,2));
grad3 = (1/m) * sum ( ( sigmoid(z) .- y) .* X(:,3));

grad = [grad1; grad2 ; grad3];





% =============================================================

end
