function [J, grad] = linearRegCostFunction(X, y, theta, lambda)
%LINEARREGCOSTFUNCTION Compute cost and gradient for regularized linear 
%regression with multiple variables
%   [J, grad] = LINEARREGCOSTFUNCTION(X, y, theta, lambda) computes the 
%   cost of using theta as the parameter for linear regression to fit the 
%   data points in X and y. Returns the cost in J and the gradient in grad

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
J = 0;
grad = zeros(size(theta));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost and gradient of regularized linear 
%               regression for a particular choice of theta.
%
%               You should set J to the cost and grad to the gradient.
%

%%calculate cost J
%
%% make sigmoid function by myself...
%
%%this quiz not need sigmoid?
%%z = X * theta;
%%sig = 1./(1.+e.^(-z));
%
%J = (1/(2*m))*sum((X*theta .-y).^2) + lambda/(2*m)* (sum(theta.^2));
%
%%calculate grad
%
%grad0 = zeros(0);
%grad1 = zeros(size(theta([2,:],1)));
%
%%in matrix, no 0
%
%grad0 = (1/m)*sum((X*theta .-y) .*X(:,1),1);
%grad1 = (1/m).*sum((X*theta .-y) .*X(:,[2,:]),1).+ (lambda/m).*theta([2,:],1)';
%
%grad(1,1) = grad0;
%grad([2,:],1) = grad1';

%fprintf('size X');
%size(X)
%fprintf('size y');
%size(y)
%fprintf('size theta');
%size(theta)

%size X
%   12    2
%size y
%   12    1
%size theta
%   2   1

h = X * theta;
J = sum((h - y).^2)/(2*m) + lambda*sum([theta(2:end,1)].^2)/(2*m);


i = 1;
grad(i,1) = sum((h - y).*X(:,i))/m;

n = size(X,2);

for i = 2:n
  grad(i,1) = sum((h - y).*X(:,i))/m + (lambda/m)*theta(i,1);
end

















% =========================================================================

grad = grad(:);

end
