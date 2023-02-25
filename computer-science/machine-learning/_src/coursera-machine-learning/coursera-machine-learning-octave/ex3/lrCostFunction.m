function [J, grad] = lrCostFunction(theta, X, y, lambda)
%LRCOSTFUNCTION Compute cost and gradient for logistic regression with 
%regularization
%   J = LRCOSTFUNCTION(theta, X, y, lambda) computes the cost of using
%   theta as the parameter for regularized logistic regression and the
%   gradient of the cost w.r.t. to the parameters. 

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
% Hint: The computation of the cost function and gradients can be
%       efficiently vectorized. For example, consider the computation
%
%           sigmoid(X * theta)
%
%       Each row of the resulting matrix will contain the value of the
%       prediction for that example. You can make use of this to vectorize
%       the cost function and gradient computations. 
%
% Hint: When computing the gradient of the regularized cost function, 
%       there're many possible vectorized solutions, but one solution
%       looks like:
%           grad = (unregularized gradient for logistic regression)
%           temp = theta; 
%           temp(1) = 0;   % because we don't add anything for j = 0  
%           grad = grad + YOUR_CODE_HERE (using the temp variable)
%

%fprintf('lrCostFunction starts from here\n')
  
z = X*theta;

%  fprintf('Caliculating J \n')
J = (1/m)*sum((-y.*log(sigmoid(z)))-(1-y).*log(1.-sigmoid(z))) + (lambda/(2*m))*sum(theta(2:end).^2);

%  fprintf('Caliculating grad1 \n')
grad(1) = (1/m) * sum(((sigmoid(z)).-y).*X(:,1));

%  fprintf('Caliculating grad2 temp \n')
% このtempの中のj番目のxとシグモイド関数の出力をかけるところでエラーが出る。後で直したい。

  temp = ((1/m).*sum( (sigmoid(z).-y).*X(:,2:size(theta)) ))' .+ ((lambda/m).*theta(2:end));

%  fprintf('Caliculate now grad2 \n')
grad(2:size(theta)) =temp;
%  fprintf('Caliculat done grad2 \n')

%fprintf('lrCostFunction ends by here\n')

% =============================================================

grad = grad(:);

end
