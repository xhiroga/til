function [C, sigma] = dataset3Params(X, y, Xval, yval)
%EX6PARAMS returns your choice of C and sigma for Part 3 of the exercise
%where you select the optimal (C, sigma) learning parameters to use for SVM
%with RBF kernel
%   [C, sigma] = EX6PARAMS(X, y, Xval, yval) returns your choice of C and 
%   sigma. You should complete this function to return the optimal C and 
%   sigma based on a cross-validation set.
%

% You need to return the following variables correctly.
C = 1;
sigma = 0.3;

% ====================== YOUR CODE HERE ======================
% Instructions: Fill in this function to return the optimal C and sigma
%               learning parameters found using the cross validation set.
%               You can use svmPredict to predict the labels on the cross
%               validation set. For example, 
%                   predictions = svmPredict(model, Xval);
%               will return the predictions on the cross validation set.
%
%  Note: You can compute the prediction error using 
%        mean(double(predictions ~= yval))
%

values = [0.01 0.03 0.1 0.3 1 3 10 30];

e = 0;
e_min = inf;

printf("reset error¥n");

C_min = 0;
sigma_min = 0;

for C_temp = values
  for sigma_temp = values
    
    
    model = svmTrain(X, y, C_temp, @(x1, x2)gaussianKernel(x1, x2, sigma_temp));
    e = mean(double(svmPredict(model, Xval) ~= yval));
    if e <= e_min
      e_min = e;
      printf("error_min is update to %f¥n", e_min); 
      C = C_temp;
      sigma = sigma_temp;
      printf("C and sigma are updated to %f %f¥n",C,sigma);
      

    end
 end   
    


% =========================================================================

end
