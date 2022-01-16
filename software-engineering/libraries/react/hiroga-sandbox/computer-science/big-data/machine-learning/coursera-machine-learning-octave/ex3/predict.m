function p = predict(Theta1, Theta2, X)
%PREDICT Predict the label of an input given a trained neural network
%   p = PREDICT(Theta1, Theta2, X) outputs the predicted label of X given the
%   trained weights of a neural network (Theta1, Theta2)

% Useful values
m = size(X, 1);
num_labels = size(Theta2, 1);

% You need to return the following variables correctly 
p = zeros(size(X, 1), 1);

% ====================== YOUR CODE HERE ======================
% Instructions: Complete the following code to make predictions using
%               your learned neural network. You should set p to a 
%               vector containing labels between 1 to num_labels.
%
% Hint: The max function might come in useful. In particular, the max
%       function can also return the index of the max element, for more
%       information see 'help max'. If your examples are in rows, then, you
%       can use max(A, [], 2) to obtain the max for each row.
%

fprintf('show size(x) *m = size(X, 1) \n')
size(X )

fprintf('show num_labels \n')
num_labels

fprintf('show size(Theta1 and Theta2) \n')
size(Theta1)
size(Theta2)

%initialize activator1 & 2
%actually, I want to merge a1 and a2...
a1 = [m, size(Theta1, 1)];
a2 = [m, size(Theta2, 1)];


%theta 0th... have to add ones to X
%I do not know why this add ones to X...
X = [ones(m, 1) X];

size(X)

z1 = X * Theta1';
fprintf('show size(z1) \n')
size(z1)
a1 = sigmoid(z1);

a1 = [ones(size(a1),1) a1];

z2 = a1 * Theta2';
a2 = sigmoid(z2);

[valuepoint, p] = max(a2, [], 2);


%end point cause bug? process stop


% =========================================================================


end
