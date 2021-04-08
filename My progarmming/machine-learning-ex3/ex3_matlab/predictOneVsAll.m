function p = predictOneVsAll(all_theta, X)
%PREDICT Predict the label for a trained one-vs-all classifier. The labels 
%are in the range 1..K, where K = size(all_theta, 1). 
%  p = PREDICTONEVSALL(all_theta, X) will return a vector of predictions
%  for each example in the matrix X. Note that X contains the examples in
%  rows. all_theta is a matrix where the i-th row is a trained logistic
%  regression theta vector for the i-th class. You should set p to a vector
%  of values from 1..K (e.g., p = [1; 3; 1; 2] predicts classes 1, 3, 1, 2
%  for 4 examples) 

m = size(X, 1);
num_labels = size(all_theta, 1);

% You need to return the following variables correctly 
p = zeros(size(X, 1), 1);

% Add ones to the X data matrix
X = [ones(m, 1) X];

% ====================== YOUR CODE HERE ======================
% Instructions: Complete the following code to make predictions using
%               your learned logistic regression parameters (one-vs-all).
%               You should set p to a vector of predictions (from 1 to
%               num_labels).
%
% Hint: This code can be done all vectorized using the max function.
%       In particular, the max function can also return the index of the 
%       max element, for more information see 'help max'. If your examples 
%       are in rows, then, you can use max(A, [], 2) to obtain the max 
%       for each row.
%       

%将 X 与 theta 相乘得到预测结果，得到一个 5000*10 的矩阵，每行对应一个分类结果（只有一个1，其余为0）。
%题目要求返回的矩阵 p 是一个 5000*1 的矩阵。每行是 1-K 的数字。
%使用 max 函数，得到两个返回值。第一个 x 是一个全1的向量，没用。第二个是 1 所在的 index，也就是对应的类别。
temp = X * all_theta';           %500X10
[x, p] = max(temp,[],2);        % 返回每行最大值的索引位置，也就是预测的数字
% =========================================================================

end
