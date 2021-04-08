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

a1 = [ones(m, 1) X];    %5000x401
a2 = sigmoid(a1 * Theta1');  %5000x401乘以401x25得到5000x25。即把401个feature映射到25

a2 = [ones(m, 1) a2];    %5000x26
a3 = sigmoid(a2 * Theta2');  %5000x26乘以26x10得到5000x10。即把26个feature映射到10

[x,p] = max(a3,[],2);%和上面逻辑回归多分类一样，最后使用 max 函数获得分类结果
                             %[C, I] = max(a, [], dim)表示 a 是个二维矩阵，dim = 2 表示比较的是行，返回 size(a, 1) 行，每行元素是 a 该行最大的元素；dim = 1 表示比较的是列
% =========================================================================

end
