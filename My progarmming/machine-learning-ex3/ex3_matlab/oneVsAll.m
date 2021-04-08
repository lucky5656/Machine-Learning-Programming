function [all_theta] = oneVsAll(X, y, num_labels, lambda)
%ONEVSALL trains multiple logistic regression classifiers and returns all
%the classifiers in a matrix all_theta, where the i-th row of all_theta 
%corresponds to the classifier for label i
%   [all_theta] = ONEVSALL(X, y, num_labels, lambda) trains num_labels
%   logistic regression classifiers and returns each of these classifiers
%   in a matrix all_theta, where the i-th row of all_theta corresponds 
%   to the classifier for label i

% Some useful variables
m = size(X, 1);
n = size(X, 2);

% You need to return the following variables correctly 
all_theta = zeros(num_labels, n + 1);

% Add ones to the X data matrix
X = [ones(m, 1) X];

% ====================== YOUR CODE HERE ======================
% Instructions: You should complete the following code to train num_labels
%               logistic regression classifiers with regularization
%               parameter lambda. 
%
% Hint: theta(:) will return a column vector.
%
% Hint: You can use y == c to obtain a vector of 1's and 0's that tell you
%       whether the ground truth is true/false for this class.
%
% Note: For this assignment, we recommend using fmincg to optimize the cost
%       function. It is okay to use a for-loop (for c = 1:num_labels) to
%       loop over the different classes.
%
%       fmincg works similarly to fminunc, but is more efficient when we
%       are dealing with large number of parameters.
%
% Example Code for fmincg:
%
%     % Set Initial theta
%     initial_theta = zeros(n + 1, 1);
%     
%     % Set options for fminunc
%     options = optimset('GradObj', 'on', 'MaxIter', 50);
% 
%     % Run fmincg to obtain the optimal theta
%     % This function will return theta and the cost 
%     [theta] = ...
%         fmincg (@(t)(lrCostFunction(t, X, (y == c), lambda)), ...
%                 initial_theta, options);
%

%逻辑回归分类的目标是区分两种类型。写好代价函数之后，调用一次 fminunc 函数，得到 theta、就可以画出 boundary。
%但是多分类中需要训练 K 个分类器，所以多了一个 oneVsAll.m。其实就是循环 K 次，得到一个 theta矩阵（row 为 K，column为特征值个数）。
%多分类使用的 Octave 函数是 fmincg 不是 fminunc，fmincg更适合参数多的情况。
%注意这里用到了 y == c，这个操作将 y 中每个数据和 c 进行比较。得到的矩阵中(这里是向量)相同的为1，不同的为0。 
%表示第j个训练实例是属于K类(yj = 1)，还是属于另一个类(yj = 0)
initial_theta = zeros(n+1 , 1);
options = optimset('Gradobj', 'on', 'MaxIter', 50);
for c = 1: num_labels
  [all_theta(c,:)] = fmincg(@(t)lrCostFunction(t, X, (y==c), lambda) ,initial_theta, options);%使选项中指定的优化选项最小化。
end
%当我们调用这个fminunc函数时，它会自动的从众多高级优化算法中挑选一个来使用(你也可以把它当做一个可以自动选择合适的学习速率aa的梯度下降算法)。
% =========================================================================


end
