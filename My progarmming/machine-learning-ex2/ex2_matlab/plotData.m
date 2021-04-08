function plotData(X, y)
%PLOTDATA Plots the data points X and y into a new figure 
%   PLOTDATA(x,y) plots the data points with + for the positive examples
%   and o for the negative examples. X is assumed to be a Mx2 matrix.

% Create New Figure
figure; hold on;

% ====================== YOUR CODE HERE ======================
% Instructions: Plot the positive and negative examples on a
%               2D plot, using the option 'k+' for the positive
%               examples and 'ko' for the negative examples.
%

% Find Indices of Positive and Negative Examples %找出正反例子的索引
pos = find(y==1); neg = find(y == 0);  %pos 和neg 分别是 y元素=1和0的所在的位置序号组成的向量
% Plot Examples
plot(X(pos, 1), X(pos, 2), 'k+','LineWidth', 2,'MarkerSize', 7); %x中对应y等于1的第一列和第二列
plot(X(neg, 1), X(neg, 2), 'ko', 'MarkerFaceColor', 'y','MarkerSize', 7);%将数据点以圆形、黄色实心填充
% =========================================================================

hold off;

end
