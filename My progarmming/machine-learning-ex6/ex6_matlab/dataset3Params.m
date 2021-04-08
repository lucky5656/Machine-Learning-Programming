function [C, sigma] = dataset3Params(X, y, Xval, yval)
%DATASET3PARAMS returns your choice of C and sigma for Part 3 of the exercise
%where you select the optimal (C, sigma) learning parameters to use for SVM
%with RBF kernel
%   [C, sigma] = DATASET3PARAMS(X, y, Xval, yval) returns your choice of C and 
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

C_vec = [0.01 0.03 0.1 0.3 1 3 10 30]';
sigma_vec = [0.01 0.03 0.1 0.3 1 3 10 30]';
error_val = zeros(length(C_vec),length(sigma_vec));
error_train = zeros(length(C_vec),length(sigma_vec));
for i = 1:length(C_vec)
    for j = 1:length(sigma_vec)
        model= svmTrain(X, y, C_vec(i), @(x1, x2) gaussianKernel(x1, x2, sigma_vec(j)));
        predictions = svmPredict(model, Xval); %svmPredict()使用经过训练的支持向量机模型返回预测向量
        error_val(i,j) = mean(double(predictions ~= yval));
    end
end

% figure
% error_val
% surf(C_vec,sigma_vec,error_val)   % 画出三维图找最低点

[minval,ind] = min(error_val(:));   % 0.03  %[minval,ind] = min():minval表示最小值，ind表示最小值的位置 
[I,J] = ind2sub([size(error_val,1) size(error_val,2)],ind);%ind2sub()把数组或者矩阵的线性索引转化为相应的下标,返回i,j,也就是返回的行标和列标
C = C_vec(I)          %   1
sigma = sigma_vec(J)  %   0.100

% [I,J]=find(error_val ==  min(error_val(:)) );    % 另一种方式找最小元素位子
% C = C_vec(I)          % 1
% sigma = sigma_vec(J)  % 0.100

% =========================================================================

end
