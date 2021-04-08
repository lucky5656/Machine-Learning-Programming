function checkNNGradients(lambda)
%CHECKNNGRADIENTS Creates a small neural network to check the
%backpropagation gradients
%   CHECKNNGRADIENTS(lambda) Creates a small neural network to check the
%   backpropagation gradients, it will output the analytical gradients
%   produced by your backprop code and the numerical gradients (computed
%   using computeNumericalGradient). These two gradient computations should
%   result in very similar values.
%

% 先判断是否第一次运行，正规化第一次去掉，即lambda=0
if ~exist('lambda', 'var') || isempty(lambda)
    lambda = 0;
end

%构造新的神经网络结构
input_layer_size = 3;
hidden_layer_size = 5;
num_labels = 3;
m = 5;

% We generate some 'random' test data
% 创造一些随机的训练集（随机初始化参数）
Theta1 = debugInitializeWeights(hidden_layer_size, input_layer_size);
Theta2 = debugInitializeWeights(num_labels, hidden_layer_size);
% Reusing debugInitializeWeights to generate X
% 重用函数debugInitializeWeights去生成 X 训练集
X  = debugInitializeWeights(m, input_layer_size - 1);% X 为5x2
y  = 1 + mod(1:m, num_labels)';%这里产生的y数组很显然是元素小于等于num_labels的正数的列向量


%接下来使用前向反馈算法-->计算J-->反向传播算法计算偏导数（这之前的都包含在nnCostFunction.m这个函数里面）-->数值梯度检验
% Unroll parameters
nn_params = [Theta1(:) ; Theta2(:)];

% Short hand for cost function
costFunc = @(p) nnCostFunction(p, input_layer_size, hidden_layer_size, num_labels, X, y, lambda);%它采用函数nnCostFunction，为它提供除p之外的所有参数，并将其转换为依赖于p的可调用对象。

[cost, grad] = costFunc(nn_params);%返回的是代价和使用反向传播算法计算的梯度，后面会用来与数值检验得到的梯度对比
numgrad = computeNumericalGradient(costFunc, nn_params);%传入的J和theta（向量）得到数值梯度检测得到的梯度

% Visually examine the two gradient computations.  The two columns
% you get should be very similar. 
disp([numgrad grad]);
fprintf(['The above two columns you get should be very similar.\n' ...
         '(Left-Your Numerical Gradient, Right-Analytical Gradient)\n\n']);

% Evaluate the norm of the difference between two solutions.  
% If you have a correct implementation, and assuming you used EPSILON = 0.0001 
% in computeNumericalGradient.m, then diff below should be less than 1e-9
diff = norm(numgrad-grad)/norm(numgrad+grad);%norm(A),A是一个向量，那么我们得到的结果就是A中的元素平方相加之后开根号(求范数)

fprintf(['If your backpropagation implementation is correct, then \n' ...
         'the relative difference will be small (less than 1e-9). \n' ...
         '\nRelative Difference: %g\n'], diff);

end
