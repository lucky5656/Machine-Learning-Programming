function [h, display_array] = displayData(X, example_width)
%DISPLAYDATA Display 2D data in a nice grid
%   [h, display_array] = DISPLAYDATA(X, example_width) displays 2D data
%   stored in X in a nice grid. It returns the figure handle h and the 
%   displayed array if requested.

% Set example_width automatically if not passed in
% 如果'example_width'不存在或者为空矩阵时，定义example_width=X列数的算数平方根
if ~exist('example_width', 'var') || isempty(example_width) 
	example_width = round(sqrt(size(X, 2))); %四舍五入求出图片的宽度
end

% Gray Image
colormap(gray); %将图片定义为灰色系 

% Compute rows, cols
%m=100, n=400 , example_height=20
[m n] = size(X); 
example_height = (n / example_width); %求出图片的高度

% Compute number of items to display
% 计算出每行每列展示多少个数字图片
%display_rows=10, display_cols=10，展示数字图像是10行，每行10个
display_rows = floor(sqrt(m)); %floor(x)表示不大于x的最大整型数,把一个小数向下取整
display_cols = ceil(m / display_rows); %ceil()把一个小数向上取整

% Between images padding
% 定义了边界，图片中黑色边框就是边界。可以将pad设置为0，比较前后差别
pad = 1; %图片之间间隔

% Setup blank display
% display_array 是一个 （1 + 10 * （ 20 + 1），1 + 10 * （20 + 1））的矩阵
%创建要展示的图片像素大小，空像素，数字图片之间有1像素间隔
display_array = - ones(pad + display_rows * (example_height + pad), pad + display_cols * (example_width + pad));

% Copy each example into a patch on the display array
%将像素点填充进去
curr_ex = 1;
for j = 1:display_rows
	for i = 1:display_cols
		if curr_ex > m,  % curr_ex不超过100
			break; 
		end
		% Copy the patch
		
		% Get the max value of the patch
		max_val = max(abs(X(curr_ex, :))); %abs() 函数返回数字的绝对值 % 取 X 每一行的最大值，方便后面进行按比例缩小（暂且称为标准化）
        % 把原先的 X 里面每行标准化以后按 20 * 20 的形式 给 display_array 中的赋值
        %display_array[1+(j-1)*21+(1:20) , 1+(i-1)*21+(1:20)] = ... 是将53行中的变换后矩阵逐行逐列赋值给了display_array
		display_array(pad + (j - 1) * (example_height + pad) + (1:example_height), ...
		              pad + (i - 1) * (example_width + pad) + (1:example_width)) = ...
						reshape(X(curr_ex, :), example_height, example_width) / max_val;
                        %reshape(X(curr_ex, : ) , 20, 20)是将X的第curr_ex行（共400列）变形为20x20矩阵
                        %然后整个矩阵除以这一行中的最大值max_val，类似前面讲过的标准化，使得这个矩阵的值在[-1,1]
		curr_ex = curr_ex + 1;
	end
	if curr_ex > m, 
		break; 
	end
end

% Display Image
h = imagesc(display_array, [-1 1]); %将像素点化为图片

% Do not show axis
axis image off %不显示坐标轴

drawnow; %刷新屏幕

end
