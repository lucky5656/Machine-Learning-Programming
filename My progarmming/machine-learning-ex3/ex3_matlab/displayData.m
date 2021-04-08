function [h, display_array] = displayData(X, example_width)
%DISPLAYDATA Display 2D data in a nice grid
%   [h, display_array] = DISPLAYDATA(X, example_width) displays 2D data
%   stored in X in a nice grid. It returns the figure handle h and the 
%   displayed array if requested.

% Set example_width automatically if not passed in
% ���'example_width'�����ڻ���Ϊ�վ���ʱ������example_width=X����������ƽ����
if ~exist('example_width', 'var') || isempty(example_width) 
	example_width = round(sqrt(size(X, 2))); %�����������ͼƬ�Ŀ��
end

% Gray Image
colormap(gray); %��ͼƬ����Ϊ��ɫϵ 

% Compute rows, cols
%m=100, n=400 , example_height=20
[m n] = size(X); 
example_height = (n / example_width); %���ͼƬ�ĸ߶�

% Compute number of items to display
% �����ÿ��ÿ��չʾ���ٸ�����ͼƬ
%display_rows=10, display_cols=10��չʾ����ͼ����10�У�ÿ��10��
display_rows = floor(sqrt(m)); %floor(x)��ʾ������x�����������,��һ��С������ȡ��
display_cols = ceil(m / display_rows); %ceil()��һ��С������ȡ��

% Between images padding
% �����˱߽磬ͼƬ�к�ɫ�߿���Ǳ߽硣���Խ�pad����Ϊ0���Ƚ�ǰ����
pad = 1; %ͼƬ֮����

% Setup blank display
% display_array ��һ�� ��1 + 10 * �� 20 + 1����1 + 10 * ��20 + 1�����ľ���
%����Ҫչʾ��ͼƬ���ش�С�������أ�����ͼƬ֮����1���ؼ��
display_array = - ones(pad + display_rows * (example_height + pad), pad + display_cols * (example_width + pad));

% Copy each example into a patch on the display array
%�����ص�����ȥ
curr_ex = 1;
for j = 1:display_rows
	for i = 1:display_cols
		if curr_ex > m,  % curr_ex������100
			break; 
		end
		% Copy the patch
		
		% Get the max value of the patch
		max_val = max(abs(X(curr_ex, :))); %abs() �����������ֵľ���ֵ % ȡ X ÿһ�е����ֵ�����������а�������С�����ҳ�Ϊ��׼����
        % ��ԭ�ȵ� X ����ÿ�б�׼���Ժ� 20 * 20 ����ʽ �� display_array �еĸ�ֵ
        %display_array[1+(j-1)*21+(1:20) , 1+(i-1)*21+(1:20)] = ... �ǽ�53���еı任������������и�ֵ����display_array
		display_array(pad + (j - 1) * (example_height + pad) + (1:example_height), ...
		              pad + (i - 1) * (example_width + pad) + (1:example_width)) = ...
						reshape(X(curr_ex, :), example_height, example_width) / max_val;
                        %reshape(X(curr_ex, : ) , 20, 20)�ǽ�X�ĵ�curr_ex�У���400�У�����Ϊ20x20����
                        %Ȼ���������������һ���е����ֵmax_val������ǰ�潲���ı�׼����ʹ����������ֵ��[-1,1]
		curr_ex = curr_ex + 1;
	end
	if curr_ex > m, 
		break; 
	end
end

% Display Image
h = imagesc(display_array, [-1 1]); %�����ص㻯ΪͼƬ

% Do not show axis
axis image off %����ʾ������

drawnow; %ˢ����Ļ

end
