"""
函数说明:梯度上升算法测试函数

求函数f(x) = -x^2 + 4x的极大值

Parameters:
    无
Returns:
    无
Author:
    Jack Cui
Blog:
    http://blog.csdn.net/c406495762
Zhihu:
    https://www.zhihu.com/people/Jack--Cui/
Modify:
    2017-08-28
"""


def Gradient_Ascent_test():
    def f_prime(x_old):  # f(x)的导数
        return -2 * x_old + 4

    x_old = -1  # 初始值，给一个小于x_new的值
    x_new = 0  # 梯度上升算法初始值，即从(0,0)开始
    alpha = 0.01  # 步长，也就是学习速率，控制更新的幅度
    presision = 0.00000001  # 精度，也就是更新阈值
    while abs(x_new - x_old) > presision:
        x_old = x_new
        x_new = x_old + alpha * f_prime(x_old)  # 上面提到的公式
    print(x_new)  # 打印最终求解的极值近似值

if __name__ == '__main__':
    Gradient_Ascent_test()
