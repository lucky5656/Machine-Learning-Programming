# -*- coding:utf-8 -*-
from matplotlib.font_manager import FontProperties
import matplotlib.pyplot as plt
import numpy as np


def loadDataSet(fileName):
    """
	函数说明:加载数据
	Parameters:
		fileName - 文件名
	Returns:
		xArr - x数据集
		yArr - y数据集
	Website:
		http://www.cuijiahua.com/
	Modify:
		2017-11-12
	"""
    numFeat = len(open(fileName).readline().split('\t')) - 1
    xArr = []
    yArr = []
    fr = open(fileName)
    for line in fr.readlines():
        lineArr = []
        curLine = line.strip().split('\t')
        for i in range(numFeat):
            lineArr.append(float(curLine[i]))
        xArr.append(lineArr)
        yArr.append(float(curLine[-1]))
    return xArr, yArr


def standRegres(xArr, yArr):
    """
	函数说明:计算回归系数w
	Parameters:
		xArr - x数据集
		yArr - y数据集
	Returns:
		ws - 回归系数
	Website:
		http://www.cuijiahua.com/
	Modify:
		2017-11-12
	"""
    xMat = np.mat(xArr)
    yMat = np.mat(yArr).T
    xTx = xMat.T * xMat  # 根据文中推导的公示计算回归系数
    if np.linalg.det(xTx) == 0.0:
        print("矩阵为奇异矩阵,不能求逆")
        return
    ws = xTx.I * (xMat.T * yMat)
    print("回归系数ws:", ws)
    return ws


def plotDataSet():
    """
	函数说明:绘制数据集
	Parameters:
		无
	Returns:
		无
	Website:
		http://www.cuijiahua.com/
	Modify:
		2017-11-12
	"""
    xArr, yArr = loadDataSet('ex0.txt')  # 加载数据集
    n = len(xArr)  # 数据个数
    xcord = []
    ycord = []  # 样本点
    for i in range(n):
        xcord.append(xArr[i][1])
        ycord.append(yArr[i])  # 样本点
    fig = plt.figure()
    ax = fig.add_subplot(111)  # 添加subplot
    ax.scatter(xcord, ycord, s=20, c='blue', alpha=.5)  # 绘制样本点
    plt.title('DataSet')  # 绘制title
    plt.xlabel('X')
    plt.show()


def plotRegression():
    """
	函数说明:绘制回归曲线和数据点
	Parameters:
		无
	Returns:
		无
	Website:
		http://www.cuijiahua.com/
	Modify:
		2017-11-12
	"""
    xArr, yArr = loadDataSet('ex0.txt')  # 加载数据集
    ws = standRegres(xArr, yArr)  # 计算回归系数
    xMat = np.mat(xArr)  # 创建xMat矩阵
    yMat = np.mat(yArr)  # 创建yMat矩阵
    xCopy = xMat.copy()  # 深拷贝xMat矩阵
    xCopy.sort(0)  # 排序
    yHat = xCopy * ws  # 计算对应的y值
    fig = plt.figure()
    ax = fig.add_subplot(111)  # 添加subplot
    ax.plot(xCopy[:, 1], yHat, c='red')  # 绘制回归曲线
    ax.scatter(xMat[:, 1].flatten().A[0], yMat.flatten().A[0], s=20, c='blue', alpha=.5)  # 绘制样本点
    plt.title('DataSet')  # 绘制title
    plt.xlabel('X')
    plt.show()


if __name__ == '__main__':
    plotDataSet()  # 可视化数据,可以看到数据的分布情况

    plotRegression()  # 用线性回归找到最佳拟合直线

    xArr, yArr = loadDataSet('ex0.txt')  # 加载数据集
    ws = standRegres(xArr, yArr)  # 计算回归系数
    xMat = np.mat(xArr)  # 创建xMat矩阵
    yMat = np.mat(yArr)  # 创建yMat矩阵
    yHat = xMat * ws
    print(np.corrcoef(yHat.T, yMat))  # 判断拟合曲线的拟合效果,使用corrcoef方法来比较预测值和真实值的相关性
