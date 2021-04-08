import numpy as np
import matplotlib.pyplot as plt
import scipy.io as sio
from sklearn import svm#导入svm的svc类（支持向量分类）

# =============== Part 1: Loading and Visualizing Data ================
# 数据可视化
def plotData(x, y):
    pos = np.where(y == 1)
    neg = np.where(y == 0)
    plt.plot(x[pos, 0], x[pos, 1], 'k+', lw=1, ms=7)
    plt.plot(x[neg, 0], x[neg, 1], 'ko', mfc='y', ms=7)

print('Loading and Visualizing Data ...')
datainfo = sio.loadmat('ex6data1.mat')
X = datainfo['X']#提取原始输入特征矩阵
Y = datainfo['y'][:, 0]#[:, 0]返回第一列,提取标签 并转换为1维数组
plotData(X, Y)
plt.show()
_ = input('Press [Enter] to continue.')

# ==================== Part 2: Training Linear SVM ====================
# 线性可视化
def visualBoundaryLinear(x, y, theta, b):
    xp = np.linspace(np.min(x[:, 0]), np.max(x[:, 0]), 100)
    yp = -(theta[0]*xp+b)/theta[1]
    plotData(x, y)
    plt.plot(xp, yp, '-b')

print('Training Linear SVM ...')
c = 1.0
clf = svm.SVC(C=c, kernel='linear')#clf = SVC()创建分类器对象,线性核函数kernel=‘linear’
clf.fit(X, Y)#用训练数据拟合分类器模型
theta = clf.coef_.flatten()#取出权重矩阵,有两个值，分别是w0和w1，klearn中对超平面的表示并不是y=kx+b这样的，而是x作为第一个特征x0，y作为另一个特征x1，表示为：w0x0+w1x1+bias=0
b = clf.intercept_  #取出截距,使用clf.intercept_即可获取bias的值,画超平面y=-(w0/w1)x-bias/w1=-(w0*x+bias)/w1
visualBoundaryLinear(X, Y, theta, b)
plt.show()
_ = input('Press [Enter] to continue.')

# =============== Part 3: Implementing Gaussian Kernel ===============
# 高斯核
def gaussianKernel(x1, x2, sigma):
    sim = np.exp(-(x1-x2).dot(x1-x2)/(2*sigma**2))
    return sim

print('Evaluating the Gaussian Kernel ...')
x1 = np.array([1, 2, 1])
x2 = np.array([0, 4, -1])
sigma = 2
sim = gaussianKernel(x1, x2, sigma)
print('Gaussian Kernel between x1 = [1; 2; 1], x2 = [0; 4; -1], sigma = 0.5 :\
      \t%f\n(this value should be about 0.324652)' % sim)
_ = input('Press [Enter] to continue.')

# =============== Part 4: Visualizing Dataset 2 ================
print('Loading and Visualizing Data ...')
datainfo = sio.loadmat('ex6data2.mat')
X = datainfo['X']
Y = datainfo['y'][:, 0]
plotData(X, Y)
plt.show()
_ = input('Press [Enter] to continue.')

# ========== Part 5: Training SVM with RBF Kernel (Dataset 2) ==========
# 绘制边界
def visualBoundary(x, y, model):
    plotData(x, y)
    x1plot = np.linspace(np.min(x[:, 0]), np.max(x[:, 0]), 100)
    x2plot = np.linspace(np.min(x[:, 1]), np.max(x[:, 1]), 100)
    x1, x2 = np.meshgrid(x1plot, x2plot)
    vals = np.zeros(np.shape(x1))
    for i in range(np.size(x1, 1)):
        this_x = np.vstack((x1[:, i], x2[:, i])).T
        vals[:, i] = model.predict(this_x) 
    plt.contour(x1, x2, vals)#等高线图 将0/1分界线（决策边界）画出来5);



# 这里需要注意的是gamma并不是原本中的sigma
print('Training SVM with RBF Kernel (this may take 1 to 2 minutes) ...')
c = 1; sigma = 0.1
gam = 1/(2*sigma**2)#gam=1/2sigma^2
clf = svm.SVC(kernel='rbf', C=1.0, gamma=gam)#RBF函数：exp(-gamma|u-v|^2)
clf.fit(X, Y)
visualBoundary(X, Y, clf)

plt.show()
_ = input('Press [Enter] to continue.')

# =============== Part 6: Visualizing Dataset 3 ================
datainfo = sio.loadmat('ex6data3.mat')
X = datainfo['X']
Y = datainfo['y'][:, 0]
Xval = datainfo['Xval']
Yval = datainfo['yval'][:, 0]
plotData(X, Y)
plt.show()
_ = input('Press [Enter] to continue.')

# ========== Part 7: Training SVM with RBF Kernel (Dataset 3) ==========
# 参数选择
def dataset3Params(x, y, xval, yval):
    c = 1; sigma = 0.3
    err_best = np.size(yval, 0)
    c_choice = [0.3, 1.0]
    sigma_choice = [0.1, 0.3]
    for i in range(len(c_choice)):
        for j in range(len(sigma_choice)):
            clf = svm.SVC(C=c_choice[i], gamma=1/(2*sigma_choice[j]**2))
            clf.fit(x, y)
            pred = clf.predict(xval) #用训练好的分类器去预测
            err = np.sum(pred != yval)/np.size(yval, 0)
            if err_best>err:
                err_best = err
                c = c_choice[i]
                sigma = sigma_choice[j]
    return c, sigma

c, sigma = dataset3Params(X, Y, Xval, Yval)
clf = svm.SVC(C=c, gamma=1/(2*sigma**2))
clf = clf.fit(X, Y)
visualBoundary(X, Y, clf)
plt.show()
