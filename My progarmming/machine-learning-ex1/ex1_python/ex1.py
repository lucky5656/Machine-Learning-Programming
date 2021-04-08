import numpy as np  #numpy模块：支持大量的维度数组与矩阵运算
import matplotlib.pylab as plt #matplotlib.pylab模块：绘图
from mpl_toolkits.mplot3d import Axes3D #mpl_toolkits.mplot3d绘制 3D 图像

# ==================== Part 1: Basic Function ====================
print('Running warmUpExercise...')
print('5x5 Identity Matrix: ')
A = np.eye((5))
print(A)
_ = input('Press [Enter] to continue.')

# ======================= Part 2: Plotting =======================
# 绘制散点图
def plotData(x, y):
    plt.plot(x, y, 'rx', ms=10)
    plt.xlabel('Population of City in 10,000')
    plt.ylabel('Profit in $10,000')
    plt.show()

print('Plotting Data...')
data = np.loadtxt('ex1data1.txt', delimiter=',')
X = data[:, 0]; Y = data[:, 1]
m = np.size(Y, 0)
plotData(X, Y)
_ = input('Press [Enter] to continue.')

# =================== Part 3: Gradient descent ===================
# 计算损失函数值
def computeCost(x, y, theta):
    ly = np.size(y, 0)
    cost = (x.dot(theta)-y).dot(x.dot(theta)-y)/(2*ly) #dot()函数:向量点积和矩阵乘法 x.dot(y) 等价于 np.dot(x,y)-x是m*n 矩阵y是n*m矩阵，则x.dot(y) 得到m*m矩阵
    return cost

# 迭代计算theta
def gradientDescent(x, y, theta, alpha, num_iters):
    m = np.size(y, 0)
    j_history = np.zeros((num_iters,))

    for i in range(num_iters):
        deltaJ = x.T.dot(x.dot(theta)-y)/m
        theta = theta-alpha*deltaJ
        j_history[i] = computeCost(x, y, theta)
    return theta, j_history

print('Running Gradient Descent ...')
X = np.vstack((np.ones((m,)), X)).T #vstack()在列上合并
theta = np.zeros((2,))             # 初始化参数 #若A = np.zeros(5)结果 [0. 0. 0. 0. 0.]；B = np.zeros((5,), dtype=np.int)结果 [0 0 0 0 0]

iterations = 3000
alpha = 0.01

J = computeCost(X, Y, theta)
print(J)

theta, j_history = gradientDescent(X, Y, theta, alpha, iterations)
print('Theta found by gradient descent: ', theta)

plt.plot(X[:, 1], Y, 'rx', ms=10, label='Training data')
plt.plot(X[:, 1], X.dot(theta), '-', label='Linear regression')
plt.xlabel('Population of City in 10,000')
plt.ylabel('Profit in $10,000')
plt.legend(loc='upper right')
plt.show()

# Predict values for population sizes of 35,000 and 70,000
predict1 = np.array([1, 3.5]).dot(theta)
print('For population = 35,000, we predict a profit of ', predict1*10000)
predict2 = np.array([1, 7.0]).dot(theta)
print('For population = 70,000, we predict a profit of ', predict2*10000)
_ = input('Press [Enter] to continue.')


# ============= Part 4: Visualizing J(theta_0, theta_1) =============
print('Visualizing J(theta_0, theta_1) ...')
theta0_vals = np.linspace(-10, 10, 100)
theta1_vals = np.linspace(-1, 4, 100)

J_vals = np.zeros((np.size(theta0_vals, 0), np.size(theta1_vals, 0)))

for i in range(np.size(theta0_vals, 0)):
    for j in range(np.size(theta1_vals, 0)):
        t = np.array([theta0_vals[i], theta1_vals[j]])
        J_vals[i, j] = computeCost(X, Y, t)

# 绘制三维图像
theta0_vals, theta1_vals = np.meshgrid(theta0_vals, theta1_vals)
fig = plt.figure()
#ax = Axes3D(fig)
ax = fig.gca(projection='3d')
ax.plot_surface(theta0_vals, theta1_vals, J_vals.T)
ax.set_xlabel(r'$\theta$0')
ax.set_ylabel(r'$\theta$1')

# 绘制等高线图
fig2 = plt.figure()
ax2 = fig2.add_subplot(111)
ax2.contour(theta0_vals, theta1_vals, J_vals.T, np.logspace(-2, 3, 20))
ax2.plot(theta[0], theta[1], 'rx', ms=10, lw=2)
ax2.set_xlabel(r'$\theta$0')
ax2.set_ylabel(r'$\theta$1')
plt.show()
