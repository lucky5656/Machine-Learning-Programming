from sklearn import tree  # 导入决策树
from sklearn.datasets import load_iris  # 导入datasets创建数组

iris = load_iris()
iris_data = iris.data  # 选择训练数组
iris_target = iris.target  # 选择对应标签数组

clf = tree.DecisionTreeClassifier()  # 创建决策树模型
clf = clf.fit(iris_data, iris_target)  # 拟合模型
import graphviz  # 导入决策树可视化模块

dot_data = tree.export_graphviz(clf, out_file=None)  # 以DOT格式导出决策树
graph = graphviz.Source(dot_data)
graph.render(r'iris')  # 使用garDphviDz将决策树转存PDF存放到桌面，文件名叫iris