import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.io as scio
from sklearn import svm 
import re #处理正则表达式的模块
import nltk #自然语言处理工具包

'''============================part1 邮件预处理========================='''

#查看样例邮件
f = open('emailSample1.txt', 'r').read()
print(f)

def processEmail(email):
    email = email.lower() #转化为小写
    email = re.sub('<[^<>]+>', ' ', email) #移除所有HTML标签
    email = re.sub('(http|https)://[^\s]*', 'httpaddr', email) #将所有的URL替换为'httpaddr'
    email = re.sub('[^\s]+@[^\s]+', 'emailaddr', email) #将所有的地址替换为'emailaddr'
    email = re.sub('\d+', 'number', email) #将所有数字替换为'number'
    email = re.sub('[$]+', 'dollar', email) #将所有美元符号($)替换为'dollar'
    
    #将所有单词还原为词根//移除所有非文字类型，空格调整
    stemmer = nltk.stem.PorterStemmer() #使用Porter算法
    tokens = re.split('[ @$/#.-:&*+=\[\]?!()\{\},\'\">_<;%]', email) #把邮件分割成单个的字符串,[]里面为各种分隔符
    tokenlist = []
    for token in tokens:
        token = re.sub('[^a-zA-Z0-9]', '', token) #去掉任何非字母数字字符
        try: #porterStemmer有时会出现问题,因此用try
            token = stemmer.stem(token) #词根
        except:
            token = ''
        if len(token) < 1: 
            continue #字符串长度小于1的不添加到tokenlist里
        tokenlist.append(token)
    
    return tokenlist

#查看处理后的样例
processed_f = processEmail(f)
for i in processed_f:
    print(i, end=' ')

#得到单词表，序号为索引号+1
vocab_list = np.loadtxt('vocab.txt', dtype='str', usecols=1)
#得到词汇表中的序号
def word_indices(processed_f, vocab_list):
    indices = []
    for i in range(len(processed_f)):
        for j in range(len(vocab_list)):
            if processed_f[i]!=vocab_list[j]:
                continue
            indices.append(j+1)
    return indices

#查看样例序号
f_indices = word_indices(processed_f, vocab_list)
for i in f_indices:
    print(i, end=' ')
    
input('Program paused. Press enter to continue')

'''============================part2 提取特征========================='''
def emailFeatures(indices):
    features = np.zeros((1899))
    for each in indices:
        features[each-1] = 1 #若indices在对应单词表的位置上词语存在则记为1
    return features

sum(emailFeatures(f_indices)) #45

input('Program paused. Press enter to continue')


'''============================part3 训练SVM========================='''
#训练模型
train = scio.loadmat('spamTrain.mat')
train_x = train['X']
train_y = train['y']

clf = svm.SVC(C=0.1, kernel='linear')
clf.fit(train_x, train_y)

#精度
def accuracy(clf, x, y):
    predict_y = clf.predict(x)
    m = y.size
    count = 0
    for i in range(m):
        count = count + np.abs(int(predict_y[i])-int(y[i])) #避免溢出错误得到225
    return 1-float(count/m) 

accuracy(clf, train_x, train_y) #0.99825
print('train Accuracy:')
print(accuracy(clf, train_x, train_y))

#测试模型
test = scio.loadmat('spamTest.mat')
accuracy(clf, test['Xtest'], test['ytest']) #0.989
print('test Accuracy:')
print(accuracy(clf, test['Xtest'], test['ytest']))
 
input('Program paused. Press enter to continue')


'''============================part4 高权重词========================='''
#打印权重最高的前15个词,邮件中出现这些词更容易是垃圾邮件
i = (clf.coef_).size-1
while i >1883:
    #返回从小到大排序的索引，然后再打印
    print(vocab_list[np.argsort(clf.coef_).flatten()[i]], end=' ')
    i = i-1
    
input('Program paused. Press enter to continue')

'''============================part5 预测邮件========================='''

t = open('spamSample2.txt', 'r').read()
#预处理
processed_f = processEmail(t) 
f_indices = word_indices(processed_f, vocab_list)
#特征提取
x = np.reshape(emailFeatures(f_indices), (1,1899))
#预测
clf.predict(x)
