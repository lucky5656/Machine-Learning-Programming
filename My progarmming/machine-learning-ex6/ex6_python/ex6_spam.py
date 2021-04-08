import matplotlib.pyplot as plt
import numpy as np
import scipy.io as scio
from sklearn import svm
import re#处理正则表达式的模块
import nltk, nltk.stem.porter
 
plt.ion()#使matplotlib的显示模式转换为交互(interactive)模式
np.set_printoptions(formatter={'float': '{: 0.6f}'.format})
 
'''================================================part1 邮件预处理============================================='''

def process_email(email_contents):
    vocab_list = get_vocab_list()#vocab_list为1899x1
    word_indices = np.array([], dtype=np.int64)
 
    # ===================== Preprocess Email =====================
 
    email_contents = email_contents.lower()
    email_contents = re.sub('<[^<>]+>', ' ', email_contents)
    email_contents = re.sub('[0-9]+', 'number', email_contents)
    email_contents = re.sub('(http|https)://[^\s]*', 'httpaddr', email_contents)
    email_contents = re.sub('[^\s]+@[^\s]+', 'emailaddr', email_contents)
    email_contents = re.sub('[$]+', 'dollar', email_contents)
 
    # ===================== Tokenize Email =====================
 
    print('==== Processed Email ====')
    stemmer = nltk.stem.porter.PorterStemmer()
    tokens = re.split('[@$/#.-:&*+=\[\]?!(){\},\'\">_<;% ]', email_contents)
 
    for token in tokens:
        token = re.sub('[^a-zA-Z0-9]', '', token)
        token = stemmer.stem(token)
 
        if len(token) < 1:
            continue
 
        for i in range(1, len(vocab_list) + 1):
            if vocab_list[i] == token:
                word_indices = np.append(word_indices, i)

        print(token)
 
    print('==================')
 
    return word_indices
 
def get_vocab_list(): #得到词汇列表
    vocab_dict = {}     #新建空字典 并以字典形式获取
    with open('vocab.txt') as f:
        for line in f:
            (val, key) = line.split()  #读取每一行的键和值
            vocab_dict[int(val)] = key #存放到字典中
 
    return vocab_dict

# ===================== Part 1: Email Preprocessing =====================
 
print('Preprocessing sample email (emailSample1.txt) ...')
 
file_contents = open('emailSample1.txt', 'r').read()
word_indices = process_email(file_contents)
 
# Print stats
print('Word Indices: ')
print(word_indices)
 
input('Program paused. Press ENTER to continue')
 
'''================================================part2 提取特征============================================='''

def email_features(word_indices):
    # Total number of words in the dictionary
    n = 1899
 
    # Since the index of numpy array starts at 0, to align with the word indices we make n + 1 size array
    features = np.zeros((n + 1,1))#(1-1899在python中从0开始，要得到1899需要1900大小的元组)
 
    for i in word_indices:
        features[i] =  1
 
    return features

# ===================== Part 2: Feature Extraction =====================
 
print('Extracting Features from sample email (emailSample1.txt) ... ')
 
# Extract features
features = email_features(word_indices)
 
# Print stats
print('Length of feature vector: {}'.format(features.size))
print('Number of non-zero entries: {}'.format(np.flatnonzero(features).size))
 
input('Program paused. Press ENTER to continue')

'''================================================part3 训练SVM============================================='''

# ===================== Part 3: Train Linear SVM for Spam Classification =====================
 
data = scio.loadmat('spamTrain.mat')
X = data['X']
y = data['y'].flatten()
 
print('Training Linear SVM (Spam Classification)')
print('(this may take 1 to 2 minutes)')
 
c = 0.1
clf = svm.SVC(c, kernel='linear')
clf.fit(X, y)
 
p = clf.predict(X)#X 4000x1899
 
print('Training Accuracy: {}'.format(np.mean(p == y) * 100))

# ===================== Part 4: Test Spam Classification =====================

# Load the test dataset
data = scio.loadmat('spamTest.mat')
Xtest = data['Xtest']
ytest = data['ytest'].flatten()
 
print('Evaluating the trained linear SVM on a test set ...')
 
p = clf.predict(Xtest)#Xtest 1000x1899
 
print('Test Accuracy: {}'.format(np.mean(p == ytest) * 100))
 
input('Program paused. Press ENTER to continue')

'''================================================part4 高权重词============================================='''

# ===================== Part 5: Top Predictors of Spam =====================
 
vocab_list = get_vocab_list()
indices = np.argsort(clf.coef_).flatten()[::-1]
print(indices)
 
for i in range(15):
    print('{} ({:0.6f})'.format(vocab_list[indices[i]], clf.coef_.flatten()[indices[i]]))
 
input('Program paused. Press ENTER to continue')

# =================== Part 6: Try Your Own Emails =====================

filename = 'spamSample1.txt';
file_contents = open(filename, 'r').read()
word_indices = process_email(file_contents)
x = email_features(word_indices)#x 1900x1
p = clf.predict(x[1:,:].T)#x[1:,:].T 1X1899

print('\nProcessed ', filename,'\nSpam Classification:', p);
print('(1 indicates spam, 0 indicates not spam)\n\n');

input('ex6_spam Finished. Press ENTER to exit')
