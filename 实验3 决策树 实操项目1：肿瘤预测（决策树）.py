import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
# 1.加载sklearn自带的威斯康星乳腺癌数据集，探索数据。
from sklearn.datasets import load_breast_cancer
breast_cancer = load_breast_cancer() 
#探索数据
print('target_names:',breast_cancer.target_names)
print('feature_names:',breast_cancer.feature_names)
# 2.进行数据集分割。
bc = pd.DataFrame(breast_cancer.data)
print(bc.head(6))
bc_target = pd.DataFrame(breast_cancer.target)
print(bc_target.head(6))

from sklearn.model_selection import train_test_split#导入数据划分包 
X = np.array(bc.values)
y = np.array(bc_target.values)
    # 以25%的数据构建测试样本，剩余作为训练样本
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.25)
X_train.shape,X_test.shape,y_train.shape,y_test.shape
# 3.配置决策树模型。
from sklearn import tree # 导入决策树包
clf = tree.DecisionTreeClassifier() #加载决策树模型
# 4.训练决策树模型。
clf.fit(X_train, y_train) # 模型训练，取前五分之四作训练集
# 5.模型预测。
y_hat = clf.predict(X_test)
# 6.模型评估。
from sklearn.metrics import accuracy_score # 导入准确率评价指标
print('Accuracy:%s'% accuracy_score(y_test, y_hat))
# 7.参数调优。可以根据评估结果，对模型设置或调整为更优的参数，使评估结果更准确。
score = 0
effective_c=0
effective_d=0
criterions = ['gini','entropy']
for c in criterions:
    for d in range(1,100):
        clf = tree.DecisionTreeClassifier(criterion=c,max_depth=d)
        clf.fit(X_train, y_train) # 模型训练，取前75%作训练集
        y_hat = clf.predict(X_test)# 5.模型预测。
        current_score = accuracy_score(y_test, y_hat)
        if current_score>score:
            score=current_score
            effective_c=c
            effective_d=d

clf = tree.DecisionTreeClassifier(criterion=c,max_depth=d)
clf.fit(X_train, y_train) # 模型训练，取前75%作训练集
y_hat = clf.predict(X_test)# 5.模型预测。
print(effective_c,effective_d)
print('Accuracy:%s'% accuracy_score(y_test, y_hat))