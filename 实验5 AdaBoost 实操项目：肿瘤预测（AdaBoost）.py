import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
#  1.加载sklearn自带的数据集，使用DataFrame形式探索数据。
#加载
from sklearn.datasets import load_breast_cancer
breast_cancer = load_breast_cancer()
#使用dataframe形式探索数据
bc = pd.DataFrame(breast_cancer.data)
print(bc.head(6))
bc_target = pd.DataFrame(breast_cancer.target)
print(bc_target.head(6))
#  2.划分训练集和测试集，检查训练集和测试集的平均癌症发生率。
from sklearn.model_selection import train_test_split#导入数据划分包 
X = np.array(bc.values)
y = np.array(bc_target.values).ravel()
    # 以25%的数据构建测试样本，剩余作为训练样本
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.25)
X_train.shape,X_test.shape,y_train.shape,y_test.shape
print('训练集平均癌症发生率(y_train_mean):',np.mean(y_train))
print('测试集平均癌症发生率(y_test_mean):',np.mean(y_test))
#  3.配置模型，训练模型，模型预测，模型评估。
from sklearn import tree # 导入决策树包
from sklearn.ensemble import AdaBoostClassifier # 导入 AdaBoost 包
from sklearn.metrics import accuracy_score # 导入准确率评价指标
#   （1）构建一棵最大深度为2的决策树弱学习器，训练、预测、评估。
estimator = tree.DecisionTreeClassifier(max_depth=2)
estimator.fit(X_train,y_train)
y1_hat = estimator.predict(X_test)
estimator_accuracy_score=accuracy_score(y_test,y1_hat)
print("estimator_accuracy_score:",estimator_accuracy_score)
#   （2）再构建一个包含50棵树的AdaBoost集成分类器（步长为3），训练、预测、评估。   
#       参考：将决策树的数量从1增加到50，步长为3。输出集成后的准确度。
i=1
aSs=[]
eNs=[]
max_eN=0
max_aS=0
while i<50:
    clf = AdaBoostClassifier(n_estimators=i)
    clf.fit(X_train,y_train)
    y2_hat=clf.predict(X_test)
    aS=accuracy_score(y_test,y2_hat)
    print('n_estimators:',i,"accuracy_score:",aS)
    aSs.append(aS)
    eNs.append(i)
    if aS>max_aS:
        max_aS=aS
        max_eN=i
    i+=3
#   （3）将（2）的性能与弱学习者进行比较。
print('estimator:',estimator_accuracy_score)
print('clf:',max_aS)
#  4.绘制准确度的折线图，x轴为决策树的数量，y轴为准确度。
import matplotlib.pyplot as plt
plt.plot(eNs,aSs,[0,50],[estimator_accuracy_score,estimator_accuracy_score])
plt.xlabel='n_estimator'
plt.ylabel='accuracy_score'
plt.show()