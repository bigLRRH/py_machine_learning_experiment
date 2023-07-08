import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
#  1.加载sklearn自带的数据集，使用DataFrame形式探索数据。
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
# 3.建立BP模型（评估后可进行调参，从而选择最优参数）。
from sklearn.neural_network import MLPClassifier
clf = MLPClassifier()
# 4.进行模型训练。
clf.fit(X_train,y_train)
# 5.进行模型预测，对真实数据和预测数据进行可视化（用Axes3D绘制3d散点图）。
from mpl_toolkits.mplot3d import Axes3D#导入三维显示工具
#真实数据
y_hat = clf.predict(X_test)
ax = Axes3D(plt.figure())
ax.scatter(X_test[:,0],X_test[:,1],y_test)
plt.title("y_train")
plt.show()
#预测数据
ax = Axes3D(plt.figure())
ax.scatter(X_test[:,0],X_test[:,1],y_hat)
plt.title("y_hat")
plt.show()
# 6.进行模型评估，并进行预测结果指标统计（统计每一类别的预测准确率、召回率、F1分数）。
from sklearn.metrics import classification_report, confusion_matrix# 导入预测指标计算函数和混淆矩阵计算函数
classification_report(y_test, y_hat)
# 7.计算混淆矩阵，并用热力图显示。
# 计算混淆矩阵
confusion_mat = confusion_matrix(y_test,y_hat)
# 打印混淆矩阵
print(confusion_mat)
import seaborn as sns# 导入绘图包
sns.heatmap(confusion_mat)
plt.show()
# 注：混淆矩阵（confusion matrix）衡量的是一个分类器分类的准确程度。

#    混淆矩阵的每一列代表了预测类别 ，每一列的总数表示预测为该类别的数据的数目；每一行代表了数据的真实归属类别 ，每一行的数据总数表示该类别的数据实例的数目。