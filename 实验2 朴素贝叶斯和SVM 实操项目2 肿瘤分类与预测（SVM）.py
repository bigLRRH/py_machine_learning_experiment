#!/usr/bin/env python
# coding: utf-8

# 请点击[此处](https://ai.baidu.com/docs#/AIStudio_Project_Notebook/a38e5576)查看本环境基本用法.  <br>
# Please click [here ](https://ai.baidu.com/docs#/AIStudio_Project_Notebook/a38e5576) for more detailed instructions. 

# # 【实验内容】肿瘤分类与预测（SVM）
# 
#     采用SVM方法，对美国威斯康星州的乳腺癌诊断数据集进行分类，实现针对乳腺癌检测的分类器，以判断一个患者的肿瘤是良性还是恶性。
# 
# 
# # 【实验要求】
# 
#    **参考实现步骤：（具体实现可以不同）**
#     
#     1.加载data文件夹里的数据集：威斯康星乳腺肿瘤数据集（数据集路径：data/data74924/data.csv）。
#     
#     2.查看样本特征和特征值，查看样本特征值的描述信息。
#     
#     3.进行数据清洗（如删除无用列，将诊断结果的字符标识B、M替换为数值0、1等）。
#     
#     4.进行特征选取（方便后续的模型训练）。用热力图呈现features_mean字段之间的相关性，从而选取特征。
#       
#       注：（1）热力图中，颜色越浅代表相关性越大。
#       
#          （2）通过热力图找到相关性大的几个属性，每组相关性大的属性只选一个属性做代表。这样就可以把10个属性缩小。
#     
#     5.进行数据集的划分（训练集和测试集），抽取特征选择的数值作为训练和测试数据。
#     
#     6.进行数据标准化操作（可采用Z-Score规范化数据）。
#     
#     7.配置模型，创建SVM分类器。
#     
#     8.训练模型。
#     
#     9.模型预测。
#     
#     10.模型评估。       

# # **【数据集】：威斯康星乳腺肿瘤数据集**
# 
#     该数据集中肿瘤是一个非常经典的用于医疗病情分析的数据集，包括569个病例的数据样本，每个样本具有30个特征。
#     
#     样本共分为两类：恶性（Malignant）和良性（Benign）。
# 
#     该数据集的特征是从一个乳腺肿块的细针抽吸（FNA）的数字化图像计算出来的。它们描述了图像中细胞核的特征。
#     
#     特征值很多，涉及一定的医学知识。（具体特征及含义见此节实验指导书）

# In[ ]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
# 1.加载data文件夹里的数据集：威斯康星乳腺肿瘤数据集（数据集路径：data/data74924/data.csv）。
breast_cancer = pd.read_csv('data/data74924/data.csv')
# 2.查看样本特征和特征值，查看样本特征值的描述信息。
pd.set_option('display.max_columns', None)
print(breast_cancer.keys())
print(breast_cancer.head(6))
print(breast_cancer.describe())
# 3.进行数据清洗（如删除无用列，将诊断结果的字符标识B、M替换为数值0、1等）。
# 将特征字段分成3组
features_mean= list(breast_cancer.columns[2:12])#平均值
features_se= list(breast_cancer.columns[12:22])#标准差
features_worst=list(breast_cancer.columns[22:32])#最大值
# ID列没有用，删除该列
breast_cancer.drop("id",axis=1,inplace=True)
# 将B良性替换为0，M恶性替换为1
breast_cancer['diagnosis']=breast_cancer['diagnosis'].map({'M':1,'B':0})


# In[10]:


# 4.进行特征选取（方便后续的模型训练）。用热力图呈现features_mean字段之间的相关性，从而选取特征。
#   注：（1）热力图中，颜色越浅代表相关性越大。
#      （2）通过热力图找到相关性大的几个属性，每组相关性大的属性只选一个属性做代表。这样就可以把10个属性缩小。
import seaborn as sns
# 用热力图呈现features_mean字段之间的相关性
corr = breast_cancer[features_mean].corr()
# annot=True显示每个方格的数据
sns.heatmap(corr, annot=True)
plt.show()


# In[21]:


# 特征选择
features_remain = ['radius_mean','texture_mean',  'smoothness_mean','compactness_mean','symmetry_mean', 'concavity_mean'] 
X = np.array(breast_cancer[features_remain])
y = np.array(breast_cancer['diagnosis'])
    # 以25%的数据构建测试样本，剩余作为训练样本
from sklearn.model_selection import train_test_split#导入数据划分包 
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.25)
X_train.shape,X_test.shape,y_train.shape,y_test.shape
# 6.进行数据标准化操作（可采用Z-Score规范化数据）。
from sklearn.preprocessing import StandardScaler
ss = StandardScaler()
X_train=ss.fit_transform(X_train)
X_test = ss.fit_transform(X_test)
# 7.配置模型，创建SVM分类器。
from sklearn.svm import SVC
classifier=SVC(kernel='linear',C=10)
# 8.训练模型。
classifier.fit(X_train,y_train) 
# 9.模型预测。
y_hat = classifier.predict(X_test)
# 10.模型评估。       
from sklearn.metrics import classification_report  #导入分类报告模板
print('分类精度为：',classifier.score(X_test,y_test)*100,'%')  #模型评估
print('评价指标：\n',classification_report(y_test,y_hat))


# In[29]:


# 5.进行数据集的划分（训练集和测试集），抽取特征选择的数值作为训练和测试数据。
corr = breast_cancer.corr()
# annot=True显示每个方格的数据
sns.heatmap(corr, annot=False)
plt.show()

from sklearn.model_selection import train_test_split#导入数据划分包 
#根据热力图选择1，2，5，6，9，10，11，12，15，16，18，19，29
features_remain2 = [1,2,5,6,9,10,11,12,15,16,18,19,29]
X = np.array(breast_cancer.iloc[:,features_remain2])
y = np.array(breast_cancer['diagnosis'])
    # 以25%的数据构建测试样本，剩余作为训练样本
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.25)
X_train.shape,X_test.shape,y_train.shape,y_test.shape
# 6.进行数据标准化操作（可采用Z-Score规范化数据）。
from sklearn.preprocessing import StandardScaler
ss = StandardScaler()
X_train=ss.fit_transform(X_train)
X_test = ss.fit_transform(X_test)
# 7.配置模型，创建SVM分类器。
from sklearn.svm import SVC
classifier=SVC(kernel='linear',C=10)
# 8.训练模型。
classifier.fit(X_train,y_train) 
# 9.模型预测。
y_hat = classifier.predict(X_test)
# 10.模型评估。       
from sklearn.metrics import classification_report  #导入分类报告模板
print('分类精度为：',classifier.score(X_test,y_test)*100,'%')  #模型评估
print('评价指标：\n',classification_report(y_test,y_hat))

#其他评估方法：采用准确率(accuracy)作为评估函数：预测结果正确的数量占样本总数，(TP+TN)/(TP+TN+FP+FN)
from sklearn.metrics import accuracy_score # 导入准确率评价指标
print('Accuracy:%s' % accuracy_score(y_test, y_hat))