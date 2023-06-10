import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
# 1.读取顾客购买服装的数据集（数据集路径：data/data76088/3_buy.csv），探索数据。
data = pd.read_csv("data/data76088/3_buy.csv")
print(data)
# 2.分别用ID3算法和CART算法进行决策树模型的配置、模型的训练、模型的预测、模型的评估。

from sklearn.model_selection import train_test_split  #导入数据划分包
X=np.array(data.iloc[:,0:4])
y=np.array(data.iloc[:,4:5])
# 以25%的数据构建测试样本，剩余作为训练样本
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.25)
X_train.shape,X_test.shape,y_train.shape,y_test.shape


from sklearn import tree # 导入决策树包
    #加载决策树模型
clf1 = tree.DecisionTreeClassifier(criterion="entropy") 
clf2 = tree.DecisionTreeClassifier(criterion='gini') 
    #使用训练数据进行参数求解
clf1.fit(X_train,y_train) 
clf2.fit(X_train,y_train) 
# 6.根据求出的参数对测试集进行预测。
y1_hat = clf1.predict(X_test) #对测试集的预测
y2_hat = clf2.predict(X_test)

from sklearn.metrics import accuracy_score # 导入准确率评价指标
print('Accuracy:%s'% accuracy_score(y_test, y1_hat))
print('Accuracy:%s'% accuracy_score(y_test, y2_hat))

# 3.扩展内容（选做）：对不同算法生成的决策树结构图进行可视化。
from sklearn import tree
print('clf1')
print(tree.export_text(clf1))
print('clf2')
print(tree.export_text(clf2))
