import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt
# 1.读取给定文件中数据集文件。（数据集路径：data/data72160/1_film.csv）
data = pd.read_csv('data/data72160/1_film.csv')
print(data)
# 2.绘制影厅观影人数（filmnum）与影厅面积（filmsize）的散点图。
X =  data['filmsize']
y =  data['filmnum']
plt.scatter(X, y)    #绘制散点图
plt.xlabel(u'filmsize')    #x轴标签
plt.ylabel(u'filmnum')  #y轴标签
plt.title(u'The relation of filmsize and filmnum') #标题
plt.show()
# 3.绘制影厅人数数据集的散点图矩阵。
from pandas.plotting import scatter_matrix
scatter_matrix(data)
# 4.选取特征变量与相应变量，并进行数据划分。
from sklearn.model_selection import train_test_split  #导入数据划分包
X = np.array(data.iloc[:,1:4])
y = np.array(data.iloc[:,0:1])
# 以25%的数据构建测试样本，剩余作为训练样本
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.25)
X_train.shape,X_test.shape,y_train.shape,y_test.shape
# 5.进行线性回归模型训练。
from sklearn.linear_model import LinearRegression  #使用LinearRegression库
lr=LinearRegression()   #设定回归算法
lr.fit(X_train,y_train) #使用训练数据进行参数求解
print ('求解截距项为：',lr.intercept_)  #打印截距的值
print ('求解系数为：',lr.coef_)         #打印权重向量的值
# 6.根据求出的参数对测试集进行预测。
y_hat = lr.predict(X_test) #对测试集的预测
# 7.绘制测试集相应变量实际值与预测值的比较。
#y_test与y_hat的可视化
plt.figure(figsize=(10,6))  #设置图片尺寸
t = np.arange(len(X_test))  #创建t变量

plt.plot(t, y_test, 'r', linewidth=2, label='y_test') #绘制y_test曲线
plt.plot(t, y_hat, 'g', linewidth=2, label='y_hat')   #绘制y_hat曲线

plt.legend() #设置图例

plt.xlabel('test data')
plt.ylabel('price')
plt.show()

plt.figure(figsize=(10,6))   #绘制图片尺寸
plt.plot(y_test,y_hat,'o')   #绘制散点
plt.xlabel('features')   #设置X轴坐标轴标签
plt.ylabel('predicted')      #设置y轴坐标轴标签
plt.grid()  #绘制网格线

# 8.对预测结果进行评价。
from sklearn import metrics
from sklearn.metrics import r2_score
# 拟合优度R2的输出方法一
print ("r2:",lr.score(X_test, y_test))  #基于Linear-Regression()的回归算法得分函数，来对预测集的拟合优度进行评价
# 拟合优度R2的输出方法二
print ("r2_score:",r2_score(y_test, y_hat)) #使用metrics的r2_score来对预测集的拟合优度进行评价
# 用scikit-learn计算MAE
print ("MAE:", metrics.mean_absolute_error(y_test, y_hat)) #计算平均绝对误差
# 用scikit-learn计算MSE
print ("MSE:", metrics.mean_squared_error(y_test, y_hat))  #计算均方误差
# # 用scikit-learn计算RMSE
print ("RMSE:", np.sqrt(metrics.mean_squared_error(y_test, y_hat))) #计算均方根误差