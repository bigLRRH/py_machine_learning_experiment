###################################################################

#一、加载糖尿病数据集diabetes，观察数据

###################################################################
#1.载入糖尿病情数据库diabetes，查看数据。
import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt
from sklearn.datasets import load_diabetes#从sklearn数据集库导入diabetes数据
diabetes = load_diabetes()
print(diabetes.keys())
print(diabetes.feature_names)
#2.切分数据，组合成DateFrame数据，并输出数据集前几行，观察数据。
dm = pd.DataFrame(diabetes.data)#将data转换为DataFrame格式
print(dm.head(6))
dm_target = pd.DataFrame(diabetes.target)    #将target转换为DataFrame格式
print(dm_target.head(6))
###################################################################

#二、基于线性回归对数据集进行分析

###################################################################
#3.查看数据集信息，从数据集中抽取训练集和测试集。
from sklearn.model_selection import train_test_split#导入数据划分包 
# 把dm、dm_target转化为数组形式，以便于计算
X = np.array(dm.values)
y = np.array(dm_target.values)
# 以25%的数据构建测试样本，剩余作为训练样本
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.25)
X_train.shape,X_test.shape,y_train.shape,y_test.shape
#4.建立线性回归模型，训练数据，评估模型。
from sklearn.linear_model import LinearRegression  #使用LinearRegression库
lr=LinearRegression()   #设定回归算法
lr.fit(X_train,y_train) #使用训练数据进行参数求解
print ('求解截距项为：',lr.intercept_)  #打印截距的值
print ('求解系数为：',lr.coef_)         #打印权重向量的值
#模型预测
y_hat = lr.predict(X_test) #对测试集的预测
#模型评估
#y_test与y_hat的可视化
plt.figure(figsize=(10,6))  #设置图片尺寸
t = np.arange(len(X_test))  #创建t变量
plt.plot(t, y_test, 'r', linewidth=2, label='y_test') #绘制y_test曲线
plt.plot(t, y_hat, 'g', linewidth=2, label='y_hat')   #绘制y_hat曲线
plt.legend() #设置图例
plt.xlabel('test data')
plt.ylabel('target')
plt.show()
###################################################################

#三、考察每个特征值与结果之间的关联性，观察得出最相关的特征

###################################################################
print(X_test)
#5.考察每个特征值与结果之间的关系，分别以散点图展示。
max_i=0
tmp = 0
for i in range(len(diabetes.feature_names)):
    plt.figure(figsize=(10,6))   #绘制图片尺寸
    plt.scatter(X_test[:,i],y_hat)  #绘制散点图
    plt.xlabel(diabetes.feature_names[i])   #设置X轴坐标轴标签
    plt.ylabel('predicted')      #设置y轴坐标轴标签
    rho=np.corrcoef(X_test[:,i],y_hat[:,0])#求pearson相关系数
    if abs(rho[0,1])>tmp:
        tmp = abs(rho[0,1])
        max_i = i
    print(i,rho[0,1])
#思考：根据散点图结果对比，哪个特征值与结果之间的相关性最高？

###################################################################

#四、使用回归分析找出XX特征值与糖尿病的关联性，并预测出相关结果

###################################################################
# 6.把5中相关性最高的特征值提取，然后进行数据切分。
X = dm.iloc[:,max_i:max_i+1]#选取data中的最高的特征值变量
y = dm_target

plt.scatter(X, y)    #绘制散点图
plt.xlabel(u'ltg')    #x轴标签
plt.ylabel(u'target')  #y轴标签
plt.title(u'The relation of ltg and target') #标题
plt.show()

# 把X、y转化为数组形式，以便于计算
X = np.array(X.values)  
y = np.array(y.values) 
# 以25%的数据构建测试样本，剩余作为训练样本
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.25)
X_train.shape,X_test.shape,y_train.shape,y_test.shape

# 8.创建线性回归模型，进行线性回归模型训练。
lr=LinearRegression()   #设定回归算法
lr.fit(X_train,y_train) #使用训练数据进行参数求解
# 9.对测试集进行预测，求出权重系数。
y_hat = lr.predict(X_test) #对测试集的预测
print ('求解截距项为：',lr.intercept_)  #打印截距的值
print ('求解系数为：',lr.coef_)         #打印权重向量的值
# 10.对预测结果进行评价，结果可视化。
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
plt.plot()
plt.xlabel('ground truth')   #设置X轴坐标轴标签
plt.ylabel('predicted')      #设置y轴坐标轴标签
plt.grid()  #绘制网格线

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


# 6.把5中相关性最高的特征值提取，然后进行数据切分。
X = dm.iloc[:,2:3]#选取data中的最高的特征值变量
y = dm_target

# 把X、y转化为数组形式，以便于计算
X = np.array(X.values)  
y = np.array(y.values) 
# 以25%的数据构建测试样本，剩余作为训练样本
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.25)
X_train.shape,X_test.shape,y_train.shape,y_test.shape

# 8.创建线性回归模型，进行线性回归模型训练。
lr=LinearRegression()   #设定回归算法
lr.fit(X_train,y_train) #使用训练数据进行参数求解
# 9.对测试集进行预测，求出权重系数。
y_hat = lr.predict(X_test) #对测试集的预测
print ('求解截距项为：',lr.intercept_)  #打印截距的值
print ('求解系数为：',lr.coef_)         #打印权重向量的值
# 10.对预测结果进行评价，结果可视化。
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
plt.plot()
plt.xlabel('ground truth')   #设置X轴坐标轴标签
plt.ylabel('predicted')      #设置y轴坐标轴标签
plt.grid()  #绘制网格线

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