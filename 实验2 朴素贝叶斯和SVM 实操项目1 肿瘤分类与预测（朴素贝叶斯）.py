import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
# 1.导入sklearn自带的数据集：威斯康星乳腺肿瘤数据集（load_breast_cancer）。
from sklearn.datasets import load_breast_cancer
breast_cancer = load_breast_cancer() 
# 2.打印数据集键值（keys），查看数据集包含的信息。
print(breast_cancer.keys())
# 3.打印查看数据集中标注好的肿瘤分类（target_names）、肿瘤特征名称（feature_names）。
print('target_names:',breast_cancer.target_names)
print('feature_names:',breast_cancer.feature_names)
# 4.将数据集拆分为训练集和测试集，打印查看训练集和测试集的数据形态（shape）。
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

# 5.配置高斯朴素贝叶斯模型。
from sklearn.naive_bayes import GaussianNB    #导入先验概率为高斯分布的朴素贝叶斯
model = GaussianNB()
# 6.训练模型。
model.fit(X_train,y_train) #训练高斯朴素贝叶斯算法模型
# 7.评估模型，打印查看模型评分（分别打印训练集和测试集的评分）。
from sklearn.model_selection import cross_val_score
sorce = cross_val_score(model,X_train,y_train,cv=10,scoring='accuracy') #计算高斯朴素贝叶斯算法模型的准确率
print("高斯朴素贝叶斯模型的准确率：",sorce.mean())
print("高斯朴素贝叶斯模型训练集的得分：",model.score(X_train,y_train)) #高斯朴素贝叶斯算法模型训练集得分
print("高斯朴素贝叶斯模型测试集的得分：",model.score(X_test,y_test))   #高斯朴素贝叶斯算法模型测试集得分
# 8.模型预测：选取某一样本进行预测。（可以进行多次不同样本的预测）
#   参考方法：可以打印模型预测的分类和真实的分类，进行对比，看是否一致，如果一致，判断这个样本的肿瘤是一个良性的肿瘤，否则结果相反。
#           也可以用其他方法进行预测。
y_hat = model.predict(X_test)
#y_test与y_hat的可视化
plt.figure(figsize=(10,6))  #设置图片尺寸
t = np.arange(len(X_test))  #创建t变量
plt.plot(t, y_test, 'r', linewidth=2, label='y_test') #绘制y_test曲线
plt.plot(t, y_hat, 'g', linewidth=2, label='y_hat')   #绘制y_hat曲线
plt.legend() #设置图例
plt.xlabel('test data')
plt.ylabel('target')
plt.show()
# 9.扩展（选做）：绘制高斯朴素贝叶斯在威斯康星乳腺肿瘤数据集中的学习曲线。
from sklearn.model_selection import learning_curve
from sklearn.model_selection import ShuffleSplit

def plot_learning_curve(estimator, title, X, y, ylim=None, cv=None,
                        n_jobs=1, train_sizes=np.linspace(.1, 1.0, 5)):
    plt.figure()
    plt.title(title)
    if ylim is not None:
        plt.ylim(*ylim)
    plt.xlabel("Training examples")
    plt.ylabel("Score")
    train_sizes, train_scores, test_scores = learning_curve(
        estimator, X, y, cv=cv, n_jobs=n_jobs, train_sizes=train_sizes)
    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)
    plt.grid()
 
    plt.fill_between(train_sizes, train_scores_mean - train_scores_std,
                     train_scores_mean + train_scores_std, alpha=0.1,
                     color="r")
    plt.fill_between(train_sizes, test_scores_mean - test_scores_std,
                     test_scores_mean + test_scores_std, alpha=0.1, color="g")
    plt.plot(train_sizes, train_scores_mean, 'o-', color="r",
             label="Training score")
    plt.plot(train_sizes, test_scores_mean, 'o-', color="g",
             label="Cross-validation score")
 
    plt.legend(loc="best")
    return plt


title = r"Learning Curves (Naive Bayes)"
cv = ShuffleSplit(n_splits=100, test_size=0.2, random_state=0)
estimator = GaussianNB()    #建模
plot_learning_curve(estimator, title, X, y, ylim=(0.7, 1.01), cv=cv, n_jobs=1)