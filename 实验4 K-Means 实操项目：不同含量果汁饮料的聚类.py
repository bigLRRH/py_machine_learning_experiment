import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
# 1.加载数据集，读取数据，探索数据。（数据集路径：data/data76878/4_beverage.csv）
data = pd.read_csv("data/data76878/4_beverage.csv")
data.head()

# 2.样本数据转化（可将pandasframe格式的数据转化为数组形式），并进行可视化（绘制散点图），观察数据的分布情况，从而可以得出k的几种可能取值。
from sklearn.model_selection import train_test_split#导入数据划分包 
X = np.array(data)
plt.scatter(X[:,0],X[:,1]);

from sklearn.metrics import calinski_harabasz_score
# 3.针对每一种k的取值，进行如下操作：
for k in range(2,8):
    estimator = KMeans(n_clusters=k)  # 构建聚类器
    estimator.fit(X)
# （2）输出相关聚类结果，并评估聚类效果。
#     这里可采用CH指标来对聚类有效性进行评估。在最后用每个k取值时评估的CH值进行对比，可得出k取什么值时，聚类效果更优。
    y_hat = estimator.predict(X)
    print("k:",k,"ch:",calinski_harabasz_score(X,y_hat))
#     注：这里缺乏外部类别信息，故采用内部准则评价指标（CH）来评估。 (metrics.calinski_harabaz_score())       
# （3）输出各类簇标签值、各类簇中心，从而判断每类的果汁含量与糖分含量情况。
    print(estimator.cluster_centers_)
# （4）聚类结果及其各类簇中心点的可视化（散点图），从而观察各类簇分布情况。（不同的类表明不同果汁饮料的果汁、糖分含量的偏差情况。）
    plt.scatter(X[:,0],X[:,1],c=y_hat,label=k)
    plt.xlabel('juice')
    plt.ylabel('sweet')
    plt.scatter(
        estimator.cluster_centers_[:, 0],
        estimator.cluster_centers_[:, 1],
        marker="x",
        c="black",
        s=48
    )
    plt.show()
# 4.【扩展】（选做）：设置k一定的取值范围，进行聚类并评价不同的聚类结果。
# 参考思路：设置k的取值范围；对不同取值k进行训；计算各对象离各类簇中心的欧氏距离，生成距离表；提取每个对象到其类簇中心的距离，并相加；依次存入距离结果；绘制不同看、值对应的总距离值折线图。    