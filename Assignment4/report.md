 <h3><center>Report of Classification</center></h3>

 <h6><center>张茗曈 2017011214</center></h6>

#### Qusetion 1 K-Means clustering

对于 k-means 聚类部分的代码实现，遵循最小误差平方和准则，首先对于数据集中的各个点计算其到达各个质心之间的欧式距离，将各个点分配到距其最近的簇，对于每个簇，计算该簇中的哥哥数据点的均值并将均值作为质心，进行更新；

#### Question 2 KNN classifier

对于 KNN 部分的代码实现，根据 KNN 的思想，即数据点在特征空间中的 K 个最近邻样本属于该类，则将该数据点也归为此类；计算已知分类的点与当前分类点的距离，按照距离进行排序，选取与当前点的距离最小的 K 个点统计所在的类出现的次数，将次数出现最多的类作为该点的预测类；

#### Question 3 Perceptron

该部分需要完成训练过程的代码实现，核心为权重和偏置量的更新，即：
$$
w_j^{(t+1)}=w_j^{(t)}-\eta\lambda w^{(t)}_j-\frac{\eta}{k}\sum^k_{i=1}\nabla_{w_j}-\log p(y=y_i|x_i)|_{w_j=w_j^{(t)}}\\
b_j^{(t+1)}=b_j^{(t)}-\eta\lambda b^{(t)}_j-\frac{\eta}{k}\sum^k_{i=1}\nabla_{b_j}-\log p(y=y_i|x_i)|_{b_j=b_j^{(t)}}
$$

##### Question 1

下图为感知机模型的权重：

![weights](https://img.wzf2000.top/image/2021/05/19/weights.png)

对于权重的可视化结果，可以基本看出分为 10 个数字，图中较亮的地方表示该位置的权重参数值较大，而数字的轮廓已经变得非常模糊；

#### Question 4 SVM with sklearn

该部分的实现较简单，即调用 `svm.SVC` 实现即可，按照指导书中给出的参数设置，将 `C` 设为 $5$，kernel type 利用缺省值即可，对 gamma 参数进行手动设置即可，该参数决定数据在特征空间中映射后的分布，gamma 越大，支持向量越少，反之则越多，根据：
$$
k(x,y)=\exp(-\frac{d(x,y)}{2\sigma^2})=\exp(-\text{gamma}\cdot d(x,y)^2)\Rightarrow\text{gamma}=\frac{1}{2\sigma^2}
$$
以及 $\sigma=10$，计算得 gamma 为 0.005；

#### Question 5 Better Classification Accuracy

该部分根据指导书中的提示，可更改分类算法、参数以及优化特征等方式进行实现，在尝试决策树、贝叶斯分类等方法后发现性能较差，最终通过支持向量机并通过调参实现较好的效果；