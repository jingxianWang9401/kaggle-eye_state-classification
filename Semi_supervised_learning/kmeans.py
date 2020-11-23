# -*- coding: utf-8 -*-
"""
Created on Mon Jun 22 09:36:24 2020

@author: wangjingxian
"""

from sklearn.datasets import load_iris
from sklearn.preprocessing import MinMaxScaler
from sklearn.cluster import KMeans
import pandas as pd


dataset_train=pd.read_csv('E:\data_mining\eye_classification\clustering_model\data\eeg_train.csv')
trainingSet=dataset_train.iloc[:,0:14] 
trainingLabels=dataset_train.iloc[:,[14]] 


#scale=MinMaxScaler().fit(X)#训练规则
#X_dataScale=scale.transform(X)#应用规则

kmeans=KMeans(n_clusters=2).fit(trainingSet)#构建并训练模型

print('构建的KMeans模型为：',kmeans)

print('聚类结果为：',kmeans.labels_)


quantity = pd.Series(kmeans.labels_).value_counts()
print( "聚类后每个类别的样本数量\n", (quantity))


#获取聚类之后每个聚类中心的数据
resSeries = pd.Series(kmeans.labels_)


res0 = resSeries[resSeries.values == 0]
print("聚类后类别为0的数据\n",(trainingSet.iloc[res0.index]))
train_label0=dataset_train.iloc[res0.index]
#train_label0.to_csv('E:\\data_mining\\eye_classification\\clustering_model\\result\\egg_train_test_class00.csv',index=False)




res1 = resSeries[resSeries.values == 1]
print("聚类后类别为1的数据\n",(trainingSet.iloc[res1.index]))
train_label1=dataset_train.iloc[res1.index]
#train_label1.to_csv('E:\\data_mining\\eye_classification\\clustering_model\\result\\egg_train_test_class11.csv',index=False)




'''
dataset_class0=pd.read_csv('E:\\data_mining\\eye_classification\\clustering_model\\result\\egg_train_class0.csv')

#dataset_class0=dataset_class0.iloc[1200:,:]
#print(dataset_class0)
print('类别输出为：\n',dataset_class0['label'].value_counts())

dataset_class00=dataset_class0[-dataset_class0.label.isin([1])]

dataset_class00.to_csv('E:\\data_mining\\eye_classification\\clustering_model\\result\\egg_train_class00.csv',index=False)
'''







#predict_result=kmeans.predict([[0.320347155,0.478602869]])

#print('预测类别为：',predict_result[0])




trainingLabels_list=[]
for i in range(len(trainingLabels)):
    trainingLabels_list.append(trainingLabels.label[i])
trainingLabels=trainingLabels_list

#聚类结果评分:FMI评价分值（需要真实值）
from sklearn.metrics import fowlkes_mallows_score

kmeans=KMeans(n_clusters=2).fit(trainingSet)#构建并训练模型
score=fowlkes_mallows_score(trainingLabels,kmeans.labels_)
print('iris数据集聚类为2类FMI评价分值为：%f' %(score))
 
     


    
'''
iris数据集聚类为2类FMI评价分值为：0.750473
iris数据集聚类为3类FMI评价分值为：0.820808
iris数据集聚类为4类FMI评价分值为：0.753970
iris数据集聚类为5类FMI评价分值为：0.725483
iris数据集聚类为6类FMI评价分值为：0.614345
'''
   


''' 
#聚类结果评分：silhouette_score评分值（不需要真实值对比）,轮廓系数法
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt
silhouetteScore=[]

kmeans=KMeans(n_clusters=2).fit(trainingSet)#构建并训练模型
score=silhouette_score(trainingSet,kmeans.labels_)
silhouetteScore.append(score)
print(silhouetteScore)
'''





'''
适用于实际类别信息未知情况下，对于单个样本，设：
a是与它同类别中其他样本的平均距离，
b是与它距离最近不同类别中样本的平均距离，轮廓系数为：
S=（b-a)/max(a,b)
对于一个样本集合，它的轮廓系数是所有样本轮廓系数的平均值
轮廓系数取值范围是[-1,1],同类别样本越距离相近且不同类别样本距离越远，分数越高
'''



'''
#calinski_harabaz指数（不需要真实值对比）
from sklearn.metrics import calinski_harabaz_score

kmeans=KMeans(n_clusters=2,random_state=123).fit(trainingSet)#构建并训练模型
score=calinski_harabaz_score(trainingSet,kmeans.labels_)
print('iris数据聚类数为2类calinski_harabaz指数为:%f' %(score))
'''






'''
iris数据聚类数为2类calinski_harabaz指数为:513.924546
iris数据聚类数为3类calinski_harabaz指数为:561.627757
iris数据聚类数为4类calinski_harabaz指数为:530.487142
iris数据聚类数为5类calinski_harabaz指数为:495.541488
iris数据聚类数为6类calinski_harabaz指数为:469.836633

得到的Calinski-Harabasz分数值ss越大则聚类效果越好。
也就是说，类别内部数据的协方差越小越好，
类别之间的协方差越大越好，
这样的Calinski-Harabasz分数会高。 
在scikit-learn中， Calinski-Harabasz Index对应的方法是metrics.calinski_harabaz_score. 
在真实的分群label不知道的情况下，可以作为评估模型的一个指标。 
同时，数值越小可以理解为：组间协方差很小，组与组之间界限不明显。 
与轮廓系数的对比，笔者觉得最大的优势：快！相差几百倍！毫秒级
'''   

