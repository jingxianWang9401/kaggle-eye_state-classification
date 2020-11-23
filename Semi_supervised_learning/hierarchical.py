# -*- coding: utf-8 -*-
"""
Created on Mon Jun 22 09:55:44 2020

@author: wangjingxian
"""

from sklearn.datasets.samples_generator import make_blobs
from sklearn.cluster import AgglomerativeClustering
import numpy as np
import matplotlib.pyplot as plt
from itertools import cycle  ##python自带的迭代器模块
import pandas as pd
from sklearn.preprocessing import MinMaxScaler


dataset_train=pd.read_csv('E:\data_mining\eye_classification\clustering_model\data\eeg_train.csv')
X=dataset_train.iloc[:,0:14] 
trainingLabels=dataset_train.iloc[:,[14]] 


##设置分层聚类函数
linkages = ['ward', 'average', 'complete']
n_clusters_ = 2
ac = AgglomerativeClustering(linkage=linkages[2],n_clusters = n_clusters_)
##训练数据
ac.fit(X)

##每个数据的分类
lables = ac.labels_
print(lables)

##簇中心的点的集合
#cluster_centers = ac.cluster_centers_
#print('cluster_centers:',cluster_centers)

##总共的标签分类
labels_unique = np.unique(lables)

##聚簇的个数，即分类的个数
n_clusters_ = len(labels_unique)
print("number of estimated clusters : %d" % n_clusters_)

#print ("聚类中心\n", (ac.cluster_centers_))

quantity = pd.Series(ac.labels_).value_counts()
print( "聚类后每个类别的样本数量\n", (quantity))

#获取聚类之后每个聚类中心的数据
resSeries = pd.Series(ac.labels_)
res0 = resSeries[resSeries.values == 0]
print("聚类后类别为0的数据\n",(dataset_train.iloc[res0.index]))


res1 = resSeries[resSeries.values == 1]
print("聚类后类别为1的数据\n",(dataset_train.iloc[res1.index]))



dataset_class0=pd.read_csv('E:\\data_mining\\eye_classification\\clustering_model\\result\\egg_train_class0.csv')

#dataset_class0=dataset_class0.iloc[1200:,:]

#print(dataset_class0)

print('类别输出为：\n',dataset_class0['label'].value_counts())











'''
trainingLabels_list=[]
for i in range(len(trainingLabels)):
    trainingLabels_list.append(trainingLabels.label[i])
trainingLabels=trainingLabels_list

#聚类结果评分:FMI评价分值（需要真实值）
from sklearn.metrics import fowlkes_mallows_score

ac = AgglomerativeClustering(linkage=linkages[2],n_clusters = n_clusters_).fit(X)#构建并训练模型
score=fowlkes_mallows_score(trainingLabels,ac.labels_)
print('iris数据集聚类为2类FMI评价分值为：%f' %(score))
'''