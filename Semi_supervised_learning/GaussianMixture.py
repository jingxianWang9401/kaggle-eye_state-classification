# -*- coding: utf-8 -*-
"""
Created on Mon Jun 22 09:51:34 2020

@author: wangjingxian
"""

import matplotlib.pyplot as plt
from sklearn.datasets.samples_generator import make_blobs
from sklearn.mixture import GaussianMixture
import pandas as pd
import numpy as np

'''
# X为样本特征，Y为样本簇类别， 共1000个样本，每个样本2个特征，共4个簇，簇中心在[-1,-1], [0,0],[1,1], [2,2]
X, y = make_blobs(n_samples=1000, n_features=2, centers=[[-1,-1], [0,0], [1,1], [2,2]], cluster_std=[0.4, 0.3, 0.4, 0.3], 
                  random_state = 0)
'''

dataset_train=pd.read_csv('E:\data_mining\eye_classification\clustering_model\data\eeg_train.csv')
X=dataset_train.iloc[:,0:14] 
trainingLabels=dataset_train.iloc[:,[14]] 


##设置gmm函数
gmm = GaussianMixture(n_components=2, covariance_type='spherical').fit(X)
##训练数据
y_pred = gmm.predict(X)

print(y_pred)

#predicted_label=gmm.predict([[0.320347155,0.478602869]])
#print('预测标签为：',predicted_label)

#print ("聚类中心\n", (gmm.cluster_centers_))

##每个数据的分类
#lables = gmm.labels_
#print('标签预测为：',lables)

##总共的标签分类
labels_unique = np.unique(y_pred)
##聚簇的个数，即分类的个数
n_clusters_ = len(labels_unique)
print("number of estimated clusters聚类数量为 : %d" % n_clusters_)

#print ("聚类中心\n", (spectral_clustering.cluster_centers_))
quantity = pd.Series(y_pred).value_counts()
print( "聚类后每个类别的样本数量\n", (quantity))

#获取聚类之后每个聚类的数据
resSeries = pd.Series(y_pred)
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

gmm = GaussianMixture(n_components=2, covariance_type='spherical').fit(X)#构建并训练模型
score=fowlkes_mallows_score(trainingLabels,gmm.labels_)
print('iris数据集聚类为2类FMI评价分值为：%f' %(score))
'''