# -*- coding: utf-8 -*-
"""
Created on Wed Jun 17 14:05:47 2020

@author: wangjingxian
"""

#相关性矩阵（高：保留其一！）

import pandas as pd
pd.set_option('display.max_columns',1000)
pd.set_option('display.width', 1000)
pd.set_option('display.max_colwidth',1000)

dataset_train=pd.read_csv('E:\data_mining\eye_classification\data\eeg_train.csv')
print('训练数据特征：',dataset_train.shape)
print('类别输出为：\n',dataset_train['label'].value_counts())

dataset_test=pd.read_csv('E:\data_mining\eye_classification\data\eeg_test.csv')
print('测试数据特征：',dataset_test.shape)

data_train=dataset_train.iloc[:,0:14]  
print(data_train.head(5))
#print('归一化前训练数据的相关性关系：\n',data_train.corr())

data_test=dataset_test.iloc[:,0:14]  
print(data_test.head(5))
#print('归一化前测试数据的相关性关系：\n',data_test.corr())

data_train_scale = (data_train-data_train.min())/(data_train.max()-data_train.min())#简单实现标准化
rDf_train=data_train_scale.corr()
#print('归一化后训练数据的相关性关系：\n',rDf_train)

data_test_scale = (data_test-data_test.min())/(data_test.max()-data_test.min())#简单实现标准化
rDf_test=data_test_scale.corr()
#print('归一化后测试数据的相关性关系：\n',rDf_test)


