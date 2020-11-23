# -*- coding: utf-8 -*-
"""
Created on Thu Jun 18 08:52:25 2020

@author: wangjingxian
"""

import pandas as pd
#from sklearn.linear_model import LogisticRegression, RandomizedLogisticRegression
from sklearn.linear_model import LogisticRegression
#from sklearn.cross_validation import train_test_split
from sklearn.model_selection import train_test_split
import numpy as np
from sklearn import metrics
from sklearn import model_selection

# 导入数据
#读取csv,获取训练集和测试集
train_data = pd.read_csv('E:\data_mining\eye_classification\data\data_add_train\eeg_train_add_final3.csv')
test_data = pd.read_csv('E:\data_mining\eye_classification\data\data_add_train\eeg_test_diff_label3.csv')

trainingSet=train_data.iloc[:,0:14] 
#trainingSet = (trainingSet-trainingSet.min())/(trainingSet.max()-trainingSet.min())#简单实现标准化
trainingLabels=train_data.iloc[:,[14]] 

testSet=test_data.iloc[:,0:14]
#testSet = (testSet-testSet.min())/(testSet.max()-testSet.min())#简单实现标准化

lr = LogisticRegression(penalty='l2',solver='lbfgs')    # 建立LR模型
lr.fit(trainingSet, trainingLabels)    # 用处理好的数据训练模型

#print ('袋外样本来评估模型:',lr.oob_score_)
y_predprob = lr.predict_proba(trainingSet)[:,1]
print(y_predprob)
print ("AUC Score (Train): %f" % metrics.roc_auc_score(trainingLabels, y_predprob))
#print ('逻辑回归的准确率为：{0:.2f}%'.format(lr.score(X_validation, y_validation) *100))


#进行交叉验证
scores = model_selection.cross_val_score(
    lr,
    trainingSet,
    trainingLabels,
    cv=7
)

print('k折交叉验证的准确率为：',scores.mean())
print(scores.std())


predictions=lr.predict(testSet)
predictions_proba=lr.predict_proba(testSet)
predictions_proba1=predictions_proba.max(axis=1)

print(lr.predict(testSet))
print(lr.predict_proba(testSet))


submission = pd.DataFrame({
        'index':test_data['index'],
        'probability':predictions_proba1,
        'label':predictions
    })

submission.to_csv('E:\\data_mining\\eye_classification\\result\\LR\\LogisticRegression_predict4.csv',index=False)





