# -*- coding: utf-8 -*-
"""
Created on Thu Jun 18 09:26:54 2020

@author: wangjingxian
"""

import pandas as pd
import numpy as np
import array
from sklearn import model_selection

import GCForest 
#from GCForest.gcForest import gcforest
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.model_selection import GridSearchCV

dataset_train=pd.read_csv('E:\data_mining\eye_classification\data\data_add_train\eeg_train_add_final3.csv')
#print('训练数据特征：',dataset_train.shape)
dataset_test=pd.read_csv('E:\data_mining\eye_classification\data\data_add_train\eeg_test_diff_label3.csv')
#
X_train=dataset_train.iloc[:,0:14] 
y_train=dataset_train.iloc[:,[14]] 

X_train=np.array(X_train)
y_train=np.array(y_train)

#print(X_train,y_train)
X_test=dataset_test.iloc[:,0:14]
#index=dataset_test.iloc[:,[14]]
X_test=np.array(X_test)

'''
param_test2 = {'shape_1X':range(2,14,1), 'window':range(2,14,2)}
#gsearch2 = GridSearchCV(GCForest.gcForest(n_mgsRFtree=30,param stride=1,cascade_test_size=0.2,n_cascadeRF=2,n_cascadeRFtree=101,min_samples_mgs=0.1,min_samples_cascade=0.1,tolerance=0.0,n_jobs=1),param_grid = param_test2, scoring='roc_auc',iid=False, cv=7)
gsearch2 = GridSearchCV(estimator = GCForest.gcForest(), 
                       param_grid = param_test2, scoring='roc_auc',cv=5)
gsearch2.fit(X_train,y_train)
print(gsearch2.cv_results_, gsearch2.best_params_, gsearch2.best_score_)
#{'max_depth': 21, 'min_samples_split': 2} 0.9176631969050988
'''


#shape_1X:取值1-14
#model = GCForest.gcForest(shape_1X=8,window=8)

model = GCForest.gcForest(shape_1X=8,window=8,n_mgsRFtree=60,n_cascadeRF=4,n_cascadeRFtree=70)


model.fit(X_train, y_train) #fit(X,y) 在输入数据X和相关目标y上训练gcForest;

#print ('袋外样本来评估模型:',model.oob_score_)
y_predprob = model.predict_proba(X_train)[:,1]
print(y_predprob)
print ("AUC Score (Train): %f" % metrics.roc_auc_score(y_train, y_predprob))



'''
#进行交叉验证
scores = model_selection.cross_val_score(
    model,
    X_train,
    y_train,
    cv=7
)
'''
predictions=model.predict(X_test)
predictions_proba=model.predict_proba(X_test)
predictions_proba1=predictions_proba.max(axis=1)

print(model.predict(X_test))
print(model.predict_proba(X_test))

submission = pd.DataFrame({
        'index':dataset_test['index'],
        'probability':predictions_proba1,
        'label':predictions
    })

submission.to_csv('E:\\data_mining\\eye_classification\\result\\GCForest\\GCForest_predict4.csv',index=False)


'''
y_predict = model.predict_proba(X_test) #预测未知样本X的类概率;
#y_predict = y_predict.tolist()

y_predict1 = model.predict(X_test) #预测未知样本X的类别;
predictions_proba1=predictions_proba.max(axis=1)
print('预测的分类结果:\n',y_predict1)
print("---每个样本对应每个类别的概率---")
print(y_predict)
'''

 
    