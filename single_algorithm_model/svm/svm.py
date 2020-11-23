# -*- coding: utf-8 -*-
"""
Created on Wed Jun 17 21:19:24 2020

@author: wangjingxian
"""

import numpy as np
from sklearn import svm
import pandas as pd
from sklearn.model_selection import cross_val_score
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn import model_selection
from sklearn.model_selection import GridSearchCV
from sklearn import metrics



pd.set_option('display.max_columns',1000)
pd.set_option('display.width', 1000)
pd.set_option('display.max_colwidth',1000)

dataset_train=pd.read_csv('E:\data_mining\eye_classification\data\data_add_train\eeg_train_add_final3.csv')
#print('训练数据特征：',dataset_train.shape)
dataset_test=pd.read_csv('E:\data_mining\eye_classification\data\data_add_train\eeg_test_diff_label3.csv')
#
data_train=dataset_train.iloc[:,0:14] 
train_label=dataset_train.iloc[:,[14]] 
#train_label = column_or_1d(train_label, warn=True)
train_label=np.array(train_label)
#print(train_label)
#print(data_train.head(5))
#print('归一化前训练数据的相关性关系：\n',data_train.corr())

data_test=dataset_test.iloc[:,0:14]
index=dataset_test.iloc[:,[14]]
#print(index)


'''
#kernel='rbf'
#用两层循环进行调参,首先对于SVM来说，惩罚系数C是很重要的参数，肯定要选择；其次这里的核函数选择的是RBF，因此另一个参数选择Gamma。
for C in range(1,20,1):
    for gamma in range(1,20,1):
    	#参数scoring设置为roc_auc返回的是AUC，cv=5采用的是5折交叉验证
        model=svm.SVC(C=C,kernel='rbf',gamma=gamma/10,probability = True)
        #model=svm.SVC(C=C,kernel='rbf',gamma=1/gamma,probability = True)
        auc = cross_val_score(model,data_train,train_label.ravel(),cv=7,scoring='roc_auc').mean()
        #x.append(C)
        #y.append(gamma/10)
        #z.append(auc)
        print('参数一：',C,'参数二：',1/gamma,'结果性能：',auc)
        #print(gamma/10)
        #print(auc)
'''

'''
#kernel='linear'
for C in range(1,20,1):
    #参数scoring设置为roc_auc返回的是AUC，cv=5采用的是5折交叉验证
    model=svm.SVC(C=C,kernel='linear',probability = True)
    #model=svm.SVC(C=C,kernel='rbf',gamma=1/gamma,probability = True)
    auc = cross_val_score(model,data_train,train_label.ravel(),cv=7,scoring='roc_auc').mean()
    #x.append(C)
    #y.append(gamma/10)
    #z.append(auc)
    print('参数：',C,'结果性能：',auc)
    #print(gamma/10)
    #print(auc)
'''    
    
'''
#网格化搜索最有参数
#kernel='sigmoid'
param_test3 = {'gamma':range(1,15,1), 'coef0':range(-5,5,1)}
gsearch3 = GridSearchCV(estimator = svm.SVC(C=5,kernel='sigmoid',probability = True),
   param_grid = param_test3, scoring='roc_auc',iid=False, cv=7)
gsearch3.fit(data_train,train_label)
print(gsearch3.cv_results_, gsearch3.best_params_, gsearch3.best_score_)
'''


'''
#网格化搜索最优参数
#kernel='poly'
param_test3 = {'degree':range(1,5,1)}
gsearch3 = GridSearchCV(estimator = svm.SVC(C=5,kernel='poly',gamma=0.1,coef0=0.0,probability = True),
   param_grid = param_test3, scoring='roc_auc',iid=False, cv=7)
gsearch3.fit(data_train,train_label)
print(gsearch3.cv_results_, gsearch3.best_params_, gsearch3.best_score_)
'''

#模型
#model=svm.SVC(C=5,kernel='rbf',gamma=0.1,probability = True)

#model=svm.SVC(C=5,kernel='linear',probability = True)

#model=svm.SVC(C=5,kernel='sigmoid',gamma=0.1,coef0=0,probability = True)

model=svm.SVC(C=5,kernel='poly',gamma=0.1,coef0=0,degree=1,probability = True)

model.fit(data_train, train_label)


#print ('袋外样本来评估模型:',model.oob_score_)
y_predprob = model.predict_proba(data_train)[:,1]
print(y_predprob)
print ("AUC Score (Train): %f" % metrics.roc_auc_score(train_label, y_predprob))






#进行交叉验证
scores = model_selection.cross_val_score(
    model,
    data_train,
    train_label,
    cv=7
)

print('k折交叉验证的准确率为：',scores.mean())

predictions=model.predict(data_test)
decision=model.decision_function(data_test)
predictions_proba=model.predict_proba(data_test)
predictions_proba1=predictions_proba.max(axis=1)


print(model.predict(data_test))
print(model.predict_proba(data_test))


submission = pd.DataFrame({
        'index':dataset_test['index'],
        'probability':predictions_proba1,
        'label':predictions
    })
submission.to_csv('E:\\data_mining\\eye_classification\\result\\SVM\\svm_poly_predict4.csv',index=False)
