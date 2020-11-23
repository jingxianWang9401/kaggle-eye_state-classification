# -*- coding: utf-8 -*-
"""
Created on Mon Jun 22 11:19:29 2020

@author: wangjingxian
"""

#导入相关模块
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
#from sklearn import cross_validation
from sklearn import model_selection
from sklearn import metrics
from sklearn.model_selection import GridSearchCV

#读取csv,获取训练集和测试集
train_data = pd.read_csv('E:\data_mining\eye_classification\co_training\data\eeg_train4.csv')
test_data = pd.read_csv('E:\data_mining\eye_classification\co_training\data\eeg_test4.csv')

trainingSet=train_data.iloc[:,0:6] 
trainingLabels=train_data.iloc[:,[6]] 

testSet=test_data.iloc[:,0:6]



#建立模型
alg = RandomForestClassifier(
    oob_score=True,
    random_state=10,
    n_estimators=390,
    max_depth=29,
    min_samples_split=3,
    min_samples_leaf=1,
    max_features=4   
)


alg.fit(trainingSet,trainingLabels)
print ('袋外样本来评估模型:',alg.oob_score_)
y_predprob = alg.predict_proba(trainingSet)[:,1]
print(y_predprob)
print ("AUC Score (Train): %f" % metrics.roc_auc_score(trainingLabels, y_predprob))









'''
#首先对n_estimators进行网格搜索：
param_test1 = {'n_estimators':range(50,400,10)}
gsearch1 = GridSearchCV(estimator = RandomForestClassifier(min_samples_split=4,
                                  min_samples_leaf=2,max_depth=8,max_features='sqrt' ,random_state=10), 
                       param_grid = param_test1, scoring='roc_auc',cv=5)
gsearch1.fit(trainingSet,trainingLabels)
print(gsearch1.cv_results_, gsearch1.best_params_, gsearch1.best_score_)
#{'n_estimators': 140} 0.8840906048298012
#{'n_estimators': 280} 0.8613945996520108
#{'n_estimators': 390} 0.7526391849318891
# {'n_estimators': 390} 0.836095791984784
#{'n_estimators': 390} 0.7683468150245358
'''



'''
#接着对决策树最大深度max_depth和内部节点再划分所需最小样本数min_samples_split进行网格搜索。
param_test2 = {'max_depth':range(3,30,2), 'min_samples_split':range(2,21,2)}
gsearch2 = GridSearchCV(estimator = RandomForestClassifier(n_estimators= 390, 
                                  min_samples_leaf=2,max_features='sqrt' ,oob_score=True, random_state=10),
                          param_grid = param_test2, scoring='roc_auc',iid=False, cv=5)
gsearch2.fit(trainingSet,trainingLabels)
print(gsearch2.cv_results_, gsearch2.best_params_, gsearch2.best_score_)
#{'max_depth': 21, 'min_samples_split': 2} 0.9176631969050988
#{'max_depth': 19, 'min_samples_split': 2} 0.8852195906439455
#{'max_depth': 17, 'min_samples_split': 2} 0.7795942034691373
#{'max_depth': 29, 'min_samples_split': 2} 0.861894837524823

# {'max_depth': 21, 'min_samples_split': 2} 0.8079919040807939
'''



'''
#对于内部节点再划分所需最小样本数min_samples_split，暂时不能一起定下来，因为这个还和决策树其他的参数存在关联。
#下面再对内部节点再划分所需最小样本数min_samples_split和叶子节点最少样本数min_samples_leaf一起调参。
param_test3 = {'min_samples_split':range(2,10,1), 'min_samples_leaf':range(1,10,1)}
gsearch3 = GridSearchCV(estimator = RandomForestClassifier(n_estimators=  390, max_depth=21,
                                  max_features='sqrt' ,oob_score=True, random_state=10),
   param_grid = param_test3, scoring='roc_auc',iid=False, cv=5)
gsearch3.fit(trainingSet,trainingLabels)
print(gsearch3.cv_results_, gsearch3.best_params_, gsearch3.best_score_)
#{'min_samples_leaf': 1, 'min_samples_split': 3} 0.9221082311421183
#{'min_samples_leaf': 1, 'min_samples_split': 4} 0.8886907564889288
 #{'min_samples_leaf': 1, 'min_samples_split': 3} 0.7805582382262546
 #{'min_samples_leaf': 1, 'min_samples_split': 2} 0.8682671680731735
#{'min_samples_leaf': 1, 'min_samples_split': 3} 0.8148896643215118
'''


'''
param_test4 = {'max_features':range(1,7,1)}
gsearch4 = GridSearchCV(estimator = RandomForestClassifier(n_estimators= 390, max_depth=19, min_samples_split=3,
                                  min_samples_leaf=1 ,oob_score=True, random_state=10),
                        param_grid = param_test4, scoring='roc_auc',iid=False, cv=5)
gsearch4.fit(trainingSet,trainingLabels)
print(gsearch4.cv_results_, gsearch4.best_params_, gsearch4.best_score_)
#{'max_features': 7} 0.9228205650057426
#{'max_features': 2} 0.8885543388933611
#{'max_features': 5} 0.7803572742740608
#{'max_features': 3} 0.8681867786699204
#{'max_features': 4} 0.8146389363500541
'''






#进行交叉验证
scores = model_selection.cross_val_score(
    alg,
    trainingSet,
    #train_data[predictors],
    train_data['label'],
    cv=7
)

print('k折交叉验证的准确率为：',scores.mean())
print(scores.std())

#预测结果输出
def create_submission(alg,trainingSet,trainingLabels,testSet,test_data,filename):
    alg.fit(trainingSet,trainingLabels)
    predictions = alg.predict(testSet)
    #print (alg.predict_proba(testSet))  # 输出为概率值
    predictions_proba=alg.predict_proba(testSet)    
    #print(predictions_proba.shape)
    predictions_proba1=predictions_proba.max(axis=1)
    #print(predictions_proba1)
    submission = pd.DataFrame({
        'index':test_data['index'],
        'probability':predictions_proba1,
        'label':predictions
    })
    submission.to_csv(filename,index=False)

#create_submission(alg,train_data,test_data,predictors,'E:\data_mining\eye_classification\data\predict.csv')
create_submission(alg,trainingSet,trainingLabels,testSet,test_data,'E:\\data_mining\\eye_classification\\co_training\\result\\random_forest_predict4.csv')










