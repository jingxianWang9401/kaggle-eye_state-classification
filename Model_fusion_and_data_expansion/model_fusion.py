# -*- coding: utf-8 -*-
"""
Created on Fri Jun 19 21:28:50 2020

@author: wangjingxian
"""


#用于筛选所有模型中标签一致的index
import pandas as pd
import numpy as np


data=pd.read_csv("E:\\data_mining\\eye_classification\\result\\multi_model_fusion\\result2\\2.csv")#数据集读取，csv格式


label_gcforest=data.ix[:,[2]]
label_gcforest=np.array(label_gcforest)
probability_gcforest=data.ix[:,[1]]
probability_gcforest=np.array(probability_gcforest)


label_logistic=data.ix[:,[4]]
label_logistic=np.array(label_logistic)
probability_logistic=data.ix[:,[3]]
probability_logistic=np.array(probability_logistic)


label_mlp=data.ix[:,[6]]
label_mlp=np.array(label_mlp)
probability_mlp=data.ix[:,[5]]
probability_mlp=np.array(probability_mlp)


label_random_forest=data.ix[:,[8]]
label_random_forest=np.array(label_random_forest)
probability_random_forest=data.ix[:,[7]]
probability_random_forest=np.array(probability_mlp)


label_svm=data.ix[:,[10]]
label_svm=np.array(label_svm)
probability_svm=data.ix[:,[9]]
probability_svm=np.array(probability_mlp)



a=[]
b=[]
probability1=[]
probability2=[]
probability3=[]
probability4=[]
probability5=[]


index_deff=[]
index_diff_label=[]
index_diff_prob=[]

for i in range(len(label_gcforest)):
    #print()
    #print(label_gcforest[i])
    if label_gcforest[i]==label_logistic[i]==label_mlp[i]==label_random_forest[i]==label_svm[i]:
        #data=np.array(data)
        #print(data)
       
        a.append(i)
        b.append(label_gcforest[i])
        
        probability1.append(probability_gcforest[i])
        probability2.append(probability_logistic[i])
        probability3.append(probability_mlp[i])
        probability4.append(probability_random_forest[i])
        probability5.append(probability_svm[i])
        
        
    else:
        index_deff.append(i)
        index_diff_label.append(label_random_forest[i])
        index_diff_prob.append(probability_random_forest[i])
        
        

#print(probability1) 
#print(len(index_deff))
       
       
#print(a)
print(len(a))
#print(b)
print(len(b))
b=np.array(b)
probability1=np.array(probability1)
probability2=np.array(probability2)
probability3=np.array(probability3)
probability4=np.array(probability4)
probability5=np.array(probability5)

data1=pd.read_csv("E:\data_mining\eye_classification\data\data_add_train\eeg_test_diff_label1.csv")#数据集读取，csv格式
#c=a[0]
#print(data1.ix[c,:])
data3=[]
prob_weight=[]


AUC1=0.92
AUC2=0.93
AUC3=0.9
AUC4=1
AUC5=0.93

sum_weight=AUC1+AUC2+AUC3+AUC4+AUC5
weight1=AUC1/sum_weight
weight2=AUC2/sum_weight
weight3=AUC3/sum_weight
weight4=AUC4/sum_weight
weight5=AUC5/sum_weight


for i in range(len(a)):
    c=a[i]
    
    #print(data1.ix[c,:])
    data2=data1.ix[c,:]
    data3.append(data2)
    
    
    prob=probability1[i]*weight1+probability2[i]*weight2+probability3[i]*weight3+probability4[i]*weight4+probability5[i]*weight5
    prob_weight.append(prob)




print(data2)
print(data3)
data3=pd.DataFrame(data3)
#data3.to_csv('E:\data_mining\eye_classification\data\eeg_test_same_label.csv',index=False)

data3['label']=b
#data3.drop(['index'],axis=1)
data3.to_csv('E:\data_mining\eye_classification\data\eeg_test_same_label2.csv',index=False)

prob_weight=np.array(prob_weight)
data3['probability']=prob_weight

data3.to_csv('E:\data_mining\eye_classification\data\eeg_test_same_label2_prob2.csv',index=False)


'''
dataset_train=pd.read_csv('E:\data_mining\eye_classification\data\eeg_train.csv')

table1=pd.DataFrame(dataset_train)
table2=data3.drop(['index'],axis=1)
data4=pd.merged(table1,table2)
data4.to_csv('E:\data_mining\eye_classification\result\multi_model_fusion\data_add_train\dataset_train_final.csv',index=False)


reader = csv.DictReader(open('E:\data_mining\eye_classification\data\eeg_train.csv'))
header = reader.fieldnames
with open('source.csv', 'a') as csv_file:
   writer = csv.DictWriter(csv_file, fieldnames=header)
   writer.writerows(reader)
'''






data_label_diff=[]

for i in range(len(index_deff)):
    d=index_deff[i]
    
    #print(data1.ix[c,:])
    data_diff=data1.ix[d,:]
    data_label_diff.append(data_diff)

#print(data2)
#print(data3)
data_label_diff=pd.DataFrame(data_label_diff)

data_label_diff.to_csv('E:\data_mining\eye_classification\data\eeg_test_diff_label2.csv',index=False)

index_diff_label=np.array(index_diff_label)
data_label_diff['label']=index_diff_label

index_diff_prob=np.array(index_diff_prob)
data_label_diff['probability']=index_diff_prob


data_label_diff.to_csv('E:\data_mining\eye_classification\data\eeg_test_diff_label2_prob2.csv',index=False)
