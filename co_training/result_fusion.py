# -*- coding: utf-8 -*-
"""
Created on Mon Jun 22 14:00:31 2020

@author: wangjingxian
"""

#用于筛选所有模型中标签一致的index
import pandas as pd
import numpy as np


data=pd.read_csv("E:\\data_mining\\eye_classification\\co_training\\result\\result_fusion\\result.csv")#数据集读取，csv格式


label1=data.ix[:,[2]]
label1=np.array(label1)
probability11=data.ix[:,[1]]
probability11=np.array(probability11)
print(probability11[0])

label2=data.ix[:,[4]]
label2=np.array(label2)
probability22=data.ix[:,[3]]
probability22=np.array(probability22)


label3=data.ix[:,[6]]
label3=np.array(label3)
probability33=data.ix[:,[5]]
probability33=np.array(probability33)


label4=data.ix[:,[8]]
label4=np.array(label4)
probability44=data.ix[:,[7]]
probability44=np.array(probability44)


label5=data.ix[:,[10]]
label5=np.array(label5)
probability55=data.ix[:,[9]]
probability55=np.array(probability55)



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

for i in range(len(label1)):
    #print()
    #print(label_gcforest[i])
    if label1[i]==label2[i]==label3[i]==label4[i]==label5[i]:
        #data=np.array(data)
        #print(data)
       
        a.append(i)
        b.append(label1[i])

        probability1.append(probability11[i])
        probability2.append(probability22[i])
        probability3.append(probability33[i])
        probability4.append(probability44[i])
        probability5.append(probability55[i])
        
        
    else:
        index_deff.append(i)
        index_diff_label.append(label1[i])
        index_diff_prob.append(probability11[i])


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

data1=pd.read_csv("E:\data_mining\eye_classification\co_training\data\eeg_test.csv")#数据集读取，csv格式
#c=a[0]
#print(data1.ix[c,:])
data3=[]
prob_weight=[]


AUC1=0.85
AUC2=0.82
AUC3=0.72
AUC4=0.79
AUC5=0.75

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
data3.to_csv('E:\data_mining\eye_classification\co_training\same_label.csv',index=False)

prob_weight=np.array(prob_weight)
data3['probability']=prob_weight

data3.to_csv('E:\data_mining\eye_classification\co_training\same_label_prob.csv',index=False)


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





number=1


data_label_diff=[]
diff_prob_average=[]
for i in range(len(index_deff)):
    number=1
    sum_prob=index_diff_prob[i]
    
    d=index_deff[i]
    
    #print(data1.ix[c,:])
    data_diff=data1.ix[d,:]
    data_label_diff.append(data_diff)

        
        

#print(data2)
#print(data3)
data_label_diff=pd.DataFrame(data_label_diff)

data_label_diff.to_csv('E:\\data_mining\\eye_classification\\test_diff_label.csv',index=False)

index_diff_label=np.array(index_diff_label)
data_label_diff['label']=index_diff_label

index_diff_prob=np.array(index_diff_prob)
data_label_diff['probability']=index_diff_prob


data_label_diff.to_csv('E:\\data_mining\\eye_classification\\test_diff_label_prob.csv',index=False)


diff_prob_average=np.array(diff_prob_average)
data_label_diff['probability_average']=diff_prob_average
data_label_diff.to_csv('E:\\data_mining\\eye_classification\\diff_label_prob_average.csv',index=False)
