#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd


# In[2]:


dataFrameTrain = pd.read_csv('./data/INS_training.csv')
dataFrameTrain.head()


# In[3]:


dataFrameTest = pd.read_csv('./data/INS_test.csv')
dataFrameTest.head()


# In[4]:


file  = open('./data/featureFileTrain.csv','a+')
for index, row in dataFrameTrain.iterrows():
    label = row['target']
    labelNumber = label[len(label)-1]
    rowID = 'ex'+str(row['id'])
    i=1
    featureValue = str(labelNumber) + ' ' + str(rowID) + '|f '
    for element in row[1:len(row)-1]:
        if element != 0:
            featureValue +=(str(i)+':'+str(element)+' ')
        i+=1
    file.write(featureValue+'\n')

file.close()

file = open('./data/featureFileTest.csv','a+')
for index, row in dataFrameTest.iterrows():
    rowID = 'ex'+str(row['id'])
    featureValue = str(rowID) + '|f '
    i = 1
    for element in row[1:len(row)-1]:
        if element != 0:
            featureValue +=(str(i)+':'+str(element)+' ')
        i+=1
    file.write(featureValue+'\n')
file.close()
    
    

