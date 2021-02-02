#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import random
import math
from bokeh.plotting import figure, show, Figure
from bokeh.models import ColumnDataSource, Label
from bokeh.models.glyphs import Text
from bokeh.palettes import Spectral3, Spectral11
from bokeh.layouts import row, column, gridplot
import numpy as np
from bokeh.io import output_notebook
from bokeh.palettes import Dark2_5 as palette
# itertools handles the cycling
import itertools
output_notebook()
from enum import Enum   


# In[2]:


def getDataFromCSVFile():
    
    return pd.read_csv('./data/DWH_Training.csv')
    


# In[3]:


numberOfIteration = 10000
learningRate = 1/numberOfIteration
dataFrame = getDataFromCSVFile()


# In[4]:


def classify(w, b, dataFrameTraining):
    
    classificationResult = []

    for index, row in dataFrameTraining.iterrows():
        
        result = ((w[0]*row['height(cm)'] + w[1]*row['weight(kg)'])+b)
        if result > 0:
            classificationResult.append(1)
        else:
            classificationResult.append(-1)
    
    return classificationResult
        


# In[5]:


def splitData(index,fold_size,nFolds,data):
    current_index=index*fold_size
    if(index==nFolds-1):
        training_data=data[0:current_index]
        test_data=data[current_index:]
    else:
        training_data=data[0:current_index].append(data[current_index+fold_size:])
        test_data=data[current_index:current_index+fold_size]
    return training_data,test_data
    


# In[6]:


def getFoldSize(data,nFolds):
    len_data=len(data)
    fold_size=len_data//nFolds 
    return fold_size


# In[7]:


def computeGradientAndUpdateHyperParam():
    
    batchSizes = [1,10,50]
    c = [0.1,1.0,10.0]
    
    biasMap = dict()
    cMap = dict()
    accuracy = []
    print(dataFrame.head())
    foldSize=getFoldSize(dataFrame,10)
    for m in range (10):
        for index in range(0,numberOfIteration):

            for cElement in c:

                for bElement in batchSizes:

                    updatedW = [-11.07,-16.61]
                    updatedB = 2976.81
                    w = [-11.07,-16.61]
                    randomDataPointIndicesList = [random.randint(0,len(dataFrame)-1) for number in range(1,bElement)]

                    magnitudeOfW = math.sqrt(math.pow(w[0],2)+math.pow(w[1],2))
                    label = 0
                    for element in randomDataPointIndicesList:

                        if (dataFrame['gender'][element] *
                        ((w[0]*dataFrame['height(cm)'][element] + w[1] * dataFrame['weight(kg)'][element]) + updatedB)) <1 :

                            label += dataFrame['gender'][element]
                            w[0] += label * dataFrame['height(cm)'][element]
                            w[1] += label * dataFrame['weight(kg)'][element]

                    w[0] *= cElement
                    w[1] *= cElement

                    w[0] += magnitudeOfW
                    w[1] += magnitudeOfW

                    w[0] *=learningRate
                    w[1] *=learningRate


                    updatedW[0] -= w[0]
                    updatedW[1] -= w[1]

                    updatedB -= ((-cElement)*(label))
                    biasMap[bElement]=(updatedW,updatedB)
                cMap[cElement]=biasMap

        return cMap


# In[8]:



def performPlot(hyperPlaneMap):  
    palette = ["SpringGreen", "Crimson"]
    dataFrame['color'] = np.where(dataFrame['gender']==1,"SpringGreen", "Crimson")
    dataFrame['legend_values'] = np.where(dataFrame['gender']==1,"Male", "Female")
    #dataFrame['line_color'] = ["SpringGreen", "Crimson", "blue","green","black","brown","red","cyan","yellow"]

#aff a figure
    p=figure(x_axis_label='height(cm)',y_axis_label='weight(kg)', width=800, height=500,
         title='Relation between Male and Female',
        )

#render the graph
    p.circle('height(cm)','weight(kg)',source=dataFrame,size=5,alpha=0.8,color='color',legend='legend_values')
    p.y_range.start = dataFrame['weight(kg)'].min()
    p.y_range.end = dataFrame['weight(kg)'].max()
    p.x_range.start = dataFrame['height(cm)'].min()
    p.x_range.end = dataFrame['height(cm)'].max()

    hyperPlaneValues = []
    xvalues = []
   

    accuracy, cI, bi = crossValidation(dataFrame,10,hyperPlaneMap)
    tupleValues = hyperPlaneMap[cI][bi]  
    weight = tupleValues[0]
    bias = tupleValues[1]
    print(tupleValues)
    print(cI)
    print(bi)
    print(accuracy)
    for index, row in dataFrame.iterrows():
        x = row['height(cm)']
        elements = []
        elements.append((-(weight[0]/weight[1])*x)-(bias/weight[1]))
        hyperPlaneValues.append(elements)
        xvalues.append(row['height(cm)'])
        p.line(x=xvalues , y=hyperPlaneValues, line_width=1, color='blue')
    show(p)
    


# In[9]:


def crossValidation(data,nFolds,hyperPlaneMap):

    maxb = 0
    maxIndexb = 0
    acc = []
    foldSize=getFoldSize(data,nFolds)
    count = 0
    countC = 0
    maxc = 0
    maxcIndex = 0
    maxcbIndex = 0
    for c,b in hyperPlaneMap.items():
        maxb = 0
        maxIndexb = 0
        count = 0
        for tupleInfo in b.values():
            weight = tupleInfo[0]
            bias = tupleInfo[1]
            accuracy = []
            for index in range(nFolds):
                training_data,test_data=splitData(index,foldSize,nFolds,data)
                testResults = classify(weight,bias,test_data)
                correctClassificationCount = 0
                test=len(test_data)
                for j in range(test):
                    test_row=test_data.iloc[j]                  
                    if testResults[j] == int(test_row['gender']):
                        correctClassificationCount +=1
                    else:
                        pass
                acc=correctClassificationCount/test*100
                accuracy.append(acc)            
            avgAcc = np.sum(accuracy)/len(accuracy)
            if  avgAcc > maxb:
                maxb = avgAcc
                maxIndexb = list(b.keys())[0]
        if maxb > maxc:
            maxc = maxb
            maxIndexc = c
            maxcbIndex = maxIndexb
    return maxc, maxIndexc, maxcbIndex


# In[10]:


def main():
    
    hyperPlaneMap = computeGradientAndUpdateHyperParam()
    performPlot(hyperPlaneMap)
    print('\n')
    print("Hyperplane Map",hyperPlaneMap)


# In[11]:


if __name__ == '__main__':
    main()

