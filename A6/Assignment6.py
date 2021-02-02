#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import math
import random
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split


# In[2]:


def readTrainigData():

    return pd.read_csv('./data/Big_DWH_Training.csv')


# In[3]:


def initializeParamteres():
    
    numberOfNeuronsInHiddenLayer = 5
    numberOfHiddenLayers = 2
        
    np.random.seed(37)
   
    # This will create a 5x2 matrix which are 10 weights of input layer to first hidden layer. 
    wHiddenLayer1 = np.random.uniform(low=0, high=1, size=(numberOfNeuronsInHiddenLayer,2))
    
    # This will create a 5x5 matrix which are 25 weights of first hidden layer to seond hidden layer.
    wHiddenLayer2 = np.random.uniform(low=0, high=1, size=(numberOfNeuronsInHiddenLayer,numberOfNeuronsInHiddenLayer))

    # This is the weight of the ouput neuron.
    wOutputLayer = np.random.uniform(low=0, high=1, size=(numberOfNeuronsInHiddenLayer,1))

    # This is the bias for each layer
    bHiddenLayer1 = bHiddenLayer2 = np.random.uniform(low=0,high=1,size=(numberOfNeuronsInHiddenLayer,1))
    bOutputLayer = np.random.uniform(low=0,high=1,size=(1,1))

    # This is is the weighed sum from input layer to first hidden layer.
    weightedSumH1 = np.zeros((len(wHiddenLayer1),1))

    # This is the activation of the first hidden layer
    activationH1 = np.zeros((len(wHiddenLayer1),1))

    # This is is the weighed sum from first hidden layer to the second hidden layer.
    weightedSumH2 = np.zeros((len(wHiddenLayer2),1))

    # This is the activation of the second hidden layer
    activationH2 = np.zeros((len(wHiddenLayer2),1))

    # This the weighted sum from 2nd hidden layer to the output layer.
    weightedSumOp = np.zeros((1,1))
    
    return wHiddenLayer1, wHiddenLayer2, wOutputLayer, weightedSumH1, weightedSumH2, weightedSumOp,             bHiddenLayer1, bHiddenLayer2, activationH1, activationH2, bOutputLayer


# In[4]:


def getTrainingSamples(trainingData):
    
    featureSet = np.zeros((len(trainingData),3))
    
    for index,row in trainingData.iterrows():
        
        featureSet[index][0] = row['Height']
        featureSet[index][1] = row['Weight']
        
        # Here the labels that are originally [1,-1] are coverted  to [1,0]
        if row['Gender'] != 1:
            featureSet[index][2] = 0
        else : 
            featureSet[index][2] = 1
    
    return featureSet


# In[5]:


def getStandardizedFeatureSet(featureSet):
    
    totalHeight = meanCorrectedHeight = 0.0
    totalWeight = meanCorrectedWeight = 0.0
    
    sampleSize = len(featureSet)
    normalizedFeatureSet = np.zeros((len(featureSet),3))
    
    for row in featureSet:
        
        totalHeight += row[0]
        totalWeight += row[1]
    
    hMean = totalHeight/sampleSize
    wMean = totalWeight/sampleSize
    print("Height mean: ",hMean)
    print("Weight mean: ",wMean)
    for rowIndex in range(len(featureSet)):
        
        normalizedFeatureSet[rowIndex][0] += (featureSet[rowIndex][0] - hMean)
        normalizedFeatureSet[rowIndex][1] += (featureSet[rowIndex][1] - wMean)
        normalizedFeatureSet[rowIndex][2] = featureSet[rowIndex][2]
        
        meanCorrectedHeight += math.pow(normalizedFeatureSet[rowIndex][0],2)
        meanCorrectedWeight += math.pow(normalizedFeatureSet[rowIndex][1],2) 
        
    
    sdHeight = math.sqrt((meanCorrectedHeight/sampleSize))
    sdWeight = math.sqrt((meanCorrectedWeight/sampleSize))
    print("Height SD: ",sdHeight)
    print("Weight SD: ",sdWeight)
    for rowIndex in range(len(featureSet)):
    
        normalizedFeatureSet[rowIndex][0] = (normalizedFeatureSet[rowIndex][0] / sdHeight)
        normalizedFeatureSet[rowIndex][1] = (normalizedFeatureSet[rowIndex][1] / sdWeight)
    
    return normalizedFeatureSet


# In[6]:


def splitTrainingData(validationDataPercentage, featureSet):

    trainingSet, validationSet = train_test_split(featureSet, test_size=0.05)            
    #print(validationSet.shape)
    return trainingSet, validationSet 


# In[7]:


def applySigmoidActivation(weightedSum):
    
    activationOutput = np.zeros((len(weightedSum),1))
    
    for index in range(len(weightedSum)):
        
        activationOutput[index][0] = 1/(1+math.exp(-(weightedSum[index][0])))
    
    return activationOutput    


# In[8]:


def computeError(output, label):
    
    # Log loss computation -(y*log(output) + (1 - y)log(1-output))
    
    if label == 0:
        return -(math.log(1 - output))
    else: 
        return -(math.log(output))


# In[9]:


def feedForward(sample, wHiddenLayer1, wHiddenLayer2, wOutputLayer, weightedSumH1, weightedSumH2, 
                weightedSumOp, bHiddenLayer1, bHiddenLayer2, activationH1, activationH2, bOutputLayer):

    heightOfSample = sample[0]
    weightOfSample = sample[1]
    label = sample[2]
    
    for index in range(len(wHiddenLayer1)):
        
        weightedSumH1[index][0] = ((heightOfSample * wHiddenLayer1[index][0]) + 
                                   (weightOfSample * wHiddenLayer1[index][1])
                                  ) + bHiddenLayer1[index][0]
    activationH1 = applySigmoidActivation(weightedSumH1)
    for rowIndex in range(len(wHiddenLayer2)):
        
        tempSum = 0
        for colIndex in range(len(wHiddenLayer2[rowIndex])):  
            
            tempSum += (activationH1[colIndex][0] * wHiddenLayer2[rowIndex][colIndex]) + bHiddenLayer2[rowIndex][0]
        
        weightedSumH2[rowIndex][0] = tempSum
    activationH2 = applySigmoidActivation(weightedSumH2)
    tempSum = 0
    for rowIndex in range(len(activationH2)):
        tempSum += (activationH2[rowIndex][0] * wOutputLayer[rowIndex][0]) + bOutputLayer[0][0]
    weightedSumOp[0][0] = tempSum
    
    outPut = applySigmoidActivation(weightedSumOp)
    error = computeError(outPut[0][0], label)
    
    #return outPut[0][0], error
    return wHiddenLayer1, wHiddenLayer2, wOutputLayer, weightedSumH1, weightedSumH2,             weightedSumOp, activationH1, activationH2,outPut[0][0], error


# In[10]:


def sigmoidDerivative(outputOfPreviousLayer):
    sigmoid = 1/(1+math.exp(-(outputOfPreviousLayer)))
    return ((sigmoid)*(1-sigmoid))


# In[11]:


def backPropogate(ouput, sample, wHiddenLayer1, wHiddenLayer2, wOutputLayer, weightedSumH1, weightedSumH2,                 weightedSumOp, bHiddenLayer1, bHiddenLayer2, activationH1, activationH2, bOutputLayer, LR):
    
    #learningRate = 0.005
    errorDerivative = (ouput - sample[2])/(ouput-(ouput*ouput))
    outputLayerDerivative = []
    totalCalcDerivate = 0.0   

    
    for rowIndex in range(len(wOutputLayer)):

        outputLayerDerivative.append(errorDerivative * sigmoidDerivative(weightedSumOp[0][0]))
        totalCalcDerivate = outputLayerDerivative[rowIndex] * activationH2[rowIndex][0]
        wOutputLayer[rowIndex][0] -= (LR * totalCalcDerivate)
    bOutputLayer[0][0] -= (LR * totalCalcDerivate)

    hiddenLayerOutputDerivative = np.zeros((5, 5))

    for rowIndex in range(len(wHiddenLayer2)):

        deltaForBias = 0
        hiddenLayerOutputDerivative[rowIndex][0] =                             wOutputLayer[rowIndex][0] * sigmoidDerivative(weightedSumH2[rowIndex][0])
        for colIndex in range(len(wHiddenLayer2[rowIndex])):
            totalCalcDerivate = outputLayerDerivative[colIndex] * hiddenLayerOutputDerivative[colIndex][0]
            deltaForBias += totalCalcDerivate
            wHiddenLayer2[colIndex][rowIndex] -= (LR * totalCalcDerivate)

        bHiddenLayer2[rowIndex][0] -= (LR * deltaForBias)

    for rowIndex in range(len(wHiddenLayer1)):
        tempSum = 0.0
        deltaForBias = 0
        tempDerivative = sigmoidDerivative(weightedSumH1[rowIndex][0])
        for colIndex in range(len(wHiddenLayer1[rowIndex])):

            for elementIndex in range(len(hiddenLayerOutputDerivative)):

                tempSum += (hiddenLayerOutputDerivative[elementIndex][0] * wHiddenLayer2[rowIndex][elementIndex])
            totalCalcDerivate = outputLayerDerivative[rowIndex] *  tempSum * tempDerivative * sample[colIndex]
            deltaForBias += totalCalcDerivate
            wHiddenLayer1[rowIndex][colIndex] -= (LR * totalCalcDerivate)
        bHiddenLayer1[rowIndex][0] -= (LR * deltaForBias)

    return wHiddenLayer1, wHiddenLayer2, wOutputLayer, bHiddenLayer1, bHiddenLayer2, bOutputLayer, LR


# In[12]:



def startNNTraining(trainingSet, validationSet, standardizedTestSet):
    
    print("Neural Network Started")
    epochs = 50000
    threshold = 0.5
    prediction = None
    correctResult = None
    validationAccuracy = None
    trainingAccuracy = None
    wHiddenLayer1, wHiddenLayer2, wOutputLayer, weightedSumH1, weightedSumH2,         weightedSumOp, bHiddenLayer1, bHiddenLayer2, activationH1, activationH2, bOutputLayer = initializeParamteres()
    
    fwHiddenLayer1 = wHiddenLayer1
    fwHiddenLayer2 = wHiddenLayer2
    fwOuputLayer = wOutputLayer
    fbHiddenLayer1 = bHiddenLayer1
    fbHiddenLayer2 = bHiddenLayer2
    fbOutputLayer = bOutputLayer
    validationIterationCount = 0
    smallestValidationError = 100
    valList = []
    LRList = []
    for randLearnParam in range(10):
        LR = random.uniform((np.divide(1,epochs)),(np.divide(10,epochs)))
        for iteration in range(epochs):
            randomIndicesForSGD = random.sample(range(0, len(trainingSet)), 50)
            validationIterationCount += 1
            if validationIterationCount == 250:

                correctResult = 0
                validationAccuracy = 0.0
                for sample in validationSet:

                    wHiddenLayer1, wHiddenLayer2, wOutputLayer, weightedSumH1, weightedSumH2,                     weightedSumOp, activationH1, activationH2, output, error =                     feedForward(sample, wHiddenLayer1, wHiddenLayer2, wOutputLayer, weightedSumH1, weightedSumH2, 
                    weightedSumOp, bHiddenLayer1, bHiddenLayer2, activationH1, activationH2, bOutputLayer)

                    if output >= threshold:
                        prediction = 1.0
                    else:
                        prediction = 0.0

                    if sample[2] == prediction:
                        correctResult += 1

                validationAccuracy = (correctResult / len(validationSet))*100

                print("Validation accuracy after 250 iterations is ", validationAccuracy,"%")

                if (error < smallestValidationError):

                    smallestValidationError = error

                    fwHiddenLayer1 = wHiddenLayer1
                    fwHiddenLayer2 = wHiddenLayer2
                    fwOutputLayer = wOutputLayer
                    fbHiddenLayer1 = bHiddenLayer1
                    fbHiddenLayer2 = bHiddenLayer2
                    fbOutputLayer = bOutputLayer

                validationIterationCount = 0
                #print('Smallest Val Error',smallestValidationError)
                #print('LR', LR)

            else:
                correctResult = 0.0
                for randomIndex in randomIndicesForSGD:

                    wHiddenLayer1, wHiddenLayer2, wOutputLayer, weightedSumH1, weightedSumH2,                     weightedSumOp, activationH1, activationH2, output, error =                     feedForward(trainingSet[randomIndex], wHiddenLayer1, wHiddenLayer2, wOutputLayer, weightedSumH1, weightedSumH2,                     weightedSumOp, bHiddenLayer1, bHiddenLayer2, activationH1, activationH2, bOutputLayer)

                    if output >= threshold:
                        prediction = 1.0
                    else:
                        prediction = 0.0

                    if trainingSet[randomIndex][2] == prediction:
                        correctResult += 1

                    wHiddenLayer1, wHiddenLayer2, wOutputLayer, bHiddenLayer1, bHiddenLayer2,                     bOutputLayer, LR = backPropogate(output, trainingSet[randomIndex], wHiddenLayer1, wHiddenLayer2, wOutputLayer,
                                                    weightedSumH1, weightedSumH2, weightedSumOp, bHiddenLayer1, bHiddenLayer2, activationH1, activationH2, bOutputLayer,LR)

                trainingAccuracy = (correctResult/50)*100



            if ((iteration +1) % 100 == 0):
                print("Training Loss after",iteration+1,"of",epochs, "is:",error)
                print("Training accuracy after",iteration+1,"= ",trainingAccuracy,"%")
        valList.append(smallestValidationError)
        LRList.append(LR)
        testModel(standardizedTestSet, fwHiddenLayer1, fwHiddenLayer2, fwOutputLayer, fbHiddenLayer1, fbHiddenLayer2,                 fbOutputLayer)
    return valList, LRList, smallestValidationError
    #return fwHiddenLayer1, fwHiddenLayer2, fwOutputLayer, fbHiddenLayer1, fbHiddenLayer2, \
    #            fbOutputLayer, valList, LRList, smallestValidationError


# In[13]:


def readTestData():
    
    return pd.read_csv('./data/DWH_test.csv')


# In[14]:


def testModel(standardizedTestSet, wHiddenLayer1, wHiddenLayer2, wOutputLayer, bHiddenLayer1, bHiddenLayer2,                 bOutputLayer):
    
    threshold = 0.5
    correctResult = 0
    # This is the activation of the first hidden layer
    activationH1 = np.zeros((len(wHiddenLayer1),1))
    
    # This is the activation of the second hidden layer
    activationH2 = np.zeros((len(wHiddenLayer2),1))
    
    # This the weighted sum from 2nd hidden layer to the output layer.
    weightedSumOp = np.zeros((1,1))
    
    
    # This is is the weighed sum from first hidden layer to the second hidden layer.
    weightedSumH2 = np.zeros((len(wHiddenLayer2),1))
    
    
    # This is is the weighed sum from first hidden layer to the second hidden layer.
    weightedSumH1 = np.zeros((len(wHiddenLayer2),1))
    
    for sample in standardizedTestSet:
        wHiddenLayer1, wHiddenLayer2, wOutputLayer, weightedSumH1, weightedSumH2,         weightedSumOp, activationH1, activationH2, output, error =         feedForward(sample, wHiddenLayer1, wHiddenLayer2, wOutputLayer, weightedSumH1, weightedSumH2, 
        weightedSumOp, bHiddenLayer1, bHiddenLayer2, activationH1, activationH2, bOutputLayer)

        #output, error = feedForward(sample, wHiddenLayer1, wHiddenLayer2, wOutputLayer, weightedSumH1, weightedSumH2, 
        #        weightedSumOp, bHiddenLayer1, bHiddenLayer2, activationH1, activationH2, bOutputLayer)
    
        if output >= threshold:
            prediction = 1.0
        else:
            prediction = 0.0
                
        if sample[2] == prediction:
            correctResult += 1
    
    testAccuracy = (correctResult / len(standardizedTestSet))*100
    print("Test accuracy = ",testAccuracy,"%.")
    


# In[15]:


def main():
    
    print('Stats of Training Data')
    trainingData = readTrainigData()
    featureSet = getTrainingSamples(trainingData)
    standardizedFeatureSet = getStandardizedFeatureSet(featureSet)
    trainingSet, validationSet = splitTrainingData(5, standardizedFeatureSet)
    
    print('Stats of Testing Data')
    testData = readTestData()
    featureTestSet = getTrainingSamples(testData)
    standardizedTestSet = getStandardizedFeatureSet(featureTestSet)
    
    valList, LRList, BestValScore = startNNTraining(trainingSet, validationSet, standardizedTestSet);
    
    #wHiddenLayer1, wHiddenLayer2, wOutputLayer, bHiddenLayer1, bHiddenLayer2, \
    #            bOutputLayer, valList, LRList, BestValScore = startNNTraining(trainingSet, validationSet);

    #testModel(standardizedTestSet, wHiddenLayer1, wHiddenLayer2, wOutputLayer, bHiddenLayer1, bHiddenLayer2, \
                #bOutputLayer)
    print('Val List:',valList)
    print('LR List:',LRList)
    print('Best Validation Score:',BestValScore)
    
    plt.plot(LRList, valList, linestyle='-', marker='o')
    plt.xlabel('Learning Rate')
    plt.ylabel('Validation Error')
    plt.title('Learning Rate vs Validation Error (10 values)')
    plt.show()


# In[16]:


if __name__ == '__main__':
    main()

