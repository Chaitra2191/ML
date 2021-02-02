import pandas as pd
import math
import numpy as np
from sklearn.svm import SVC
import statistics as stats
import matplotlib.pyplot as plt


# Module for training data set using RBF Kernel
def svmTrain(sigma,trainingData,trainingLabels):
    g = math.pow(2 * sigma, 1/2)
    clf = SVC(kernel='rbf',gamma=g)
    validationData,validationOutput,newTrainingData,newTrainingLabels = crosVailidation(trainingData,trainingLabels,10)
    #print(newTrainingData)
    clf.fit(newTrainingData,newTrainingLabels)
    return svmValidate(clf,validationData,validationOutput),clf

# Module for normalizing training input values between 0 and 1
def normalizeDataSet(trainData):
    for i in range(0,len(trainData[0])-1):
        min1 = np.min(trainData[:,i])
        trainData[:,i] = trainData[:,i] - min1
        diff = np.max(trainData[:,i]) - np.min(trainData[:,i])
        trainData[:,i] = trainData[:,i] / diff
    return trainData
    

# Module for loading the data set 
def prepeareData(dataFrame):
    X = np.zeros((len(dataFrame),2))
    Y = []
    X[:,0] = dataFrame['height'].tolist()
    X[:,1] = dataFrame['weight'].tolist()
    Y = dataFrame['gender'].tolist()
    Y[Y==-1]=0
    return X,Y

# Module for validation using the best training weights 
def svmValidate(clf,validationData,validationLabel):
    labels = []
    labels = clf.predict(validationData)
    print(labels)
    count = 0
    for i in range(0,len(labels)-1):
        if validationLabel[i] == labels[i]:
            count = count + 1
        else:
            continue
    return count/len(validationLabel)

# Module for measuring training accuracy using cross validation
def crosVailidation(trainingData,trainingLabels,numFolds):
    foldSize = len(trainingData)//numFolds
    validationData = np.zeros((foldSize,2))
    validationOutput = []
    newTrainingData = trainingData.copy()
    newTrainingLabels = trainingLabels.copy()
    for i in range(0,foldSize - 1):
        foldIndex = np.random.randint(0,len(newTrainingData))
        validationData[i,:] = trainingData[foldIndex,:]
        validationOutput.append(trainingLabels[foldIndex])
        newTrainingData = np.delete(newTrainingData,foldIndex,axis=0)
        newTrainingLabels.remove(trainingLabels[foldIndex])
    return validationData,validationOutput,newTrainingData,newTrainingLabels

def readDataFromCSV(fileName):
    return pd.read_csv(fileName)

# Main module: Entry Point of the code.
def main():
    trainFrame = readDataFromCSV('EX7data1.csv')
    trainingData,trainingLabels = prepeareData(trainFrame)
    trainingData = normalizeDataSet(trainingData)
    validationFrame = readDataFromCSV('EX7data2.csv')
    validationData,validationLabels = prepeareData(validationFrame)
    validationData = normalizeDataSet(validationData)
    
    trainAccuracy = []
    testAccuracy  = []
    sigmaList = []
    trainAccuracyS = []
    testAccuracyS  = []
    sigmaListS = []

    for i in range(50):
        sigma = np.random.randint(math.pow(10,-6),math.pow(10,6))
        sigmaList.append(sigma)
        ta,clf = svmTrain(sigma, trainingData,trainingLabels)
        trainAccuracy.append(ta)
        testAccuracy.append(svmValidate(clf, validationData,validationLabels))

    for idx in np.argsort(sigmaList):
        sigmaListS.append(math.log(sigmaList[idx]))
        trainAccuracyS.append(trainAccuracy[idx])
        testAccuracyS.append(testAccuracy[idx])
    plt.plot(sigmaListS,trainAccuracyS,'red')
    plt.plot(sigmaListS,testAccuracyS,'blue')
    plt.show()
if __name__ == '__main__':
    main()



