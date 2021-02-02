
# coding: utf-8

# In[1]:


import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.decomposition import KernelPCA
import math


# In[2]:


get_ipython().run_line_magic('matplotlib', 'inline')


# In[3]:


dataFrame = pd.read_csv('./data/Data_Sheet_9.csv')
features = dataFrame.iloc[:,0:2]
features.head()


# In[4]:


pcaMoonDataSet = PCA()
principalComponentsMoonData = pcaMoonDataSet.fit_transform(features)
print(principalComponentsMoonData)


# In[17]:


plt.title("Moon dataset")
plt.xlabel("Feature1")
plt.ylabel("Feature2")
plt.scatter(features.iloc[0:299,0],features.iloc[0:299,1], color='red')
plt.scatter(features.iloc[300:,0],features.iloc[300:,1], color='blue')
plt.show()

plt.title("Plot on Linear-PCA")
plt.xlabel("PCA1")
plt.ylabel("PCA2")
plt.scatter(principalComponentsMoonData[0:299,0],principalComponentsMoonData[0:299,1], color='red')
plt.scatter(principalComponentsMoonData[300:,0],principalComponentsMoonData[300:,1], color='blue')
plt.show()


# In[19]:


variance = 0.1
gammaForRBFKernel = 1/(2*math.pow(variance,2))
print("Gamma value for RBF Kernel", gammaForRBFKernel)
kernelPCA = KernelPCA(n_components=2,kernel='rbf',gamma=gammaForRBFKernel)
kernelPCAMoonDataSet = kernelPCA.fit_transform(features)
print(kernelPCAMoonDataSet)
plt.title("Plot on Kernel-PCA")
plt.xlabel("PCA1")
plt.ylabel("PCA2")
plt.scatter(kernelPCAMoonDataSet[0:299,0],kernelPCAMoonDataSet[0:299,1], color='red')
plt.scatter(kernelPCAMoonDataSet[300:,0],kernelPCAMoonDataSet[300:,1], color='blue')
plt.show()


# In[21]:


plt.title("Plot on first component Kernel-PCA")
plt.xlabel("PCA1")
plt.scatter(kernelPCAMoonDataSet[0:299,0],np.zeros(299), color='red')
plt.scatter(kernelPCAMoonDataSet[300:,0],np.zeros(300), color='blue')
plt.show()

