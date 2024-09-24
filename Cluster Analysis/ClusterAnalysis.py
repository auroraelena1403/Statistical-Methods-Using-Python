#!/usr/bin/env python
# coding: utf-8

# Importing libraries

# In[9]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()
from sklearn.cluster import KMeans


# Loading the data

# In[14]:


data=pd.read_csv('C:\\Users\\elena\\Downloads\\3.01.+Country+clusters.csv')
data


# Plotting the data

# In[16]:


plt.scatter(data['Longitude'], data['Latitude'])
plt.xlim(-180,180)
plt.ylim(-90,90)
plt.show()


# Selecting the features

# In[19]:


x=data.iloc[:,1:3]
x


# Clustering

# In[32]:


kmeans=KMeans(3)
kmeans.fit(x)


# In[34]:


identified_clusters=kmeans.fit_predict(x)
identified_clusters


# In[36]:


data_with_clusters=data.copy()
data_with_clusters['Cluster']=identified_clusters
data_with_clusters


# In[38]:


plt.scatter(data_with_clusters['Longitude'], data_with_clusters['Latitude'], c=data_with_clusters['Cluster'], cmap='rainbow')
plt.xlim(-180,180)
plt.ylim(-90,90)
plt.show()


# In[ ]:




