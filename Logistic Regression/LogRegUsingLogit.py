#!/usr/bin/env python
# coding: utf-8

# Importing the libraries

# In[2]:


import pandas as pd
import numpy as np
import statsmodels.api as sm
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()


# Loading the data

# In[5]:


raw_data=pd.read_csv('C:\\Users\\elena\\Downloads\\2.01.+Admittance.csv')
raw_data


# Mapping the categorical variable

# In[9]:


data=raw_data.copy()
data['Admitted']=data['Admitted'].map({'Yes':1,'No':0})
data


# Declaring the dependent and independent variables

# In[11]:


y=data['Admitted']
x1=data['SAT']


# Regression

# In[14]:


x=sm.add_constant(x1)
reg_log=sm.Logit(y,x)
result_log=reg_log.fit()


# In[18]:


result_log.summary()


# In[ ]:




