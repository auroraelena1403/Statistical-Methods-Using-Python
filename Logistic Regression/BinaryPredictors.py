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


raw_data=pd.read_csv('C:\\Users\\elena\\Downloads\\2.02.+Binary+predictors.csv')
raw_data


# Mapping the values

# In[10]:


data=raw_data.copy()
data['Admitted']=data['Admitted'].map({'Yes':1,'No':0})
data['Gender']=data['Gender'].map({'Female':1,'Male':0})
data


# Declaring the dependent and independent variables
# 

# In[21]:


y=data['Admitted']
x1=data[['SAT','Gender']]


# Logistic Regression

# In[23]:


x=sm.add_constant(x1)
reg_log=sm.Logit(y,x)
result_log=reg_log.fit()
result_log.summary()


# The odds of a female being admitted: np.exp(genderCoeffValue)

# 
