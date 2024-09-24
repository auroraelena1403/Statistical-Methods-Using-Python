#!/usr/bin/env python
# coding: utf-8

# Import librraies

# In[7]:


import pandas as pd
import numpy as np
import statsmodels.api as sm
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()


# Loading the data

# In[11]:


raw_data=pd.read_csv('C:\\Users\\elena\\Downloads\\2.01.+Admittance.csv')
raw_data


# Converting the 'Admitted' variable

# In[15]:


data=raw_data.copy()
data['Admitted']=data['Admitted'].map({'Yes':1,'No':0})
data


# Variables

# In[22]:


y=data['Admitted']
x1=data['SAT']


# Plotting the data

# SCATTER plot

# In[32]:


plt.scatter(x1,y, color='C0')
plt.xlabel('SAT')
plt.ylabel('Admitted')
plt.show()


# Categorical variables don`t go well with scatter plots

# 
# 
# Plot using LINEAR REGRESSION

# In[48]:


x=sm.add_constant(x1)
reg_lin=sm.OLS(y,x)
result_lin=reg_lin.fit()
plt.scatter(x1,y)
y_hat=x1*result_lin.params[1]+result_lin.params[0]

plt.plot(x1,y_hat, color='C8')
plt.xlabel('SAT')
plt.ylabel('Admitted')
plt.show()


# Plot using LOGISTIC REGRESSION

# In[59]:


reg_log=sm.Logit(y,x)
result_log=reg_log.fit()
def f(x,b0,b1):
    return np.array(np.exp(b0+x*b1)/(1+np.exp(b0+x*b1)))
    
f_sorted=np.sort(f(x1,result_log.params[0],result_log.params[1]))
x_sorted=np.sort(np.array(x1))

plt.scatter(x1,y)
plt.xlabel('SAT')
plt.ylabel('Admitted')
plt.plot(x_sorted,f_sorted)
plt.show()


# Interpretation: the function shows the probability of admission, givel an SAT score:
#     -when the SAT score is low, the probability of getting admitted is 0
#     -when the SAT score is high, the probability of gettint admitted is 1 or 100%
#     -when in between, the probability is uncertain
#     -when the SAT score is 1700, the probability is around 0.8
# The curve is named 'Logistic regression curve'

# In[ ]:




