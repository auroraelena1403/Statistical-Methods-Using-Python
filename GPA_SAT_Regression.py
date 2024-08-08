import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.api as sm

data=pd.read_csv("1.01.+Simple+linear+regression.csv")
print(data, data.describe())

#regression : y=b0+b1*x1
#1. Dependent variable
y=data['GPA']
#2. Independent variable
x1=data['SAT']
plt.scatter(x1,y)
plt.xlabel('SAT')
plt.ylabel('GPA')
plt.show()
x=sm.add_constant(x1)
result=sm.OLS(y,x).fit()
print(result.summary())