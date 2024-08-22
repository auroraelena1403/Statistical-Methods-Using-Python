import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import statsmodels.api as sm
import seaborn as sb

data=pd.read_csv("1.02.+Multiple+linear+regression.csv")
print(data)

#multiple regression
y=data['GPA']
x1=data[['SAT','Rand 1,2,3']]

x=sm.add_constant(x1)
result=sm.OLS(y,x).fit()
print(result.summary())

