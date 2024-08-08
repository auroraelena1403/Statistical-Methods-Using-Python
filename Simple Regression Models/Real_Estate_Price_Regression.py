#price is influenced by size
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.api as sm

data=pd.read_csv('real_estate_price_size.csv')
print(data, data.describe())

#regression : y=b0+x1*b1
y=data['price']
x1=data['size']
plt.scatter(x1,y)
plt.xlabel('Size')
plt.ylabel('Price')
plt.show()
x=sm.add_constant(x1)
result=sm.OLS(y,x).fit()
print(result.summary(), result)

