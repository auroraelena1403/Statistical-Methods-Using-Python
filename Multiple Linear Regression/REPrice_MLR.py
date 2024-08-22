import pandas as pd
import statsmodels.api as sm

data=pd.read_csv('real_estate_price_size_year.csv')
y=data['price']
x1=data[['size','year']]

x=sm.add_constant(x1)
result=sm.OLS(y,x).fit()
print(result.summary())


#the model fits the data
#all p-values are 0.000 which means the coefficient is significantly different from 0