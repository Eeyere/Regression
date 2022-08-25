# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""


import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.api as sm
import seaborn as sns
sns.set() # use to set all matplotlib to seaborn disply setting


dt=pd.read_csv('C:\\Users\\HP\\Downloads\\real_estate_price_size.csv')
plt.scatter(dt['price'],dt['size'])
plt.xlabel('price')
plt.ylabel('size')
plt.show()

y = dt['price']
x1 = dt ['size']

x=dt['price']
x = sm.add_constant(x1)


results = sm.OLS(y,x).fit()
results.summary()

plt.scatter(x1,y)
yhat = 233.1787*x1 + 1.019e5
fig = plt.plot(x1,yhat, lw=4, c='orange', label ='regression line')
plt.xlabel('price', fontsize = 20)
plt.ylabel('size', fontsize = 20)
plt.show()