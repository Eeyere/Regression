# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""


import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.api as sm
import seaborn as sns
sns.set()


dt=pd.read_csv('C:\\Users\\user\\Downloads\\Reg_text.csv')
plt.scatter(dt['SAT'],dt['GPA'])
plt.xlabel('SAT')
plt.ylabel('GPA')
plt.show()

y = dt['GPA']
x1 = dt ['SAT']

x=dt['SAT']
x = sm.add_constant(x1)


results = sm.OLS(y,x).fit()
results.summary()

plt.scatter(x1,y)
yhat = 0.0017*x1 + 0.275
fig = plt.plot(x1,yhat, lw=4, c='orange', label ='regression line')
plt.xlabel('SAT', fontsize = 20)
plt.ylabel('GPA', fontsize = 20)
plt.show()