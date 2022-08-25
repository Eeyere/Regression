# -*- coding: utf-8 -*-
"""
Created on Wed Aug 24 13:27:03 2022

@author: HP
"""

import pandas as pd
import matplotlib.pyplot as pt
import statsmodels.api as sm
import seaborn as sb
sb.set()

#Read the data into a data frame
dt=pd.read_csv('C:\\Users\\HP\\Downloads\\1.02.+Multiple+linear+regression.csv')

#To get the descriptiue statistucs if the data
print(dt.describe())

y=dt['SAT']
x1=dt[['GPA','Rand 1,2,3']]

#x=sm.add_constant(x1)
x = sm.add_constant(x1)
results=sm.OLS(y,x1).fit()

print(results.summary())
