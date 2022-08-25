# -*- coding: utf-8 -*-
"""
Created on Thu Aug 25 13:04:02 2022y
@author: HP
"""

import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.api as sm
import seaborn as sns
sns.set() # use to set all matplotlib to seaborn disply setting

from sklearn.linear_model import LinearRegression

data=pd.read_csv('C:\\Users\\HP\\Downloads\\1.01.+Simple+linear+regression.csv')
data.head()

x=data['SAT']
y=data['GPA']

x.shape
y.shape #helps know the shape of the dataframe

x_matrix=x.values.reshape(84,1) #Reshaping data when the shape is known
y_matrix=y.values.reshape(-1,1) #smart style of reshaping data

#Reshaping is necessary only for linear reression of one input

reg=LinearRegression()

reg.fit(x_matrix,y_matrix)
reg.score(x_matrix,y) ##Used to get the regression R-square value

reg.coef_ #used to get the coefficients of the regressionreg.coef_

reg.intercept_ #used to get the intercept

reg.predict([[5]]) #used to predict

n_dt=pd.DataFrame({'SAT':[1,2,3,4,5]}) #creating new dataframe to text
n_dt2=pd.DataFrame(data=[3,4,5,33,5,3],columns=['SAT']) #Another way of creating data


plt.scatter(x,y)
yhat = reg.coef_*x_matrix + reg.intercept_
fig = plt.plot(x,yhat, lw=4, c='orange', label ='regression line')
plt.xlabel('SAT', fontsize = 20)
plt.ylabel('GPA', fontsize = 20)
plt.show()
