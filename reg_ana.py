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


# Introducing dummy variable to a file 
raw_data=pd.read_csv('C:\\Users\\HP\\Downloads\\1.03.+Dummies.csv')
new_data=raw_data.copy()

new_data['Attendance']=new_data['Attendance'].map({'Yes':1,'No':0})

raw_data.describe()

## Regression
Y1=new_data['SAT']
X1=new_data[["GPA","Attendance"]]

X11=sm.add_constant(X1)
results2=sm.OLS(Y1,X11).fit()

results2.summary()

pred_data=pd.DataFrame({'count':1, 'SAT':[1700,1670], 'Attendance':[0,1]})

p_data=pred_data.rename(index={0:'Bob',1:'Alice'})

predictions=results2.predict(pred_data)
pred_dataframe=pd.DataFrame({'predictions':predictions})

Pred_res_df=pred_data.join(pred_dataframe)

Predicted_table=Pred_res_df.rename(index={0:'bob',1:'Alice'})
