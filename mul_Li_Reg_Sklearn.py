# -*- coding: utf-8 -*-
"""
Created on Thu Aug 25 14:17:47 2022

@author: HP
"""

import pandas as pd
import matplotlib.pyplot as pt
import statsmodels.api as sm
import numpy as np
import seaborn as sb
sb.set()

from sklearn.linear_model import LinearRegression



#Read the data into a data frame
dt=pd.read_csv('C:\\Users\\HP\\Downloads\\1.02.+Multiple+linear+regression.csv')

#To get the descriptiue statistucs if the data
dt.describe()

y=dt['SAT']
x=dt[['GPA','Rand 1,2,3']]

reg=LinearRegression()
reg.fit(x,y)

reg.coef_
reg.intercept_

reg.score(x,y) #Provided the R-square value
p='number of predictors'
n='number of observations'

R_sq=reg.score(x,y)
n=x.shape[0]
p=x.shape[1]
R_sq_adjusted= 1-(1-R_sq)*(n-1)/(n-p-1)
# Since R-sq-adjusted is lower than R-sq, it means that one or more of the...
# input variable does not affect predicted outcome

""" Feature selection using F statistic to tell which variables an be disregarded"""

from sklearn.feature_selection import f_regression
f_regression(x,y)

#Returns the f-statistics and the corresponding p-values

f_statistics = f_regression(x,y)[0]
p_values=f_regression(x,y)[1]
p_values.round(4) # round up to 4 deciumal places

#Preparing table summary

Reg_summary=pd.DataFrame(data=x.columns.values, columns=['Features'])
Reg_summary['Coefficients']=reg.coef_
Reg_summary['P_values']=p_values.round(4)

#Standardizing variabbles
from sklearn.preprocessing import StandardScaler

sc=StandardScaler()

sc.fit(x)

x_scaled=sc.transform(x)

#Fitting the model with scaled variables
reg.fit(x_scaled,y)

#Determining the coefficient and intercepts of the scaled variables
new_scaled_coef=reg.coef_
new_scaled_intercept=reg.intercept_

#Preparing the summary table
reg_data_summary=pd.DataFrame([['intercept'],['SAT'],['Random 1,2,3']], columns=['Features'])
reg_data_summary['weights']=new_scaled_intercept,new_scaled_coef[0],new_scaled_coef[1]


#Predicitng with standardised model
test_data=pd.DataFrame(data=[[2.700,2],[2.400,1]],columns=['SAT','RANDOM 1,2,3'])
test_data_scaled=sc.transform(test_data) #data hsa to be transformed before used for testing

test_predict=reg.predict(test_data_scaled)

#Training and testing splitted data
from sklearn.model_selection import train_test_split

aa=np.arange(1,101)

bb=np.arange(501,601)
#splitting an array into randomised parts of defaults 75:25 proportions
aa_train, aa_test = train_test_split(aa) #Default splitted into 75:25

aa_train1, aa_test1 = train_test_split(aa,test_size=0.2) #splits into 80:20 ratios
aa_train2, aa_test2 = train_test_split(aa,test_size=0.2,shuffle=False) #turn off randomising the split outcome

aa_train1, aa_test1 = train_test_split(aa,random_state=42) #splits the data with consistency of randomness
#Consistency of randomness can help use a consistent random split sample to assess a model 

#It is possible to add moore data to be splitted, like splitting aa and bb at the same time
aa_train1, aa_test1, bb_train, bb_test = train_test_split(aa,bb, test_size=0.2) #splits into 80:20 ratios

