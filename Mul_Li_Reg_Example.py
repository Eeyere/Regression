#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os
os.chdir ('C:\\Users\\HP\\Desktop\\Regression_Udemy') #Changing the current working directory


# In[38]:


import pandas as pd

dt=pd.read_csv('C:\\Users\\HP\\Downloads\\1.04.+Real-life+Example.csv')

import pandas as pd
import matplotlib.pyplot as pt
import statsmodels.api as sm
import numpy as np
import seaborn as sb
sb.set()

from sklearn.linear_model import LinearRegression


# Preprocessing: Check the descriptive statistics

# In[3]:


dt.describe(include='all')


# The statistics show values for just 4 columns as above. It shows there are ssues with several other columns, but inlcuding all shows values for all. This was achived by using includes ='all' insode the baraket for descrive method. Since the number of observations for each variables differ it shows there are some missing values. Since registration column shows that a large number of the vehicles are registered, we may want to drop it, and that is achived below. Also, model is affected by engine volume and brand, so its also useless and hence dropped.

# In[4]:


dt2=dt.drop(['Model'], axis=1)
dt2.describe(include='all')


# In[5]:


# Finding missing values 
dt2.isnull() #presents a data frame of missing values showing false where it is 
dt2.isnull().sum() #Presents a summary fo the data frame missing values 


# In[6]:


#The goal is to remove missing values column (axis=0) provided is it less than 5 % of the data, and removing 5% wont affect the result

dt_no_mv=dt2.dropna(axis=0)

dt_no_mv.describe(include='all')


# In[7]:


#One may need to plot the distribution of each variable to check for outliers and so on. We will be looking for  anormal distn
import seaborn as sns
sns.distplot(dt_no_mv['Price'])


# Price has an exponetial distn contrary to what we expect as can be seen. Also, judging by the price descriptive, we can see that 50% of the data has price less that the mean value. Also, comparing the mean and max values is unexpectedly striking. One can also lok at other statistics as well. The outliers are those values around the tail of the distn curve. prices above 150000 which have low frequencies. 
# 
# This problem can easily be solved by removing 1% of the outliers usng the quantile method

# In[8]:


q=dt_no_mv['Price'].quantile(0.99)
dt_new=dt_no_mv[dt_no_mv['Price']<q]
dt_new.describe(include='all')


# In[9]:


sns.distplot(dt_no_mv['Mileage'])


# In[10]:


q2=dt_no_mv['Mileage'].quantile(0.99)
dt_new2=dt_new[dt_new['Mileage']<q2]
sns.distplot(dt_new['Mileage'])


# In[11]:


#Time to lok at engine volume distribution and statistics
#Th eengine volume had max values far greater than what to expect normally (99.9)
sns.distplot(dt_no_mv['EngineV'])


# In[12]:


#Also we may want to remove all data with engine volume greater than 6.5, since naturally, max engine volume is 6.5.
# The data had a lot of cells containing engine volume of 99.9 which is wrong and suspected to be placed based on 
dt_new3=dt_new2[dt_new2['EngineV']<6.5]
sns.distplot(dt_new3['EngineV'])


# In[13]:


#Next is to look at the year:
sns.distplot(dt_new3['Year'])


# In[14]:


#Since the year had values whic are on the left tail side, one may want to eleiminate data produced too long ago
q3=dt_new3['Year'].quantile(0.01)
dt_new4=dt_new3[dt_new3['Year']>q2]
sns.distplot(dt_new4['Year'])


# In[15]:


#Lastly, it is to forget old index and rename data, dat_cleaned

dt_cleaned=dt_new4.reset_index(drop=True)
dt_cleaned.describe(include='all')


# In[16]:


# Here we decided to use some matplotlib code, without explaining it
# You can simply use plt.scatter() for each of them (with your current knowledge)
# But since Price is the 'y' axis of all the plots, it made sense to plot them side-by-side (so we can compare them)

import matplotlib.pyplot as plt
f, (ax1, ax2, ax3) = plt.subplots(1, 3, sharey=True, figsize =(15,3)) #sharey -> share 'Price' as y
ax1.scatter(dt_cleaned['Year'],dt_cleaned['Price'])
ax1.set_title('Price and Year')
ax2.scatter(dt_cleaned['EngineV'],dt_cleaned['Price'])
ax2.set_title('Price and EngineV')
ax3.scatter(dt_cleaned['Mileage'],dt_cleaned['Price'])
ax3.set_title('Price and Mileage')


plt.show()


# In[17]:


#Next one may want to linearised and then check for multi-correlation using the stats model
# Let's transform 'Price' with a log transformation
import numpy as np
log_price = np.log(dt_cleaned['Price']) #calculate log of price

# Then we add it to our data frame
dt_cleaned['log_price'] = log_price #Add log of proce to datat frame

dt2_cleaned = dt_cleaned.drop(['Price'],axis=1) #Delete the Price column

# Display new plots

f, (ax1, ax2, ax3) = plt.subplots(1, 3, sharey=True, figsize =(15,3)) #sharey -> share 'Price' as y
ax1.scatter(dt2_cleaned['Year'],dt2_cleaned['log_price'])
ax1.set_title('log_price and Year')
ax2.scatter(dt2_cleaned['EngineV'],dt_cleaned['log_price'])
ax2.set_title('log_price and EngineV')
ax3.scatter(dt2_cleaned['Mileage'],dt_cleaned['log_price'])
ax3.set_title('log_price and Mileage')


plt.show()


# In[18]:


# sklearn does not have a built-in way to check for multicollinearity
# one of the main reasons is that this is an issue well covered in statistical frameworks and not in ML ones
# surely it is an issue nonetheless, thus we will try to deal with it

# Here's the relevant module
# full documentation: http://www.statsmodels.org/dev/_modules/statsmodels/stats/outliers_influence.html#variance_inflation_factor
from statsmodels.stats.outliers_influence import variance_inflation_factor
data_cleaned=dt2_cleaned
# To make this as easy as possible to use, we declare a variable where we put
# all features where we want to check for multicollinearity
# since our categorical data is not yet preprocessed, we will only take the numerical ones
variables = data_cleaned[['Mileage','Year','EngineV']]

# we create a new data frame which will include all the VIFs
# note that each variable has its own variance inflation factor as this measure is variable specific (not model specific)
vif = pd.DataFrame()

# here we make use of the variance_inflation_factor, which will basically output the respective VIFs 
vif["VIF"] = [variance_inflation_factor(variables.values, i) for i in range(variables.shape[1])]
# Finally, I like to include names so it is easier to explore the result
vif["Features"] = variables.columns

vif


# In[19]:


# Since Year has the highest VIF, I will remove it from the model
# This will drive the VIF of other variables down!!! 
# So even if EngineV seems with a high VIF, too, once 'Year' is gone that will no longer be the case
data_no_multicollinearity = data_cleaned.drop(['Year'],axis=1)


# In[22]:


# To include the categorical data in the regression, let's create dummies
# There is a very convenient method called: 'get_dummies' which does that seemlessly
# It is extremely important that we drop one of the dummies, alternatively we will introduce multicollinearity
data_with_dummies = pd.get_dummies(data_no_multicollinearity, drop_first=True)


# In[23]:


# Here's the result
data_with_dummies.head()


# In[24]:


# To make our data frame more organized, we prefer to place the dependent variable in the beginning of the df
# Since each problem is different, that must be done manually
# We can display all possible features and then choose the desired order
data_with_dummies.columns.values


# In[25]:


# To make the code a bit more parametrized, let's declare a new variable that will contain the preferred order
# If you want a different order, just specify it here
# Conventionally, the most intuitive order is: dependent variable, indepedendent numerical variables, dummies
cols = ['log_price', 'Mileage', 'EngineV', 'Brand_BMW',
       'Brand_Mercedes-Benz', 'Brand_Mitsubishi', 'Brand_Renault',
       'Brand_Toyota', 'Brand_Volkswagen', 'Body_hatch', 'Body_other',
       'Body_sedan', 'Body_vagon', 'Body_van', 'Engine Type_Gas',
       'Engine Type_Other', 'Engine Type_Petrol', 'Registration_yes']


# In[26]:


# To implement the reordering, we will create a new df, which is equal to the old one but with the new order of features
data_preprocessed = data_with_dummies[cols]
data_preprocessed.head()


# In[30]:


targets=data_preprocessed['log_price']
inputs=data_preprocessed.drop(['log_price'],axis=1)

from sklearn.preprocessing import StandardScaler
scaler=StandardScaler()
scaler.fit(inputs)

inputs_scaled=scaler.transform(inputs)


# In[43]:


### Training and testing 
from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(inputs_scaled, targets)

### Training the  model

reg=LinearRegression()
reg.fit(x_train,y_train)

y_hat=reg.predict(x_train) #making predictions with the model

# The simplest way to compare the targets (y_train) and the predictions (y_hat) is to plot them on a scatter plot
# The closer the points to the 45-degree line, the better the prediction
plt.scatter(y_train, y_hat)
# Let's also name the axes
plt.xlabel('Targets (y_train)',size=18)
plt.ylabel('Predictions (y_hat)',size=18)
# Sometimes the plot will have different scales of the x-axis and the y-axis
# This is an issue as we won't be able to interpret the '45-degree line'
# We want the x-axis and the y-axis to be the same
plt.xlim(6,13)
plt.ylim(6,13)
plt.show()


# In[44]:


# Another useful check of our model is a residual plot
# We can plot the PDF of the residuals and check for anomalies
sns.distplot(y_train - y_hat)

# Include a title
plt.title("Residuals PDF", size=18)

# In the best case scenario this plot should be normally distributed
# In our case we notice that there are many negative residuals (far away from the mean)
# Given the definition of the residuals (y_train - y_hat), negative values imply
# that y_hat (predictions) are much higher than y_train (the targets)
# This is food for thought to improve our model


# In[45]:


# Find the R-squared of the model
reg.score(x_train,y_train)

# Note that this is NOT the adjusted R-squared
# in other words... find the Adjusted R-squared to have the appropriate measure :)


# In[46]:


# Obtain the bias (intercept) of the regression
reg.intercept_


# In[47]:


# Obtain the weights (coefficients) of the regression
reg.coef_

# Note that they are barely interpretable if at all


# In[48]:


# Create a regression summary where we can compare them with one-another
reg_summary = pd.DataFrame(inputs.columns.values, columns=['Features'])
reg_summary['Weights'] = reg.coef_
reg_summary


# In[49]:


# Check the different categories in the 'Brand' variable
data_cleaned['Brand'].unique()

# In this way we can see which 'Brand' is actually the benchmark


# In[51]:


# Once we have trained and fine-tuned our model, we can proceed to testing it
# Testing is done on a dataset that the algorithm has never seen
# Luckily we have prepared such a dataset
# Our test inputs are 'x_test', while the outputs: 'y_test' 
# We SHOULD NOT TRAIN THE MODEL ON THEM, we just feed them and find the predictions
# If the predictions are far off, we will know that our model overfitted
y_hat_test = reg.predict(x_test)

# Create a scatter plot with the test targets and the test predictions
# You can include the argument 'alpha' which will introduce opacity to the graph
plt.scatter(y_test, y_hat_test, alpha=0.2)
plt.xlabel('Targets (y_test)',size=18)
plt.ylabel('Predictions (y_hat_test)',size=18)
plt.xlim(6,13)
plt.ylim(6,13)
plt.show()


# In[52]:


# Finally, let's manually check these predictions
# To obtain the actual prices, we take the exponential of the log_price
df_pf = pd.DataFrame(np.exp(y_hat_test), columns=['Prediction'])
df_pf.head()


# In[53]:


# We can also include the test targets in that data frame (so we can manually compare them)
df_pf['Target'] = np.exp(y_test)
df_pf

# Note that we have a lot of missing values
# There is no reason to have ANY missing values, though
# This suggests that something is wrong with the data frame / indexing


# In[54]:


# Let's overwrite the 'Target' column with the appropriate values
# Again, we need the exponential of the test log price
df_pf['Target'] = np.exp(y_test)
df_pf


# In[56]:


# Additionally, we can calculate the difference between the targets and the predictions
# Note that this is actually the residual (we already plotted the residuals)
df_pf['Residual'] = df_pf['Target'] - df_pf['Prediction']

# Since OLS is basically an algorithm which minimizes the total sum of squared errors (residuals),
# this comparison makes a lot of sense

# Finally, it makes sense to see how far off we are from the result percentage-wise
# Here, we take the absolute difference in %, so we can easily order the data frame
df_pf['Difference%'] = np.absolute(df_pf['Residual']/df_pf['Target']*100)
df_pf


# In[57]:


# Exploring the descriptives here gives us additional insights
df_pf.describe()


# In[58]:


# Sometimes it is useful to check these outputs manually
# To see all rows, we use the relevant pandas syntax
pd.options.display.max_rows = 999
# Moreover, to make the dataset clear, we can display the result with only 2 digits after the dot 
pd.set_option('display.float_format', lambda x: '%.2f' % x)
# Finally, we sort by difference in % and manually check the model
df_pf.sort_values(by=['Difference%'])


# In[ ]:




