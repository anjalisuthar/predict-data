#!/usr/bin/env python
# coding: utf-8

# # Prediction using Supervised ML

# # Predicts the percentage of an student based on number of study hours

# In[39]:


#import libraries
import pandas as pd
import numpy as np  
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression


# In[40]:


#read data from url
df = pd.read_csv('https://raw.githubusercontent.com/AdiPersonalWorks/Random/master/student_scores%20-%20student_scores.csv')
df


# In[41]:


df.shape


# In[42]:


#Describes the values of the table
df.describe()


# VISUALIZING THE DATA

# In[43]:


#plot a graph
plt.scatter(df.Hours,df.Scores,color="blue",marker="o")

plt.title('Student score predictor')  
plt.xlabel('Hours Studied')  
plt.ylabel('Score obtained') 
plt.grid(True)
plt.show()


# In[44]:


#dividing data into inputs and outputs
X = df.iloc[:,:1].values
X


# In[45]:


y = df.iloc[:,1].values
y


# In[46]:


# Splitting data into training set and test set


from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=0)


# TRAINING THE LINEAR REGRESSION MODEL

# In[47]:


from sklearn import linear_model
regressor=linear_model.LinearRegression()
regressor.fit(X_train,y_train)


# MODEL EVALUVATION

# In[48]:


# Plotting regression line 
line = regressor.coef_*X+regressor.intercept_
# Plotting the line for the test data
plt.scatter(X, y, c='green')
plt.plot(X, line, c='red')
plt.title('Student score predictor')  
plt.xlabel('Hours Studied')  
plt.ylabel('Score obtained') 
plt.grid(True)
plt.show()


# In[49]:


# Testing data - In Hours
print(X_test) 
# Predicting the scores
y_pred = regressor.predict(X_test) 


# In[50]:


# Comparing Actual vs Predicted Scores
df = pd.DataFrame({'Actual': y_test, 'Predicted': y_pred}) 
display(df)


# In[51]:


#prediction of the data values using predict function
#if a student studies for 9.25 hrs/ day?
Hours=[[9.25]]
y_pred=reg.predict(Hours)
print(y_pred)


# MODEL EVALUATION

# In[52]:


from sklearn import metrics

predictions = lr.predict(X_test)
print('MAE:', metrics.mean_absolute_error(y_test, predictions))


# In[ ]:




