#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# *import DattaFrame*


# In[2]:


import pandas as pd 
file = pd.read_csv('A:/Data Sets/weatherAUS.csv')
print(file.head())


# In[3]:


file.head(10)


# In[4]:


file.shape


# In[11]:


file[file['Humidity3pm'].isna()]['Evaporation']


# In[12]:


#  60843 values in Evaporation are NaN
# 67816 values in Sunshine are NaN
# 57094 values are NaN in Cloud3pm
# Cloud9am has 53657 NaN values
# NaN values in other columns are much less
# Date and Location are of no use either
# RISK_MM not to be used ...because DataSet maker said that it can leak information


# In[13]:


# Drop columns listed above (columns with large amounts of Nan Values)


# In[14]:


file2 = file.drop(columns=['Sunshine','Evaporation','Cloud3pm','Cloud9am','Location','RISK_MM','Date'],axis=1)


# In[15]:


file2.shape


# In[16]:


# Now remove rows which have NaN values.


# In[18]:


file3 = file2.dropna(how='any')


# In[19]:


file3.shape


# In[20]:


# file3 now has no NaN values


# In[23]:


file3['RainToday']


# In[24]:


file3['RainTomorrow']


# In[26]:


# RainToday and RainTomorrow are in Yes/No Format


# In[33]:


# Yes ====> 1
# NO  ====> 0


# In[34]:


file3['RainToday'].replace({'No': 0, 'Yes': 1},inplace = True)
file3['RainTomorrow'].replace({'No': 0, 'Yes': 1},inplace = True)


# In[35]:


file3['RainToday'].value_counts()


# In[38]:


file3[['RainToday','RainTomorrow']]


# In[41]:


# Start working with non_numerical columns


# In[39]:


text = ['WindGustDir', 'WindDir3pm', 'WindDir9am']


# In[40]:


text


# In[45]:


import numpy as np


# In[48]:


for col in text:  
    print(np.unique(file3[col]))

file4 = pd.get_dummies(file3, columns=text)


# In[49]:


file4.shape


# In[51]:


print(file4.head(5))


# In[53]:


# Making Features and Labels

# X ====> Feature
# Y ====> Label


# In[56]:


X = file4.loc[:,file4.columns!='RainTomorrow']
Y = file4[['RainTomorrow']]


# In[57]:


print(X.shape)
print(Y.shape)


# In[58]:


# Splitting features and labels for training and testing


# In[60]:


from sklearn.model_selection import train_test_split
xtrain,xtest,ytrain,ytest = train_test_split(X,Y,test_size=0.1)


# In[63]:


print(xtrain.shape)
print(xtest.shape)
print(ytrain.shape)
print(ytest.shape)


# In[69]:


# Using Random Forest


# In[70]:


from sklearn.ensemble import RandomForestClassifier
rain  = RandomForestClassifier(n_estimators=125, max_depth=8)
rain.fit(xtrain, ytrain)


# In[71]:


# Training Accuracy
rain.score(xtrain, ytrain)


# In[72]:


# Test Accuracy
rain.score(xtest, ytest)


# In[73]:


# Saving model


# In[78]:





# In[ ]:




