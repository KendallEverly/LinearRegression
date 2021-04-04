#!/usr/bin/env python
# coding: utf-8

# In[2]:


import pandas as pd 
data=pd.read_csv("C:/Users/kenda/Downloads/sample.csv")
data.head(5)


# In[3]:


from sklearn.linear_model import LinearRegression


# In[12]:


x=data[["int_memory", "mobile_wt", "ram", "talk_time"]]
y=data[["battery_power"]]
print(x)
print(y)


# In[13]:


from sklearn import linear_model


# In[14]:


regr=linear_model.LinearRegression()
regr.fit(x,y)


# In[15]:


print(regr.coef_)


# In[16]:


print(regr.intercept_)


# In[17]:


predicted_batt=regr.predict([[20,150,2000,15]])
print(predicted_batt)


# In[ ]:




