#!/usr/bin/env python
# coding: utf-8

# In[17]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

import scipy.stats as stats

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.model_selection import cross_val_score

from sklearn.preprocessing import KBinsDiscretizer
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier

from sklearn.preprocessing import FunctionTransformer
from sklearn.compose import ColumnTransformer


# In[18]:


date= pd.read_csv('orders.csv')
time=pd.read_csv('messages.csv')


# In[19]:


date.head()


# In[20]:


time.head()


# In[21]:


date.info()


# In[22]:


#converting to datetime datatype
date['date'] = pd.to_datetime(date['date'])


# In[23]:


date.info()


# # 1) EXTRACT YEAR

# In[24]:


date['date_year']=date['date'].dt.year
date.head()


# # 2) EXTRACT MONTH

# In[25]:


date['date_month']=date['date'].dt.month
date.head()


# # 3) EXTRACT MONTH-NAME

# In[26]:


date['date_month-name']=date['date'].dt.month_name()
date.head()


# # 4) EXTRACT DAY

# In[28]:


date['date_day']=date['date'].dt.day
date.head()


# # 5) EXTRACT DAY-OF WEEK-NAME

# In[31]:


date['date_day-name']=date['date'].dt.day_name()
date.head()


# # 6) EXTRACT DAYOFWEEK

# In[32]:


date['date_dayofweek']=date['date'].dt.dayofweek
date.head()


# # 7) CHECK IF WEEKEND

# In[39]:


date['date_is_weekend'] = np.where(date['date_day-name'].isin(['Sunday','Saturday']), 1,0)

date.drop(columns=['product_id','city_id','orders']).head()


# # 8) EXTRACT QUATER

# In[41]:


date['quarter'] = date['date'].dt.quarter

date.drop(columns=['product_id','city_id','orders']).head()


# # 9) EXTRACT SEMESTER

# In[42]:


date['semester'] = np.where(date['quarter'].isin([1,2]), 1,2)

date.drop(columns=['product_id','city_id','orders']).head()


# # EXTRACT TIME BETWEEN TWO DATES

# In[46]:


import datetime

today=datetime.datetime.today()

today


# In[47]:


today-date['date']


# # DAYS PASSED

# In[49]:


(today-date['date']).dt.days


# # MONTHS PASSED

# In[58]:


np.round((today-date['date']) / np.timedelta64(1, 'M'),0)


# In[59]:


time


# In[60]:


time.info()


# In[61]:


#converting to datetime datatype
time['date'] = pd.to_datetime(time['date'])


# In[62]:


time.info()


# In[63]:


time.head()


# # EXTRACT HOURS,MINUTES AND SECONDS

# In[65]:


time['hour']=time['date'].dt.hour
time['min']=time['date'].dt.minute
time['sec']=time['date'].dt.second
time.head()


# # EXTRACT TIME PART

# In[66]:


time['time']=time['date'].dt.time
time.head()


# # TIME DIFFRENCE

# In[68]:


today - time['date']


# # IN SECONDS

# In[69]:


(today - time['date'])/np.timedelta64(1,'s')


# # IN MONTHS

# In[72]:


(today - time['date'])/np.timedelta64(1,'m')


# # IN HOURS

# In[74]:


(today - time['date'])/np.timedelta64(1,'h')


# In[ ]:




