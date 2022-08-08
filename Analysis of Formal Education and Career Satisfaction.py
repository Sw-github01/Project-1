#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error
import AllTogether as t
import WhatHappened as t2
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')

df = pd.read_csv('./survey_results_public.csv')
schema = pd.read_csv('./survey_results_schema.csv')
df.head()


# In[2]:


schema.head()


# In[3]:


df.shape
#num_rows = df.shape[0] #Provide the number of rows in the dataset
#num_cols = df.shape[1] #Provide the number of columns in the dataset


# In[4]:


# What happened
# A look at the data -- summary statistics
df.describe()


# In[5]:


df.columns


# In[6]:


df.dtypes
df.dtypes.unique()


# In[10]:


cat_df = df.select_dtypes(include=['object','O']) 
cat_df.describe()
#cat_df.shape()


# In[8]:


set(cat_df)


# In[11]:


Num_df=df.select_dtypes(include=['float64','int'])
Num_df.describe()


# In[13]:


set(Num_df)


# In[12]:


set(schema[schema['Column']=='FormalEducation']['Question'])


# In[14]:


#Provide a set of columns with more than 50% of the values missing
prop_null=df['FormalEducation'].isnull().mean() > 0.5
prop_null


# In[15]:


no_nulls =df['FormalEducation'].isnull().sum() #Provide a set of columns with 0 missing values.
no_nulls


# In[16]:


#---Proportion of individuals in the dataset
prop_null = 1 - df.isnull()['FormalEducation'].mean()

prop_null


# In[17]:


status_vals = df.FormalEducation.value_counts()


# In[22]:


perct_FormEd=df.FormalEducation.value_counts()/df.FormalEducation.shape[0]-df.FormalEducation.isnull().sum()
perct_FormEd


# In[18]:


(status_vals/df.shape[0]).plot(kind="bar");
plt.title("Which of the following best describes the highest level of formal education that you've completed?");


# In[80]:


# Using statistics to answer questions and draw insight --- grouping/ungrouping feature?
higher_ed = lambda x: 'Yes' if x in ("Master's degree", "Doctoral", "Professional degree","Bachelor's degree") else 'No'


# In[81]:


df["FormalEducation"].describe()


# In[82]:


df['HigherEd'] = df["FormalEducation"].apply(higher_ed)
df['HigherEd'].describe().reset_index()

#higher_ed_perc = df['HigherEd'].mean()
#higher_ed_perc


# In[85]:


df['HigherEd'].dtype
df['FormalEducation'].dtype


# In[88]:


formEd=df[df['FormalEducation'].isnull()==False]['CareerSatisfaction']
formEd.describe().reset_index()


# In[89]:


formEd=df[df['HigherEd'].isnull()==False]['CareerSatisfaction']
formEd.describe().reset_index()


# In[94]:


df.groupby(['HigherEd']).sum()['CareerSatisfaction'].sort_values().dropna().reset_index()


# In[29]:


# Using statistics to answer questions and draw insight
df['CareerSatisfaction'].isnull().mean()


# In[92]:


df.groupby(['FormalEducation']).mean()['CareerSatisfaction'].sort_values().dropna().reset_index()


# In[96]:


df.groupby(['HigherEd']).sum()['CareerSatisfaction'].plot(kind="bar");
plt.title("Higher Education Vs Other type Education");


# In[97]:


df.groupby(['FormalEducation']).mean()['CareerSatisfaction'].plot(kind="bar");
plt.title("Which of the following best describes the highest level of formal education that you've completed?");


# In[99]:


df.groupby(['FormalEducation','Gender']).mean()['CareerSatisfaction'].reset_index()


# In[ ]:




