#!/usr/bin/env python
# coding: utf-8

# In[ ]:


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


# In[2]:


df.shape
#num_rows = df.shape[0] #Provide the number of rows in the dataset
#num_cols = df.shape[1] #Provide the number of columns in the dataset


# In[3]:


# A look at the data -- summary statistics
df.describe()


# In[6]:


df.columns


# In[7]:


df.dtypes
df.dtypes.unique()


# In[8]:


cat_df = df.select_dtypes(include=['object','O']) 
cat_df.shape


# In[11]:


set(cat_df)


# In[10]:


Num_df=df.select_dtypes(include=['float64','int'])
Num_df.describe()


# In[11]:


set(Num_df)


# In[12]:


# A look at the data -- data value in feature
status_vals = df.CareerSatisfaction.value_counts()
status_vals 
#Provide a pandas series of the counts for each Professional status


# In[13]:


# The below should be a bar chart of the proportion of individuals in each professional category if your status_vals
# is set up correctly.

(status_vals/df.shape[0]).plot(kind="bar");
plt.title("Career satisfaction rating");


# In[14]:


schema[schema['Column']=='CareerSatisfaction']['Question']


# In[15]:


most_missing_cols=df['CareerSatisfaction'].isnull().mean() > 0.5#Provide a set of columns with more than 75% of the values missing

most_missing_cols


# In[16]:


no_nulls =df['CareerSatisfaction'].isnull().sum() #Provide a set of columns with 0 missing values.
no_nulls


# In[17]:


#---Proportion of individuals in the dataset with career satisfaction reported
prop_sals = 1 - df.isnull()['CareerSatisfaction'].mean()

prop_sals


# In[18]:


# A picture can often tell us more than numbers
df['CareerSatisfaction'].hist()


# In[19]:


df['CareerSatisfaction'].describe().reset_index()


# In[20]:


set(schema[schema['Column']=='CareerSatisfaction']['Question'])


# In[22]:


# Using statistics to answer questions and draw insight
df['CareerSatisfaction'].isnull().mean()


# In[23]:


df.groupby(['FormalEducation','EmploymentStatus','YearsCodedJob','CompanySize']).mean()['CareerSatisfaction'].reset_index()


# In[24]:


# Using statistics to answer questions and draw insight
df.groupby(['EmploymentStatus']).mean()['CareerSatisfaction'].sort_values().dropna().reset_index()


# In[20]:


# Using statistics to answer questions and draw insight
df.groupby(['YearsCodedJob']).mean()['CareerSatisfaction'].sort_values().dropna().reset_index()


# In[25]:


# Using statistics to answer questions and draw insight
df.groupby(['CompanySize']).mean()['CareerSatisfaction'].sort_values().dropna().reset_index()


# In[26]:


df.groupby(['FormalEducation']).mean()['CareerSatisfaction'].sort_values().dropna().reset_index()


# In[27]:


cat_df = df.select_dtypes(include=['object'])
cat_cols_lst = cat_df[['FormalEducation','EmploymentStatus','YearsCodedJob','CompanySize']]
cat_cols_lst


# In[28]:


# Drop rows with missing salary values
df = df.dropna(subset=['CareerSatisfaction'], axis=0)
y = df['CareerSatisfaction']


# In[29]:


#Drop respondent and expected salary columns
df = df.drop(['Respondent', 'CareerSatisfaction'], axis=1)


# In[30]:


df.shape


# In[31]:


num_vars = df.select_dtypes(include=['float', 'int']).columns
for col in num_vars:
        df[col].fillna((df[col].mean()), inplace=True)


# In[32]:


df.shape


# In[33]:


# Dummy the categorical variables
cat_vars = df.select_dtypes(include=['object']).copy().columns
for var in  cat_vars:
   # for each cat add dummy var, drop original column
   df = pd.concat([df.drop(var, axis=1), pd.get_dummies(df[var], prefix=var, prefix_sep='_', drop_first=True)], axis=1)
   
X = df


# In[35]:


df.shape


# In[36]:


#Split into train and test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = .30, random_state=42) 


# In[ ]:


lm_model = LinearRegression(normalize=True) # Instantiate
lm_model.fit(X_train, y_train) #Fit


# In[ ]:


#Predict using your model
y_test_preds = lm_model.predict(X_test)
y_train_preds = lm_model.predict(X_train)

#Score using your model
test_score = r2_score(y_test, y_test_preds)
train_score = r2_score(y_train, y_train_preds)
"The r-squared score for the model using only quantitative variables was {} on {} values.".format(r2_score(y_test, y_test_preds), len(y_test))


# In[ ]:


print("The number of CareerSatisfaction in the original dataframe is " + str(np.sum(df.CareerSatisfaction.notnull()))) 
print("The number of CareerSatisfaction predicted using our model is " + str(len(y_test_preds)))
print("This is bad because we only predicted " + str((len(y_test_preds))/np.sum(df.CareerSatisfaction.notnull())) + " of the salaries in the dataset.")


# In[ ]:




