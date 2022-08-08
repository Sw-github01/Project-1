#!/usr/bin/env python
# coding: utf-8

# In[2]:


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


# In[5]:


schema.head()


# In[6]:


df.shape
#num_rows = df.shape[0] #Provide the number of rows in the dataset
#num_cols = df.shape[1] #Provide the number of columns in the dataset


# In[7]:


# What happened
# A look at the data -- summary statistics
df.describe()


# In[8]:


df.columns


# In[9]:


df.dtypes
df.dtypes.unique()


# In[10]:


cat_df = df.select_dtypes(include=['object','O']) 
cat_df.shape


# In[11]:


set(cat_df)


# In[12]:


Num_df=df.select_dtypes(include=['float64','int'])
Num_df.describe()


# In[13]:


set(Num_df)


# In[14]:


# A look at the data --- missing value and non missing value
no_nulls = set(df.columns[df.isnull().sum()==0])#Provide a set of columns with 0 missing values.
no_nulls


# In[15]:


most_missing_cols = set(Num_df.columns[Num_df.isnull().mean() > 0.5])#Provide a set of columns with more than 75% of the values missing

most_missing_cols


# In[13]:


no_nulls = set(Num_df.columns[Num_df.isnull().sum()==0])#Provide a set of columns with 0 missing values.
no_nulls


# In[16]:


set(schema[schema['Column']=='StackOverflowMetaChat']['Question'])
set(schema[schema['Column']=='StackOverflowAdsRelevant']['Question'])
set(schema[schema['Column']=='StackOverflowJobSearch']['Question'])
set(schema[schema['Column']=='StackOverflowCommunity']['Question'])


# In[17]:


# A look at the data -- data value in feature
status_vals = df.StackOverflowAdsRelevant.value_counts()
status_vals 
#Provide a pandas series of the counts for each Professional status


# In[18]:


# The below should be a bar chart of the proportion of individuals in each professional category if your status_vals
# is set up correctly.

(status_vals/df.shape[0]).plot(kind="bar");
plt.title("The ads on Stack Overflow are relevant to me?");


# In[19]:


# Using statistics to answer questions and draw insight 
prop_stacks = 1 - df.isnull()['StackOverflowSatisfaction'].mean()

prop_stacks


# In[20]:


df.groupby(['StackOverflowCommunity']).mean()['StackOverflowSatisfaction'].sort_values().reset_index()


# In[21]:


df.groupby(['StackOverflowMetaChat']).mean()['StackOverflowSatisfaction'].sort_values().reset_index()


# In[22]:


# A look at the data -- data value in feature
status_vals = df.StackOverflowCommunity.value_counts()
status_vals 
#Provide a pandas series of the counts for each Professional status

# The below should be a bar chart of the proportion of individuals in each professional category if your status_vals
# is set up correctly.

(status_vals/df.shape[0]).plot(kind="bar");
plt.title("Person feel like a member of the Stack Overflow community");


# In[23]:


# A look at the data -- data value in feature
count_vals = df.StackOverflowMetaChat.value_counts()#Provide a pandas series of the counts for each Country

# The below should be a bar chart of the proportion of the top 10 countries for the
# individuals in your count_vals if it is set up correctly.

(count_vals[:10]/df.shape[0]).plot(kind="bar");
plt.title("How often person participated in community discussions on meta or in chat");


# In[24]:


# Using statistics to answer questions and draw insight

df.groupby(['StackOverflowAdsRelevant']).mean()['StackOverflowSatisfaction'].sort_values().dropna().reset_index()


# In[25]:


df['StackOverflowSatisfaction'].isnull().mean()


# In[26]:


df.groupby(['StackOverflowCommunity']).mean()['StackOverflowSatisfaction'].sort_values().dropna().reset_index()


# In[27]:


df.groupby(['StackOverflowMetaChat']).mean()['StackOverflowSatisfaction'].sort_values().dropna().reset_index()


# In[28]:


df.groupby(['StackOverflowJobSearch']).mean()['StackOverflowSatisfaction'].sort_values().dropna().reset_index()


# In[29]:


# identify categorical and non categorical features

cat_df = df.select_dtypes(include=['object']) # Subset to a dataframe only holding the categorical columns

# Print how many categorical columns are in the dataframe
cat_df.shape[1]
np.sum(np.sum(cat_df.isnull())/cat_df.shape[0] == 0)
# 50% categorical columns are null
np.sum(np.sum(cat_df.isnull())/cat_df.shape[0] > .5)
# 75% categorical columns are null
np.sum(np.sum(cat_df.isnull())/cat_df.shape[0] > .75)


# In[30]:


df['StackOverflowSatisfaction'].describe().reset_index()


# In[31]:


# Drop rows with missing salary values
df = df.dropna(subset=['StackOverflowSatisfaction'], axis=0)
y = df['StackOverflowSatisfaction']
#Drop respondent and expected salary columns
df = df.drop(['Respondent', 'StackOverflowSatisfaction'], axis=1)


# In[32]:


num_vars = df.select_dtypes(include=['float', 'int']).columns
for col in num_vars:
        df[col].fillna((df[col].mean()), inplace=True)


# In[ ]:


# Dummy the categorical variables
cat_vars = df.select_dtypes(include=['object']).copy().columns
for var in  cat_vars:
    # for each cat add dummy var, drop original column
    df = pd.concat([df.drop(var, axis=1), pd.get_dummies(df[var], prefix=var, prefix_sep='_', drop_first=True)], axis=1)
    
X = df


# In[ ]:


X.shape


# In[ ]:



#Split into train and test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = .30, random_state=42) 

lm_model = LinearRegression(normalize=True) # Instantiate
lm_model.fit(X_train, y_train) #Fit
        


# In[ ]:


#Predict using your model
y_test_preds = lm_model.predict(X_test)
y_train_preds = lm_model.predict(X_train)
 
"The r-squared score for the model using only quantitative variables was {} on {} values.".format(r2_score(y_test, y_test_preds), len(y_test))
#Score using your model
test_score = r2_score(y_test, y_test_preds)
train_score = r2_score(y_train, y_train_preds)
"The r-squared score for the model using only quantitative variables was {} on {} values.".format(r2_score(y_test, y_test_preds), len(y_test))


# In[ ]:


print("The number of salaries in the original dataframe is " + str(np.sum(df.StackOverflowSatisfaction.notnull()))) 
print("The number of salaries predicted using our model is " + str(len(y_test_preds)))
print("This is bad because we only predicted " + str((len(y_test_preds))/np.sum(df.StackOverflowSatisfaction.notnull())) + " of the salaries in the dataset.")


# In[ ]:




