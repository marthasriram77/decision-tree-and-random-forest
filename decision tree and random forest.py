#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import seaborn as sb


# In[3]:


# Read the dataset into a dataframe
df = pd.read_csv(r"Downloads\titanic.csv", sep='\t', engine='python')


# In[4]:


# Drop some columns which is not relevant to the analysis (they are not numeric)
cols_to_drop = ['Name', 'Ticket', 'Cabin']
df = df.drop(cols_to_drop, axis=1)


# In[5]:


df.info()
sb.heatmap(df.isnull())


# In[6]:


# To replace missing values with interpolated values, for example Age
df['Age'] = df['Age'].interpolate()


# In[7]:


# Drop all rows with missin data
df = df.dropna()
 


# In[8]:


# First, create dummy columns from the Embarked and Sex columns
EmbarkedColumnDummy = pd.get_dummies(df['Embarked'])
SexColumnDummy = pd.get_dummies(df['Sex'])


# In[9]:


df = pd.concat((df, EmbarkedColumnDummy, SexColumnDummy), axis=1)


# In[10]:


# Drop the redundant columns thus converted
df = df.drop(['Sex','Embarked'],axis=1)


# In[11]:


# Seperate the dataframe into X and y data
X = df.values
y = df['Survived'].values

# Delete the Survived column from X
X = np.delete(X,1,axis=1)


# In[12]:


# Split the dataset into 70% Training and 30% Test
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.3,random_state=0)


# In[13]:


# Using simple Decision Tree classifier
from sklearn import tree
dt_clf = tree.DecisionTreeClassifier(max_depth=5)
dt_clf.fit(X_train, y_train)
dt_clf.score(X_test, y_test)


# In[14]:


y_pred = dt_clf.predict(X_test)

from sklearn.metrics import confusion_matrix
confusion_matrix(y_test, y_pred)


# In[15]:


#Perform Classification Using Random Forest Classifier
from sklearn import ensemble
rf_clf = ensemble.RandomForestClassifier(n_estimators=100)
rf_clf.fit(X_train, y_train)
rf_clf.score(X_test, y_test)


# In[ ]:




