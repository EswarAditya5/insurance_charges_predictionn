#!/usr/bin/env python
# coding: utf-8

# In[1]:


import io
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st
get_ipython().run_line_magic('cd', '"D:\\Imarticus\\Machine Learning\\lms\\projects\\Cloud Project"')


# In[2]:


insurance=pd.read_csv('insurance.csv')
insurance.head()


# In[3]:


insurance.tail()


# In[4]:


insurance.info()


# In[5]:


insurance.age.value_counts()


# In[6]:


insurance.bmi.value_counts()


# In[7]:


insurance.children.value_counts()


# In[8]:


insurance.charges.max()


# In[9]:


insurance.describe()


# In[10]:


y=insurance.charges
X=insurance.drop('charges',axis=1)


import joblib


from sklearn.tree import DecisionTreeRegressor


# In[18]:


tree=DecisionTreeRegressor(max_depth=14)


# In[19]:


tree.fit(X,y)


# In[20]:


tree.score(X,y)


# In[21]:


joblib.dump(tree,'tree_model.sav')


# st.title('Predict charges')
# st.markdown('Model Predict Charges')
# 
# st.header('Person Details')
# col1,col2,col3=st.columns(3)
# with col1:
#     age = st.slider('age',2,100,1)
# with col2:
#     bmi = st.slider('bmi',14,60,4)
# with col3:
#     children = st.slider('children',0,5,1)

# if st.button('Predict Charges'):
#     result=predict(np.array([[age,bmi,children]]))
#     st.text(result[0])

# # import joblib
# def predict(data):
#     clf=joblib.load('tree_model.sav')
#     return clf.predict(data)
