#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import streamlit as st
import joblib

# In[2]:


st.title('Predict charges')
st.markdown('Model Predict Charges')

st.header('Person Details')
col1,col2,col3=st.columns(3)
with col1:
    age = st.slider('age',2,100,1)
with col2:
    bmi = st.slider('bmi',1,60,4)
with col3:
    children = st.slider('children',0,5,0)


# In[5]:

insurance=pd.read_csv('insurance.csv')
y=insurance.charges
X=insurance.drop('charges',axis=1)


from sklearn.tree import DecisionTreeRegressor
tree=DecisionTreeRegressor(max_depth=16)
tree.fit(X,y)
tree.score(X,y)

joblib.dump(tree,'tree_model.sav')




if st.button('Predict Charges'):
    result=tree.predict(np.array([[age,bmi,children]]))
    st.text(result[0])


# In[6]:




def predict(data):
    clf=joblib.load('tree_model.sav')
    return clf.predict(data)
