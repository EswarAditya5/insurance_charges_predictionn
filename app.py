#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import streamlit as st
from predictions import predict

# In[2]:


st.title('Predict charges')
st.markdown('Model Predict Charges')

st.header('Person Details')
col1,col2,col3=st.columns(3)
with col1:
    age = st.slider('age',2,100,1)
with col2:
    bmi = st.slider('bmi',14,60,4)
with col3:
    children = st.slider('children',0,5,1)


# In[3]:


if st.button('Predict Charges'):
    result=predict(np.array([['age','bmi','children']]))
    st.text(result[0])


# In[ ]:




