#!/usr/bin/env python
# coding: utf-8

# In[1]:


import joblib
def predict(data):
    clf=joblib.load('tree_model.sav')
    return clf.predict(data)


# In[ ]:




