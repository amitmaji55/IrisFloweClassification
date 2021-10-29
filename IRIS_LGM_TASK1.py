#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# In[2]:


from sklearn.datasets import load_iris


# In[3]:



dataset = load_iris()


# In[4]:



X = dataset.data
y = dataset.target


# In[5]:


X.shape


# In[6]:


y.shape


# In[7]:


plt.plot(X[:, 0][y == 0] * X[:, 1][y == 0], X[:, 2][y == 0] * X[:, 3][y == 0], 'r.', label="Satosa")
plt.plot(X[:, 0][y == 1] * X[:, 1][y == 1], X[:, 2][y == 1] * X[:, 3][y == 1], 'g.', label="Virginica")
plt.plot(X[:, 0][y == 2] * X[:, 1][y == 2], X[:, 2][y == 2] * X[:, 3][y == 2], 'b.', label="Versicolour")
plt.legend()
plt.show()


# In[8]:


from sklearn.preprocessing import StandardScaler
X = StandardScaler().fit_transform(X)


# In[9]:


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y)


# In[10]:



from sklearn.linear_model import LogisticRegression
log_reg = LogisticRegression()
log_reg.fit(X_train, y_train)


# In[11]:


log_reg.score(X, y)


# In[12]:



log_reg.score(X_train, y_train)


# In[13]:


log_reg.score(X_test, y_test)


# In[ ]:




