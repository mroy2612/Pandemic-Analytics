#!/usr/bin/env python
# coding: utf-8

# In[7]:


import numpy as np
import matplotlib as mtpltlib
import pandas as pd
import csv


# In[11]:


#import libraries
import pandas as pd
import os


# In[12]:


pwd


# In[13]:


import itertools
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import NullFormatter
import pandas as pd
import numpy as np
import matplotlib.ticker as ticker
from sklearn import preprocessing
get_ipython().run_line_magic('matplotlib', 'inline')


# In[16]:


df = pd.read_csv("C:\\Users\\thinkpad\\desktop\\teleCust1000t.csv")
df.head()


# In[17]:


df['custcat'].value_counts()


# In[18]:


df.hist(column='income', bins=50)


# In[19]:


df.columns


# In[20]:


X = df[['region', 'tenure','age', 'marital', 'address', 'income', 'ed', 'employ','retire', 'gender', 'reside']] .values  #.astype(float)
X[0:5]


# In[21]:


y = df['custcat'].values
y[0:5]


# In[22]:


X = preprocessing.StandardScaler().fit(X).transform(X.astype(float))
X[0:5]


# In[23]:


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split( X, y, test_size=0.2, random_state=4)
print ('Train set:', X_train.shape,  y_train.shape)
print ('Test set:', X_test.shape,  y_test.shape)


# In[24]:


from sklearn.neighbors import KNeighborsClassifier


# In[25]:


k = 4
#Train Model and Predict  
neigh = KNeighborsClassifier(n_neighbors = k).fit(X_train,y_train)
neigh


# In[26]:


yhat = neigh.predict(X_test)
yhat[0:5]


# In[27]:


from sklearn import metrics
print("Train set Accuracy: ", metrics.accuracy_score(y_train, neigh.predict(X_train)))
print("Test set Accuracy: ", metrics.accuracy_score(y_test, yhat))


# In[28]:


Ks = 10
mean_acc = np.zeros((Ks-1))
std_acc = np.zeros((Ks-1))
ConfustionMx = [];
for n in range(1,Ks):
    
    #Train Model and Predict  
    neigh = KNeighborsClassifier(n_neighbors = n).fit(X_train,y_train)
    yhat=neigh.predict(X_test)
    mean_acc[n-1] = metrics.accuracy_score(y_test, yhat)

    
    std_acc[n-1]=np.std(yhat==y_test)/np.sqrt(yhat.shape[0])

mean_acc


# In[30]:


plt.plot(range(1,Ks),mean_acc,'g')
plt.fill_between(range(1,Ks),mean_acc - 1 * std_acc,mean_acc + 1 * std_acc, alpha=0.10)
plt.legend(('Accuracy ', '+/- 3xstd'))
plt.ylabel('Accuracy ')
plt.xlabel('Number of Neighbors (K)')
plt.tight_layout()
plt.show()


# In[ ]:




