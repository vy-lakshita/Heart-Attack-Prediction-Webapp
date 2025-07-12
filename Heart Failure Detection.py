#!/usr/bin/env python
# coding: utf-8

# In[1]:

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
import pickle as pkl
from sklearn.datasets import make_classification

# <h1>Just info of the data </h1>

# In[2]:


df=pd.read_csv("heart.csv")


# In[3]:


df.shape


# In[4]:


df.head(5)


# In[5]:


df.duplicated().sum()


# In[6]:


df.isna().sum()*100/len(df)






# <h1>Making data  Pipeline and using Random Forest to train the model </h1>

# In[24]:


from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline,make_pipeline
from sklearn.impute import SimpleImputer

#imputation transformer
"""trf1=ColumnTransformer([
    ("impute_age",SimpleImputer(strategy="median"),[0]),
    ("impute_sex",SimpleImputer(strategy="most_frequent"),[1]),
    ("impute_chestPainType",SimpleImputer(strategy="most_frequent"),[2]),
    ("impute_RestingBP",SimpleImputer(strategy="median"),[3]),
    ("impute_Cholestrol",SimpleImputer(strategy="median"),[4]),
    ("impute_FastingBS",SimpleImputer(strategy="most_frequent"),[5]),
     ("impute_RestingECG",SimpleImputer(strategy="most_frequent"),[6]),
     ("impute_MaxHR",SimpleImputer(strategy="mean"),[7]),
     ("impute_ExcerciseAngina",SimpleImputer(strategy="most_frequent"),[8]),
     ("impute_Oldpeak",SimpleImputer(strategy="mean"),[9]),
     ("impute_ST_Slope",SimpleImputer(strategy="most_frequent"),[10]),
],remainder="passthrough")"""


# In[37]:


#, [1,2,6,8,10]
from sklearn.preprocessing import OneHotEncoder

# one hot encoding
trf2=ColumnTransformer([
('One_hot_encoding',OneHotEncoder(handle_unknown='ignore'), [1,2,6,8,10])
], remainder='passthrough')


# In[32]:





# In[38]:


trf3=random = RandomForestClassifier(max_depth=15, random_state=2)


# In[39]:


pipe = Pipeline ([
#('trf1', trf1),
('trf2', trf2),
('trf3', trf3),  
])


# In[40]:


#trf2.fit_transform(x_train_n)


# In[41]:


x_n=df.drop(columns=["HeartDisease"])
y_n=df["HeartDisease"]


# In[42]:


from sklearn.model_selection import train_test_split
x_train_n,x_test_n,y_train_n,y_test_n=train_test_split(x_n,y_n,test_size=0.2,random_state=3)




# In[43]:


pipe.fit(x_train_n.values,y_train_n)


# In[44]:


pipe_pred=pipe.predict(x_test_n.values)


# In[45]:


#print(classification_report(y_test_n,pipe_pred))


# In[46]:


input_n=np.array([49,"M","ASY",140,234,0,"Normal",140,"Y",1,"Flat"],dtype=object).reshape(1,11)


# In[47]:


result = pipe.predict(input_n)


# In[48]:


print(result)


# <h1>Dumping the pipeline through pickle</h1>

# In[196]:


pkl.dump(pipe,open("pipe_values.pkl","wb"))


# In[ ]:




