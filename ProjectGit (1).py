#!/usr/bin/env python
# coding: utf-8

# 0) Libraries

# In[65]:


import pandas as pd
import numpy as np
# preparing data
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV

# models
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler

#metrics
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.metrics import mean_absolute_percentage_error


# Downloading data from 
# https://www.kaggle.com/datasets/mathchi/churn-for-bank-customers
# 

# In[ ]:


dataframe = pd.read_csv(r"C:\Users\panos\Semi\Machine learning\churn.csv")


# In[ ]:


dataframe


# In[ ]:


dataframe.isna().sum()


# 2) Converting the categorical variables to Numerical

# In[ ]:


label_encoder = preprocessing.LabelEncoder()


# In[ ]:


for i in ['Surname', 'Geography', 'Gender']:
    dataframe[i]=label_encoder.fit_transform(dataframe[i])


# In[ ]:


dataframe.loc[:5,['Surname', 'Geography', 'Gender']]


# 3) Split X & y, train & test

# In[ ]:


train_columns = ['CreditScore', 'Geography', 'Gender', 'Tenure', 'Balance', 'NumOfProducts', 'HasCrCard', 'IsActiveMember', 'EstimatedSalary']


# In[ ]:


target_column = ['Exited']


# In[ ]:


X = dataframe.copy()[train_columns]


# In[ ]:


Y = dataframe.copy()[target_column]


# In[ ]:


print(X.shape)
print(Y.shape)


# In[ ]:


X_train, X_test, y_train, y_test = train_test_split(X, Y, 
                                                    test_size=0.2,
                                                    stratify = Y,
                                                    random_state=42)


# In[ ]:


print(X_train.shape)
print(X_test.shape)
print(y_train.shape)
print(y_test.shape)


# 4)Train using Random Forest

# In[ ]:


model_rf_class = RandomForestClassifier() 


# In[ ]:


param_grid_class = {'n_estimators': [200, 300, 500],
              'max_depth' : [3, 5, 10],
              'criterion' :['gini', 'entropy']
             }


# In[ ]:


grid_search = GridSearchCV(estimator=model_rf_class, 
                              param_grid=param_grid_class, 
                              scoring= 'precision',
                              cv=5,
                             verbose=1)


# In[ ]:


grid_search.fit(X_train, y_train.values.ravel())


# In[ ]:


grid_search.best_params_


# 5) Table of feature importance 

# In[1]:


pd.DataFrame(data = grid_search.best_estimator_.feature_importances_,
            index = X.columns,
            columns=['feature_importance']).sort_values(by='feature_importance',
                                                       ascending =False)


# # 
6) Train using SVM

# In[66]:


scaler = StandardScaler()
scaler.fit(X_train)


# In[68]:


X_train_scaled = scaler.transform(X_train)
X_test_scaled = scaler.transform(X_test)


# In[69]:


X_train_scaled = pd.DataFrame(X_train_scaled,columns = X.columns)
X_test_scaled = pd.DataFrame(X_test_scaled,columns = X.columns) 


# In[70]:


model_SVC = SVC(class_weight= 'balanced')

tuned_parameters = [{'kernel': ['rbf'], 
                     'gamma': ["auto", "scale"],
                     'C': [0.001, 0.1, 1, 10, 100, 1000]}
                   ]


# In[71]:


model_SVC_grid = GridSearchCV(estimator  =  model_SVC,
                              param_grid = tuned_parameters,
                              scoring="balanced_accuracy",
                              cv=5,
                              verbose = True
                             )


# In[76]:


model_SVC_grid = model_SVC_grid.fit(X_train_scaled, y_train.values.ravel()) 


# In[77]:


model_SVC_grid.best_params_


# In[78]:


pd.DataFrame(model_SVC_grid.cv_results_).sort_values('rank_test_score').head(1)['params']


# 7) Prediction of the two models in the test sample

# **Random Forest**

# In[90]:


predictions_rf = grid_search.predict(X_test)


# In[93]:


confusion_matrix(y_test,predictions_rf)


# In[95]:


accuracy_score(y_test,predictions_rf)


# **Support Vector Machine**

# In[79]:


y_predictions_SVC = model_SVC_grid.predict(X_test_scaled)


# In[81]:


print(y_predictions_SVC)


# In[82]:


confusion_matrix(y_test, y_predictions_SVC)


# In[97]:


accuracy_score(y_test, y_predictions_SVC)

