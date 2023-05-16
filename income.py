#!/usr/bin/env python
# coding: utf-8

# ## importing libraries

# In[2]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
import warnings
warnings.filterwarnings("ignore")
get_ipython().run_line_magic('matplotlib', 'inline')


# ## Load the dataset

# In[3]:


data = pd.read_csv("salary.csv")
data.head()


# In[4]:


data.tail()


# ## Data Description

# In[5]:


data.info()


# ## EDA

# In[6]:


data.columns


# In[7]:


sns.countplot(data=data,x='capital-gain')


# In[8]:


data["education"] = [cols.replace(' ', '') for cols in data["education"]]
sns.countplot(data=data,x='education')
plt.xticks(rotation=90)


# In[9]:


data["workclass"] = [cols.replace(' ', '') for cols in data["workclass"]]
sns.countplot(data=data,x='workclass')
plt.xticks(rotation=90)


# In[10]:


data["marital-status"] = [cols.replace(' ', '') for cols in data["marital-status"]]
sns.countplot(data=data,x='marital-status')
plt.xticks(rotation=90)


# In[11]:


data["native-country"] = [cols.replace(' ', '') for cols in data["native-country"]]
sns.countplot(data=data,x='native-country')
plt.xticks(rotation=90)


# In[12]:


data["occupation"] = [cols.replace(' ', '') for cols in data["occupation"]]
sns.countplot(data=data,x='occupation')
plt.xticks(rotation=90)


# In[13]:


data["relationship"] = [cols.replace(' ', '') for cols in data["relationship"]]
sns.countplot(data=data,x='relationship')
plt.xticks(rotation=90)


# In[14]:


data["race"] = [cols.replace(' ', '') for cols in data["race"]]
sns.countplot(data=data,x='race')
plt.xticks(rotation=90)


# In[17]:


data["sex"] = [cols.replace(' ', '') for cols in data["sex"]]
sns.countplot(data=data,x='sex')
plt.xticks(rotation=90)


# ## Filling Up missing values

# In[18]:


data = data.replace('?', np.nan)


# In[19]:


# Chechking null values 
def about_data(df):
    total_missing_values = df.isnull().sum().reset_index()
    total_missing_values = total_missing_values.rename(columns={'index':'columns',0:'total missing'})
    total_missing_values['ration of missing'] = total_missing_values['total missing']/len(df)
    return total_missing_values
about_data(data)


# In[20]:


about_data(data)


# In[21]:


data.dropna(inplace=True,axis=0)
about_data(data)


# In[22]:


sns.pairplot(data,hue='fnlwgt',corner=True)


# In[23]:


plt.figure(figsize=(10,8),dpi=100)
sns.heatmap(data.corr(),cmap="viridis",annot=True,linewidth=0.5)


# ### Drop unnecessary columns

# In[24]:


data.drop(['fnlwgt'], axis=1, inplace=True)


# ### Encode categorical features

# In[27]:


categorical_features = ['workclass', 'education', 'marital-status', 'occupation', 'relationship', 'race', 'sex', 'native-country']
for feature in categorical_features:
    le = LabelEncoder()
    data[feature] = le.fit_transform(data[feature])


# ### Split the dataset into training and testing sets

# In[28]:


X = data.drop('salary', axis=1)
y = data['salary']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)


# ###  Scale the features

# In[29]:


scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)


# ## Decision Tree Classifier

# In[30]:


dt = DecisionTreeClassifier(random_state=42)
dt.fit(X_train, y_train)
y_pred_dt = dt.predict(X_test)
print("Decision Tree Classifier:\n", classification_report(y_test, y_pred_dt))


# ## K-Nearest Neighbors Classifier 

# In[31]:


knn = KNeighborsClassifier()
knn.fit(X_train, y_train)
y_pred_knn = knn.predict(X_test)
print("K-Nearest Neighbors Classifier:\n", classification_report(y_test, y_pred_knn))


# ## Support Vector Machine Classifier 

# In[32]:


svc = SVC(random_state=42)
svc.fit(X_train, y_train)
y_pred_svc = svc.predict(X_test)
print("Support Vector Machine Classifier:\n", classification_report(y_test, y_pred_svc))

