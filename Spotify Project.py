#!/usr/bin/env python
# coding: utf-8

# # Imports

# In[1]:


import numpy as np
import pandas as pd 
from sklearn import tree
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report,confusion_matrix,accuracy_score
from sklearn.model_selection import train_test_split
from sklearn import metrics
from matplotlib import pyplot as plt
from sklearn import preprocessing
import seaborn as sns
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn import model_selection
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.preprocessing import LabelBinarizer
get_ipython().run_line_magic('matplotlib', 'inline')
import warnings
warnings.filterwarnings('ignore')


# # Read Data

# In[2]:


data = pd.read_csv('Final database.csv',low_memory=False)


# In[25]:


data.describe(include='all')


# # Remove Columns which are not required

# In[3]:


df_columns_removed = data.drop(columns=['Country','Title', 'Album','Artist', 'Uri', 'Album/Single', 'Genre', 'Explicit', 'LDA_Topic', 'bolero', 'boy band', 'country', 'dance/electronic', 'else', 'funk', 'hip hop', 'house', 'indie', 'jazz', 'k-pop', 'latin', 'metal', 'opm', 'pop', 'r&b/soul', 'rap', 'reggae', 'reggaeton', 'rock', 'trap' ])


# In[ ]:





# # Feature Engineering of date based properties

# In[4]:


df_columns_removed['Release_date'] = pd.to_datetime(df_columns_removed['Release_date'], errors = 'coerce')

df_columns_removed['Release_date:year'] = df_columns_removed['Release_date'].dt.year#2
df_columns_removed['Release_date:month'] = df_columns_removed['Release_date'].dt.month#3
df_columns_removed['Release_date:day_of_week'] = df_columns_removed['Release_date'].dt.day_of_week#4
df_columns_removed['Release_date:day_of_year'] = df_columns_removed['Release_date'].dt.day_of_year#5
df_columns_removed['Release_date:week_of_year'] = df_columns_removed['Release_date'].dt.weekofyear#5
df_columns_removed['Release_date:week'] = df_columns_removed['Release_date'].dt.week#5
df_columns_removed['Release_date:weekday'] = df_columns_removed['Release_date'].dt.weekday
df_columns_removed['Release_date:quarter'] = df_columns_removed['Release_date'].dt.quarter
df_columns_removed['Release_date:days_in_month'] = df_columns_removed['Release_date'].dt.days_in_month

df_columns_removed['Release_date:is_month_start'] = df_columns_removed['Release_date'].dt.is_month_start
df_columns_removed['Release_date:is_month_end'] = df_columns_removed['Release_date'].dt.is_month_end
df_columns_removed['Release_date:is_quarter_start'] = df_columns_removed['Release_date'].dt.is_quarter_start
df_columns_removed['Release_date:is_quarter_end'] = df_columns_removed['Release_date'].dt.is_quarter_end
df_columns_removed['Release_date:is_year_start'] = df_columns_removed['Release_date'].dt.is_year_start
df_columns_removed['Release_date:is_year_end'] = df_columns_removed['Release_date'].dt.is_year_end
df_columns_removed['Release_date:is_leap_year'] = df_columns_removed['Release_date'].dt.is_leap_year


y1 = pd.get_dummies(df_columns_removed['Release_date:is_month_start'], prefix='Release_date:is_month_start')
df_columns_removed = df_columns_removed.drop(columns=['Release_date:is_month_start'])
df_columns_removed = df_columns_removed.join(y1)

y2 = pd.get_dummies(df_columns_removed['Release_date:is_month_end'], prefix='Release_date:is_month_end')
df_columns_removed = df_columns_removed.drop(columns=['Release_date:is_month_end'])
df_columns_removed = df_columns_removed.join(y2)

y3 = pd.get_dummies(df_columns_removed['Release_date:is_quarter_start'], prefix='Release_date:is_quarter_start')
df_columns_removed = df_columns_removed.drop(columns=['Release_date:is_quarter_start'])
df_columns_removed = df_columns_removed.join(y3)

y4 = pd.get_dummies(df_columns_removed['Release_date:is_quarter_end'], prefix='Release_date:is_quarter_end')
df_columns_removed = df_columns_removed.drop(columns=['Release_date:is_quarter_end'])
df_columns_removed = df_columns_removed.join(y4)

y5 = pd.get_dummies(df_columns_removed['Release_date:is_year_start'], prefix='Release_date:is_year_start')
df_columns_removed = df_columns_removed.drop(columns=['Release_date:is_year_start'])
df_columns_removed = df_columns_removed.join(y5)

y6 = pd.get_dummies(df_columns_removed['Release_date:is_year_end'], prefix='Release_date:is_year_end')
df_columns_removed = df_columns_removed.drop(columns=['Release_date:is_year_end'])
df_columns_removed = df_columns_removed.join(y6)

y7 = pd.get_dummies(df_columns_removed['Release_date:is_leap_year'], prefix='Release_date:is_leap_year')
df_columns_removed = df_columns_removed.drop(columns=['Release_date:is_leap_year'])
df_columns_removed = df_columns_removed.join(y7)

df_columns_removed = df_columns_removed.drop(['Release_date'], axis = 1)

df_columns_removed.head()


# In[ ]:





# # Applying One hot encoding for Cluster feature

# In[5]:


y = pd.get_dummies(df_columns_removed.Cluster, prefix='Cluster')
df_cluster_removed = df_columns_removed.drop(columns=['Cluster'])
df_cluster_encoded = df_cluster_removed.join(y)


# # Replacing the missing values with the median value of the feature column

# In[6]:


df_cluster_encoded = df_cluster_encoded.fillna(df_cluster_encoded.median())
df_feature2 = df_cluster_encoded
print(df_feature2)


# In[7]:


df_2 = df_feature2


# In[ ]:





# # Converting the non-float values to Float 

# In[8]:


df_feature = df_2


# In[9]:


df_feature = df_feature.rename(columns={'Genre_new': 'genre'})
df_feature = df_feature.round(5)


df_feature_x = df_feature[df_feature.columns.difference(['genre'])]
cols = df_feature_x.select_dtypes(exclude=['float']).columns

print(df_feature_x)

df_feature_x[cols] = df_feature_x[cols].apply(pd.to_numeric, downcast='float', errors='coerce')

df_feature_x.info()

df_feature_t = pd.concat([df_feature_x, df_feature['genre']], axis=1)

print(df_feature_t)


# # Applying label encoder to encode genre field

# In[10]:


df_feature_t = df_feature_t.dropna()
df_feature_tt = df_feature_t

le = preprocessing.LabelEncoder()
df_feature_tt = df_feature_t.apply(le.fit_transform)
df_feature_tt['genre']


# # Finding correlation between the song characteristic fields

# In[11]:


corr = df_feature_tt[['danceability', 'energy', 'key', 'loudness', 'mode', 'speechiness', 'acoustics', 'instrumentalness', 'liveliness', 'valence', 'tempo']].corr(method='pearson')
corr.style.background_gradient(cmap='coolwarm').set_precision(2)


# In[ ]:





# # Introducing New columns after finding correlation

# In[12]:


multiplying_columns = ['danceability', 'energy', 'key', 'loudness', 'mode', 'speechiness', 'acoustics', 'instrumentalness', 'liveliness', 'valence', 'tempo']

for i in range(len(multiplying_columns)):
    for j in range(i+1, len(multiplying_columns)):
        df_feature_tt[multiplying_columns[i]+ ' * '+multiplying_columns[j]] = df_feature_tt[multiplying_columns[i]]*df_feature_tt[multiplying_columns[j]]
        
df_feature_tt.head()


# In[ ]:





# # Test, Train dataframes

# In[13]:


X = df_feature_tt.drop(['genre'], axis = 1)
Y = df_feature_tt['genre']

X_train, X_test, Y_train, Y_test = train_test_split(X, Y,test_size=0.2, random_state=112)

X_train.info()


# In[14]:





# In[ ]:





# # Model: Logistic Regression

# In[15]:


clf3=LogisticRegression()                    # importing Logistic Regression Algorithm

kf = model_selection.KFold(n_splits=5, shuffle=True)
for train_index, test_index in kf.split(df_feature_tt):
    X_train, X_test = X.iloc[train_index], X.iloc[test_index]
    Y_train, Y_test = Y.iloc[train_index], Y.iloc[test_index]
    clf3.fit(X_train,Y_train)
    Y_pred=clf3.predict(X_test)
    print("Logistic Regression metrics:")
    print("precision={}".format(metrics.precision_score(Y_test, Y_pred, average="weighted")))
    print("recall=   {}".format(metrics.recall_score(Y_test, Y_pred, average="weighted")))
    print("f1=       {}".format(metrics.f1_score(Y_test, Y_pred, average="weighted")))
    print()

clf3.fit(X_train,Y_train)

clf3.score(X_train,Y_train)

clf3.score(X_test,Y_test)

y_pred3=clf3.predict(X_test)                 
y_pred3

accuracyLogisticRegression=accuracy_score(y_pred3,Y_test)          # calculating accuracy
accuracyLogisticRegression


# # Model: Gaussian Naive Bayes

# In[16]:


gnb=GaussianNB()                            

kf = model_selection.KFold(n_splits=5, shuffle=True)
for train_index, test_index in kf.split(df_feature_tt):
    X_train, X_test = X.iloc[train_index], X.iloc[test_index]
    Y_train, Y_test = Y.iloc[train_index], Y.iloc[test_index]
    gnb.fit(X_train,Y_train)
    Y_pred4=clf3.predict(X_test)
    print("Gaussian Naive Bayes metrics:")
    print("precision={}".format(metrics.precision_score(Y_test, Y_pred4, average="weighted")))
    print("recall=   {}".format(metrics.recall_score(Y_test, Y_pred4, average="weighted")))
    print("f1=       {}".format(metrics.f1_score(Y_test, Y_pred4, average="weighted")))
    print()

gnb.fit(X_train,Y_train)

gnb.score(X_train,Y_train)

gnb.score(X_test,Y_test)

y_predGNB=gnb.predict(X_test)                 
y_predGNB

accuracyGNB=accuracy_score(y_predGNB,Y_test)          # calculating accuracy
accuracyGNB


# # Model: Decision Tree Classifier

# In[17]:


c=DecisionTreeClassifier(min_samples_split=1000)

kf = model_selection.KFold(n_splits=5, shuffle=True)
for train_index, test_index in kf.split(df_feature_tt):
    X_train, X_test = X.iloc[train_index], X.iloc[test_index]
    Y_train, Y_test = Y.iloc[train_index], Y.iloc[test_index]
    dt = c.fit(X_train,Y_train)
    Y_pred = c.predict(X_test)
    print("Decision Tree metrics:")
    print("precision={}".format(metrics.precision_score(Y_test, Y_pred, average="weighted")))
    print("recall=   {}".format(metrics.recall_score(Y_test, Y_pred, average="weighted")))
    print("f1=       {}".format(metrics.f1_score(Y_test, Y_pred, average="weighted")))


c.fit(X_train,Y_train)

c.score(X_train,Y_train)

c.score(X_test,Y_test)

y_predDT=c.predict(X_test)                 

accuracyDT=accuracy_score(y_predDT,Y_test)          # calculating accuracy
accuracyDT


# # Model: RandomForestClassifier

# In[18]:



clf1=RandomForestClassifier(n_estimators=50)

kf = model_selection.KFold(n_splits=5, shuffle=True)
for train_index, test_index in kf.split(df_feature_tt):
    X_train, X_test = X.iloc[train_index], X.iloc[test_index]
    Y_train, Y_test = Y.iloc[train_index], Y.iloc[test_index]
    model1=clf1.fit(X_train,Y_train)
    Y_pred=clf1.predict(X_test)
    print("Random Forest classifier metrics:")
    print("precision={}".format(metrics.precision_score(Y_test, Y_pred, average="weighted")))
    print("recall=   {}".format(metrics.recall_score(Y_test, Y_pred, average="weighted")))
    print("f1=       {}".format(metrics.f1_score(Y_test, Y_pred, average="weighted")))
    print()

clf1.fit(X_train,Y_train)
y_pred1=clf1.predict(X_test)                        # predicting X_test
y_pred1

accuracyRF=accuracy_score(y_pred1,Y_test)
accuracyRF

classificationreport1=classification_report(y_pred1,Y_test)
print(classificationreport1)


# In[19]:





# In[20]:


a = [accuracyDT,accuracyRF,accuracyLogisticRegression,accuracyGNB]                                          
b = ['Decision Tree Classifier','Random Forest Classifier','Logistic Regression','Naive Bayes Classifier'] 

sns.barplot(x=a,y=b,ci=None,data=df_feature_tt)                      


# In[ ]:




