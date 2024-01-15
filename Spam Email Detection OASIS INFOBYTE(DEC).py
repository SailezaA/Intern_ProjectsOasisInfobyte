#!/usr/bin/env python
# coding: utf-8

# # Pragya Awasthi
# Intern task of Data Science
#         

# ## Importing necessary libraries

# In[2]:


import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report


# ## Loading dataset

# In[4]:


data = pd.read_csv("C:\\Users\\hp\\Downloads\\archive\\spam.csv", encoding="ISO-8859-1")
data


# In[5]:


data.head()


# In[6]:


data.tail()


# ### Split the dataset into training and testing sets

# In[7]:


X_train, X_test, y_train, y_test = train_test_split(data['v2'], data['v1'], test_size=0.2, random_state=42)


# ### Create a TF-IDF representation of the text data

# In[8]:


vectorizer = TfidfVectorizer()
X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)


# ### Train a Support Vector Machine (SVM) classifier 
# Support vector classification(svc)

# In[9]:


classifier = SVC(kernel='linear')
classifier.fit(X_train_tfidf, y_train)


# ### Making predictions on the given dataset

# In[10]:


predictions = classifier.predict(X_test_tfidf)


# ### Evaluation of the given model

# In[11]:


accuracy = accuracy_score(y_test, predictions)
conf_matrix = confusion_matrix(y_test, predictions)
classification_rep = classification_report(y_test, predictions)


# #### Here, first of all we print the accuracy of the model
# #### The second line prints the confusion matrix of the model and
# #### The third line prints the classification report of the model
# ####  The actual values for accuracy, confusion matrix, and classification report are assumed to be stored in the   variables accuracy, conf_matrix, and classification_rep as shown below.

# In[12]:


print(f"Accuracy: {accuracy:.2f}")
print(f"Confusion Matrix:\n{conf_matrix}")
print(f"Classification Report:\n{classification_rep}")


# ### The model achieved a strong accuracy of 98%, accurately classifying "ham" and "spam."
# It excelled in precision (98% for "ham," 99% for "spam") and recall (100% for "ham," 89% for "spam"). 
# The F1-score, a balance between precision and recall, was high for both classes (99% for "ham," 93% for "spam"). 
# Despite slightly lower recall for "spam," the model's overall performance, as indicated by macro and weighted averages, is robust across both classes.
# 
# 
# 
# 
# 
# 

# In[ ]:




