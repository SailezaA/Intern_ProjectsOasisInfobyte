#!/usr/bin/env python
# coding: utf-8

# ## Internship in Data Science project of OASIS INFOBYTE(DEC)

# In[1]:


import pandas as pd
import numpy as np 
import matplotlib.pyplot as plt
import plotly.express as px
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.ensemble import RandomForestClassifier

get_ipython().run_line_magic('matplotlib', 'inline')
get_ipython().system('pip install pandas plotly')
get_ipython().system('pip install --upgrade plotly pandas')
get_ipython().system('pip install seaborn matplotlib')
get_ipython().system('pip install scikit-learn')



# ## Iris flower has three species; 
# Species differs according to their measurements. 
# 
# TASK
# 
# Now, assuming that I have the measurements of the iris flowers according to
# their species, and here my task is to train a machine learning model that can learn from the
# measurements of the iris species and classify them.
# 
# Dataset Information:
# 
# Dataset information is from kaggle as provided in the task link(https://www.kaggle.com/datasets/saurabh00007/iriscsv)
# 
# 1.SepalLengthCm
# 
# 2.SepalWidthCm
# 
# 3.PetalLengthCm
# 
# 4.PetalWidthCm5
# 
#         Species:
#             Setosa
#             Versicolor
#             Verginica

# ### Importing dataset 

# In[2]:


Iris_df=pd.read_csv("iris.csv")

print("Data loaded successfully")


# In[3]:


Iris_df


# In[4]:


#checking the basic information provided in the data
Iris_df.info()


# In[5]:


Iris_df.shape


# In[6]:


Iris_df.describe()


# In[7]:


Iris_df.isnull().sum()


# In[8]:


unique_species = Iris_df["Species"].unique()
print("Unique number of values in dataset species:", len(unique_species))
print("Unique species in Iris dataset:", unique_species)


# ## Data Analysis

# In[9]:


# Create scatter matrix using Plotly Express scatter function
fig = px.scatter_matrix(
    Iris_df,
    dimensions=["SepalLengthCm", "SepalWidthCm", "PetalLengthCm", "PetalWidthCm"],
    color="Species",
    title="Pair Plot of Iris Dataset",
    labels={"Species": "Target"}
)



# In[10]:


# Update marker symbol for the scatter plots
fig.update_traces(marker=dict(symbol="x"))

# Show the plot
fig.show()
print("Iris-setosa is differ from all species in the given dataset ")


# In[11]:


##check correlation in given datset

# Calculate and print the correlation matrix for numeric columns

correlation_matrix = Iris_df.select_dtypes(include=['float64', 'int64']).corr()
print("Correlation Matrix:\n", correlation_matrix)


# In[12]:


# Plotting a heatmap
plt.figure(figsize=(8, 6))
sns.heatmap(correlation_matrix, annot=True, cmap="coolwarm", fmt=".2f", linewidths=.5)
plt.title("Correlation Matrix")
plt.show()


# In[13]:


#Creating scatter plots to visualize the relationships between pairs of variables. 

sns.pairplot(Iris_df, hue="Species")
plt.show()


# ## Using Naive-Bayes Classifier

# In[14]:


X = Iris_df.drop("Species", axis=1)

# Use the "Species" column as the target variable
y = Iris_df["Species"]

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create a Gaussian Naive Bayes classifier
naive_bayes_classifier = GaussianNB()

# Train the classifier on the training set
naive_bayes_classifier.fit(X_train, y_train)

# Make predictions on the testing set
y_pred = naive_bayes_classifier.predict(X_test)



# In[15]:


# Evaluate the performance of the classifier
accuracy = accuracy_score(y_test, y_pred)
conf_matrix = confusion_matrix(y_test, y_pred)
class_report = classification_report(y_test, y_pred)



# In[16]:


# Print the results
print(f"Accuracy: {accuracy:.2f}")
print("\nConfusion Matrix:")
print(conf_matrix)
print("\nClassification Report:")
print(class_report)


# In[17]:


# Check for duplicates
print("Number of duplicate rows:", Iris_df.duplicated().sum())

# Remove duplicates
Iris_df = Iris_df.drop_duplicates()


# ## Using Random Forest Classifier

# In[18]:


# Drop the "Species" column to get the features
X = Iris_df.drop("Species", axis=1)

# Use the "Species" column as the target variable
y = Iris_df["Species"]


# In[19]:


# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# In[20]:


# Create a Random Forest classifier
random_forest_classifier = RandomForestClassifier(random_state=42)


# In[21]:


# Train the classifier on the training set
random_forest_classifier.fit(X_train, y_train)


# In[22]:


# Make predictions on the testing set
y_pred = random_forest_classifier.predict(X_test)


# In[23]:


# Evaluate the performance of the classifier
accuracy = accuracy_score(y_test, y_pred)
conf_matrix = confusion_matrix(y_test, y_pred)
class_report = classification_report(y_test, y_pred)


# In[24]:


# Print the results
print(f"Accuracy: {accuracy:.2f}")
print("\nConfusion Matrix:")
print(conf_matrix)
print("\nClassification Report:")
print(class_report)


# In[ ]:





# In[ ]:




