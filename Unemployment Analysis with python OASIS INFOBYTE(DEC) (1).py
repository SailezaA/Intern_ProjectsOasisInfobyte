#!/usr/bin/env python
# coding: utf-8

# #### Unemployment is measured by the unemployment rate which is the number of people who are unemployed as a percentage of the total labour force. We have seen a sharp increase in the unemployment rate during Covid-19.
# 
# Task:
#      To analyze the unemployment rate. 

# In[1]:


import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
get_ipython().system('pip install pandas matplotlib seaborn')


# In[2]:


# Loading the dataset
df = pd.read_csv("Unemployment in India.csv")
df


# In[3]:


# Display basic information about the dataset
print(df.info())


# In[4]:


# Display the first few rows of the dataset
print(df.head())


# In[5]:


# Check for missing values
print("Missing Values:")
print(df.isnull().sum())


# In[6]:


# Print the column names
print("Column Names:")
print(df.columns)


# In[7]:


print(df.describe())


# In[8]:


df.columns=['Region','Date','Frequency','Estimated Unemployment Rate','Estimated Employed','Estimated Labour Participation Rate','Area']


# In[9]:


#analyzing thr top rows of dataset
df.head()


# ### checking the correlation between the features of dataset
# plotting the correlation heatmap

# In[10]:


df = df.dropna() # Drop rows with NaN values
df


# In[11]:


df.columns = ['Region', 'Date', 'Frequency', 'Estimated Unemployment Rate',
              'Estimated Employed', 'Estimated Labour Participation Rate',
              'Area']




# In[12]:


#converting into numeric type
numeric_columns = ['Region', 'Date', 'Frequency', 'Area']
df1 = df.copy()
df1[numeric_columns] = df1[numeric_columns].apply(pd.to_numeric, errors='coerce')


# In[13]:


# Plotting correlation heatmap
plt.figure(figsize=(10, 8))
correlation_matrix = df1.corr()
sns.heatmap(correlation_matrix, annot=True, cmap="coolwarm", fmt=".2f", linewidths=1.5)
plt.title('Correlation Matrix')
plt.show()


# In[14]:


# plotting histplot

df.columns=['Region','Date','Frequency','Estimated Unemployment Rate',
                'Estimated Employed','Estimated Labour Participation Rate',
                'Area']
plt.title('Indian Unemployment')
sns.histplot(x='Estimated Employed',hue='Region',data=df)
plt.show()
     


# In[15]:


# plotting histplot

plt.figure(figsize=(10,8))
plt.title("Indian Unemployment")
sns.histplot(x="Estimated Unemployment Rate",hue='Region',data=df)
plt.show()
     


# In[16]:


# Plotting pie chart for Estimated Unemployment Rate by Region

plt.figure(figsize=(10, 8))
region_unemployment = df.groupby('Region')['Estimated Unemployment Rate'].mean()
plt.pie(region_unemployment, labels=region_unemployment.index, autopct='%1.1f%%', startangle=140)
plt.title('Estimated Unemployment Rate by Region')
plt.show()


# In[17]:


#Barplot for analysis of unemployment

plt.figure(figsize=(10, 8))
region_unemployment = df.groupby('Region')['Estimated Unemployment Rate'].mean()
region_unemployment = region_unemployment.sort_values(ascending=False)  # Sort the values for better visualization
plt.bar(region_unemployment.index, region_unemployment)
plt.xlabel('Region')
plt.ylabel('Estimated Unemployment Rate')
plt.title('Estimated Unemployment Rate by Region')
plt.xticks(rotation=45, ha='right')  # Rotate x-axis labels for better readability
plt.show()


# In[ ]:





# In[ ]:




