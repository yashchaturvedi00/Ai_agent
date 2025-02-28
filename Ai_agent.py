#!/usr/bin/env python
# coding: utf-8

# In[4]:


import pandas as pd  
import numpy as np  
import seaborn as sns  
import matplotlib.pyplot as plt  

df = pd.read_csv("./placement-dataset.csv")

print(df.describe())  
print(df.info())  
x = df[["cgpa"]]  
y = df["package"]  

# Visualize the relationships in the dataset
sns.pairplot(df, kind="scatter")  # Scatter plot to see correlations
plt.show()

# Import required functions
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=42)

lm = LinearRegression()
lm.fit(x_train, y_train)  # Train the model

predictions = lm.predict(x_test) 

mae = mean_absolute_error(y_test, predictions)
print("Mean Absolute Error:", mae)
import joblib
joblib.dump(lm, "model.pkl")


# In[ ]:




