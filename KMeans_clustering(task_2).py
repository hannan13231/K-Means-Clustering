#!/usr/bin/env python
# coding: utf-8

# 
# ## **TASK 2 - SPARK FOUNDATION**
# 
# In this task we will predict the optimum number of clusters and represent it visually
# 
# ### *CREATED BY : Sanurhanaan Shaikh*
# 

# 
# ### STEP 1 :
# Importing all the libraries and data required for the regression task
# 

# In[99]:


# Importing the libraries

import pandas as pd
import numpy as np
from sklearn import datasets
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import sklearn.metrics as sm
import math
get_ipython().run_line_magic('matplotlib', 'inline')


# In[100]:


# Load the iris dataset

iris = datasets.load_iris()
iris_df = pd.DataFrame(iris.data, columns = iris.feature_names)

iris_df.head(20)


# ### STEP 2 :
# Checking for null values, data-type and errors in the given data repository

# In[101]:


data.isna()


# In[102]:


data.info()


# In[103]:


print(iris.target_names)


# In[104]:


print(iris.target)


# In[105]:


y = pd.DataFrame(iris.target, columns=['Target'])
y.head()


# ### STEP 3 :
# Creating a scatter plot to find relation between the data

# In[106]:


# Plotting the distribution of iris_target data

plt.figure(figsize=(20,5))
colors = np.array(['red', 'green', 'blue'])
iris_targets_legend = np.array(iris.target_names)
red_patch = mpatches.Patch(color='red', label='Setosa')
green_patch = mpatches.Patch(color='green', label='Versicolor')
blue_patch = mpatches.Patch(color='blue', label='Virginica')

#iris_sepal data
plt.subplot(1, 2, 1)
plt.scatter(iris_df['sepal length (cm)'], iris_df['sepal width (cm)'], c=colors[y['Target']])
plt.title('Sepal Length vs Sepal Width')
plt.legend(handles=[red_patch, green_patch, blue_patch])

#iris_petal data
plt.subplot(1, 2, 2)
plt.scatter(iris_df['petal length (cm)'], iris_df['petal width (cm)'], c=colors[y['Target']])
plt.title('Petal Length vs Petal Width')
plt.legend(handles=[red_patch, green_patch, blue_patch])


# In[107]:


iris_k_mean_model = KMeans(n_clusters=3)
iris_k_mean_model.fit(iris_df)


# In[108]:


print(iris_k_mean_model.labels_)


# In[109]:


print(iris_k_mean_model.cluster_centers_)


# ### STEP 4 :
# Classification of target value and predicted value

# In[110]:


plt.figure(figsize=(20,5))

colors = np.array(['red', 'green', 'blue'])

predictedY = np.choose(iris_k_mean_model.labels_, [1, 0, 2]).astype(np.int64)

plt.subplot(1, 2, 1)
plt.scatter(iris_df['petal length (cm)'], iris_df['petal width (cm)'], c=colors[y['Target']])
plt.title('Before classification')
plt.legend(handles=[red_patch, green_patch, blue_patch])

plt.subplot(1, 2, 2)
plt.scatter(iris_df['petal length (cm)'], iris_df['petal width (cm)'], c=colors[predictedY])
plt.title("Model's classification")
plt.legend(handles=[red_patch, green_patch, blue_patch])


# ### STEP 5 :
# Checking the Accuracy of the model

# In[111]:


sm.accuracy_score(predictedY, y['Target'])


# ### STEP 6 :
# **Interpretation of Confusion Matrix -**
# Correctly identifed all 0 classes as 0’s
# correctly classified 48 class 1’s but miss-classified 2 class 1’s as class 2
# correctly classified 36 class 2’s but miss-classified 14 class 2’s as class 1

# In[112]:


sm.confusion_matrix(predictedY, y['Target'])


# In[113]:


iris_df.shape


# In[114]:


dist_points_from_cluster_center = []
K = range(1,10)
for no_of_clusters in K:
  k_model = KMeans(n_clusters=no_of_clusters)
  k_model.fit(iris_df)
  dist_points_from_cluster_center.append(k_model.inertia_)


# In[115]:


dist_points_from_cluster_center


# ### STEP 6 :
# Making Predictions

# In[116]:


plt.plot(K, dist_points_from_cluster_center)


# In[117]:


plt.plot(K, dist_points_from_cluster_center)
plt.plot([K[0], K[8]], [dist_points_from_cluster_center[0], 
                        dist_points_from_cluster_center[8]], 'ro-')
plt.show()


# In[118]:


x = [K[0], K[8]]
y = [dist_points_from_cluster_center[0], dist_points_from_cluster_center[8]]

# Calculate the coefficients. This line answers the initial question. 
coefficients = np.polyfit(x, y, 1)

# Print the findings
print('a =', coefficients[0])
print('b =', coefficients[1])

# Let's compute the values of the line
polynomial = np.poly1d(coefficients)
x_axis = np.linspace(0,9,100)
y_axis = polynomial(x_axis)

# ...and plot the points and the line
plt.plot(x_axis, y_axis)
plt.grid('on')
plt.show()


# In[119]:


# Function to find distance
# https://www.geeksforgeeks.org/perpendicular-distance-
# between-a-point-and-a-line-in-2-d/
def calc_distance(x1, y1, a, b, c):
  d = abs((a * x1 + b * y1 + c)) / (math.sqrt(a * a + b * b))
  return d

# (y1 – y2)x + (x2 – x1)y + (x1y2 – x2y1) = 0

a = dist_points_from_cluster_center[0] - dist_points_from_cluster_center[8]
b = K[8] - K[0]
c1 = K[0] * dist_points_from_cluster_center[8]
c2 = K[8] * dist_points_from_cluster_center[0]
c = c1 - c2


# In[120]:


dist_points_from_cluster_center


# In[121]:


distance_of_points_from_line = []
for k in range(9):
  distance_of_points_from_line.append(
      calc_distance(K[k], dist_points_from_cluster_center[k], a, b, c))


# In[122]:


distance_of_points_from_line


# In[123]:


plt.plot(K, distance_of_points_from_line)

