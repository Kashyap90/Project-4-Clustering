
# coding: utf-8

# In[1]:


# Import libraries into working environment:


# In[2]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn import decomposition
from sklearn import datasets
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
get_ipython().run_line_magic('matplotlib', 'inline')


# In[3]:


# Read input data into a Dataframe:


# In[4]:


df = pd.read_csv('C:/Users/kashyap/Downloads/data_stocks.csv')
df1 = df.copy()
print(df.shape)
df.head()


# In[5]:


# Feature Scaling:


# In[6]:


from sklearn.preprocessing import StandardScaler
features = df.values
sc = StandardScaler()
X_scaled = sc.fit_transform(features)
print('Shape of Scaled features : ')
print('---------------------------------------------------------------------')
print(X_scaled.shape)


# In[7]:


# Determining optimal number of components for PCA looking at the explained variance as a function of the components:


# In[8]:


sns.set()
sns.set_style('whitegrid')
pca = PCA().fit(X_scaled)
plt.plot(np.cumsum(pca.explained_variance_ratio_))
plt.xlabel('number of components')
plt.ylabel('cumulative explained variance')
plt.show()


# In[9]:


#Note:

#Here we see that we'd need about 100 components to retain 100% of the variance.Looking at this plot for a high-dimensional 
#dataset can help us understand the level of redundancy present in multiple observations


# In[10]:


# Apply PCA to reduce the number of dimensions from 502 to 2 dimensions for better data visualization.


# In[11]:


pca = PCA(n_components=2)
pca.fit(X_scaled)
print('explained variance :')
print('--------------------------------------------------------------------')
print(pca.explained_variance_)
print('--------------------------------------------------------------------')
print('PCA Components : ')
print('--------------------------------------------------------------------')
print(pca.components_)
print('--------------------------------------------------------------------')
X_transformed = pca.transform(X_scaled)
print('Transformed Feature values first five rows :')
print('--------------------------------------------------------------------')
print(X_transformed[:5,:])
print('--------------------------------------------------------------------')
print('Transformed Feature shape :')
print('--------------------------------------------------------------------')
print(X_transformed.shape)
print('--------------------------------------------------------------------')
print('Original Feature shape :')
print('--------------------------------------------------------------------')
print(X_scaled.shape)
print('--------------------------------------------------------------------')
print('Restransformed Feature shape :')
print('--------------------------------------------------------------------')
X_retransformed = pca.inverse_transform(X_transformed)
print(X_retransformed.shape)
print('--------------------------------------------------------------------')
print('Retransformed Feature values first five rows :')
print('--------------------------------------------------------------------')
print(X_retransformed[:5,:])
print('--------------------------------------------------------------------')


# In[12]:


# Problem 1:

# There are various stocks for which we have collected a data set, which all stocks are apparently similar in performance.


# In[13]:


# Finding optimum number of clusters for KMEANS cluster:


# In[14]:


wcss=[]
for i in range(1, 21):
    kmeans = KMeans(n_clusters = i, init = 'k-means++', random_state = 0)
    kmeans.fit(X_transformed)
    wcss.append(kmeans.inertia_)
plt.plot(range(1, 21), wcss)
plt.title('The Elbow Method')
plt.xlabel('Number of clusters')
plt.ylabel('Mean Squared Errors')
plt.show()


# In[15]:


import scikitplot

##scikitplot.cluster.plot_elbow_curve(KMeans(),X_transformed,cluster_ranges=range(1,20))


# In[16]:


# Note :
# Optimum number of cluster from the elbow method is determined to be 5.


# In[17]:


# Applying K-Means Clustering to find stocks which are similar in performance:


# In[18]:


k_means = KMeans(n_clusters=5,random_state=0,init='k-means++')
k_means.fit(X_transformed)
y_kmeans = kmeans.fit_predict(X_transformed)
labels = k_means.labels_
print("labels generated :\n",labels)


# In[19]:


len(labels)


# In[20]:


# Visualising the clusters:


# In[21]:


plt.scatter(X_transformed[y_kmeans == 0, 0], X_transformed[y_kmeans == 0, 1], s = 100, c = 'red', label = 'Cluster 1')
plt.scatter(X_transformed[y_kmeans == 1, 0], X_transformed[y_kmeans == 1, 1], s = 100, c = 'blue', label = 'Cluster 2')
plt.scatter(X_transformed[y_kmeans == 2, 0], X_transformed[y_kmeans == 2, 1], s = 100, c = 'green', label = 'Cluster 3')
plt.scatter(X_transformed[y_kmeans == 3, 0], X_transformed[y_kmeans == 3, 1], s = 100, c = 'cyan', label = 'Cluster 4')
plt.scatter(X_transformed[y_kmeans == 4, 0], X_transformed[y_kmeans == 4, 1], s = 100, c = 'magenta', label = 'Cluster 5')
#plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], s = 300, c = 'yellow', label = 'Centroids')
plt.title('Clusters of stocks')
plt.xlabel('Principal Component (1)')
plt.ylabel('Principal Component (2)')
plt.legend()
plt.show()


# In[22]:


# Note :

#The above 5 clusters shows the stocks which are similar in stock performance.


# In[23]:


# Problem 2:

#How many Unique patterns that exist in the historical stock data set, based on fluctuations in price.


# In[24]:


df_comp = pd.DataFrame(pca.components_,columns=df1.columns)
df_comp.head()


# In[25]:


sns.set_style('whitegrid')
sns.heatmap(df_comp)


# In[26]:


plt.figure(figsize=(11,8))
df_corr = df1.corr().abs()
sns.heatmap(df_corr,annot=True)


# In[27]:


# Problem 3:


# In[28]:


# Identify which all stocks are moving together and which all stocks are different from each other.


# In[32]:


df['labels'] = labels


# In[33]:


df.head()


# In[34]:


df['labels'].unique().tolist()


# In[35]:


for i in df['labels'].unique().tolist():
    count = df[df['labels'] == i].shape[0]
    print('\nFor label {} the number of similar stock performance is : {}'.format(i, count))


# In[36]:


# Fitting Hierarchical Clustering to the dataset:
from sklearn.cluster import SpectralClustering
hc = SpectralClustering(n_clusters = 5, affinity = 'nearest_neighbors')
hc.fit(X_transformed)


# In[37]:


hc.fit_predict(X_transformed)


# In[38]:


y_labels = hc.labels_


# In[39]:


len(y_labels),np.unique(y_labels)


# In[40]:


# Visualising the clusters:

X = X_transformed
plt.scatter(X[y_labels == 0, 0], X[y_labels == 0, 1], s = 100, c = 'red', label = 'Cluster 1')
plt.scatter(X[y_labels == 1, 0], X[y_labels == 1, 1], s = 100, c = 'blue', label = 'Cluster 2')
plt.scatter(X[y_labels == 2, 0], X[y_labels == 2, 1], s = 100, c = 'green', label = 'Cluster 3')
plt.scatter(X[y_labels == 3, 0], X[y_labels == 3, 1], s = 100, c = 'cyan', label = 'Cluster 4')
plt.scatter(X[y_labels == 4, 0], X[y_labels == 4, 1], s = 100, c = 'magenta', label = 'Cluster 5')
plt.title('Clusters of customers')
plt.xlabel('Annual Income (k$)')
plt.ylabel('Spending Score (1-100)')
plt.legend()
plt.show()


# In[41]:


df1.columns


# In[42]:


df2 = df1.copy()
df2['labels'] = y_labels
for i in df2['labels'].unique().tolist():
    count = df2[df2['labels'] == i].shape[0]
    print('\nFor label {} the number of similar stock performances is: {}' .format(i, count))


# In[43]:


# Advantages - Hierarchical Clustering

#1) No apriori information about the number of clusters required.

#2) Easy to implement and gives best result in some cases.


# In[44]:


# Disadvantages - Hierarchical Clustering

#1) Algorithm can never undo what was done previously.

#2) Time complexity of at least O(n2 log n) is required, where ‘n’ is the number of data points.

#3) Based on the type of distance matrix chosen for merging different algorithms can suffer with one or more of the following:

#i) Sensitivity to noise and outliers

#ii) Breaking large clusters

#iii) Difficulty handling different sized clusters and convex shapes

#4) No objective function is directly minimized


# In[ ]:


#CONCLUSION:
#For the given data set KMeans Clustering creates a better and distinct clustering compared to Spectral Clustering.

