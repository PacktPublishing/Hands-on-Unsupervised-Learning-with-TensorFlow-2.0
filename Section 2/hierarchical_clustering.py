"""
THIS CODE MUST BE COPIED TO GOOGLE COLAB. FOR RUNNING, DO NOT RUN IT ON PYTHON IDEs
author: Mahsa Lotfi
"""

#Importing the useful librarues
from scipy.cluster.hierarchy import dendrogram, linkage  
from matplotlib import pyplot as plt
import tensorflow as tf
import pandas as pd  
import numpy as np 


#Generate a random dataset and plot it
x=np.array([[6,2],[11,15],[14,11],[27,13],[30,30],[88,77],[71,84],[61,74],[74,52],[81,94]])
for i in range(len(x)):
  X=x[i][0]
  Y=x[i][1]
  plt.scatter(X,Y)
  plt.xlabel('x axis')
  plt.ylabel('y axis')
  plt.title('The raw dataset')
Labels=range(1,11)#Labeling the points  
  
#Let's plot the dendrogram for our data points, we must use Scipy Library

from scipy.cluster.hierarchy import dendrogram, linkage  
from matplotlib import pyplot as plt

linked = linkage(x, 'single')#Determine whether this is a single_linkage, complete_linkage or average clustering

labelList = range(1, 11)

plt.figure(figsize=(10, 7))  
dendrogram(linked,  
            orientation='top',
            labels=labelList,
            distance_sort='descending',
            show_leaf_counts=True)
plt.xlabel('point labels')
plt.ylabel('The distance and the cluster tress')
plt.show()

from sklearn.cluster import AgglomerativeClustering
cluster = AgglomerativeClustering(n_clusters=4, affinity='euclidean', linkage='single')#You can easily change the number of clusters to change the horizontal threshold in the dendrogram 
cluster_lables=cluster.fit_predict(x) 

plt.scatter(x[:,0],x[:,1], c=cluster.labels_, cmap='rainbow')  


#Example 2: Let's solve a more real problem, we will load a data set that has two properties of the shoppers, anual income and the spending score
#For loading the CSV file we need to import pandas 

from google.colab import files
uploaded = files.upload()
import io
Data = pd.read_csv(io.BytesIO(uploaded['dataset.csv']))
Data

from scipy.cluster.hierarchy import dendrogram, linkage
Data.shape #to make sure that the labels are not included in the dimensions
Link= linkage(Data, 'single')

labelList = range(1, 201)

plt.figure(figsize=(10, 7))  
dendrogram(Link,  
            orientation='top',
            labels=labelList,
            distance_sort='descending',
            show_leaf_counts=True)
plt.xlabel('point labels')
plt.ylabel('The distance and the cluster tress')
plt.title('Hierarchical Clustering Using Single-Linkage Method')
plt.show()

plt.figure(figsize=(10,7))
dend = dendrogram(linkage(Data, method='ward')) 
plt.xlabel("point labels")
plt.ylabel("The distance and the cluster trees")
plt.title("Hierarchical Clustering Using Ward method")
plt.show

from sklearn.cluster import AgglomerativeClustering
ClusterModel = AgglomerativeClustering(n_clusters=6, affinity='euclidean', linkage='ward')#you can change the number of clusters here 
cluster_lables=ClusterModel.fit_predict(Data) 
data_array=Data.as_matrix(columns=None)
data_x=data_array[:,0]
data_y=data_array[:,1]
plt.scatter(data_x,data_y,c=cluster_lables,cmap='rainbow')

