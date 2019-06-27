"""
Anomaly Detcetion using LOF method and Boston Housing dataset
@author: MAHSA
"""
#importing the required libraries
from sklearn.datasets import load_boston
import tensorflow as tf
import numpy as np

##loading the boston housing dataset using the Scikit learn library
boston = load_boston()
data=boston.data
print(boston.data.shape)
print(data[1])#show one sampel of the dataset

##Number of samples and features
dim=data.shape
sample_num=dim[0]
feature_num=dim[1]
print("number of sampels is:",sample_num, "\nnumber of features is:",feature_num)
print(type(data))##we would like to change this type from a numpy array to a tensor

data_tf1=tf.convert_to_tensor(data)
data_tf2=tf.convert_to_tensor(data)

##let's use the broadcasting property of the subtract function in tensorflow to find the subtraction of points with each other
data1=tf.expand_dims(data_tf1,0)
data2=tf.expand_dims(data_tf2,1)
print('The expanded data template 1 is like:',data1.shape)
print('The expanded data template 1 is like:',data2.shape)
subb=tf.subtract(data1,data2)#subtract data
sq=tf.square(subb)
distance=tf.reduce_sum(sq,2)#this shows the distance of every point from the rest of the points
print(distance.shape)
distance=tf.sqrt(distance)

distance_sorted=tf.sort(distance,axis=1,direction='ASCENDING',name=None)##we are sorting each row of the distance matrix 
distance_sorted_ind=tf.argsort(distance,axis=1,direction='ASCENDING',name=None)##we are retriving the indices of the sort
k=5##consider "k" as the value which determines the "k_th" neighbor, we can choose this value 

##Now let's find the k_distance for each point which will lie at the k+1 location
k_distance_vec=np.zeros((sample_num,1))
N_k_len=np.zeros((sample_num,1))
for i in range(sample_num):
  dis=distance_sorted[i]
  dis_np=tf.Session().run(dis)#change the tensor value to a numpy array
  dis_set=np.unique(dis_np)#remove the repetitions
  k_distance_vec[i,0]=dis_set[k]#the "k_distance" vector in the form of a numpy array
  N_k=np.where(dis_np<=dis_set[k])#the neighborhood with distance less than or equal to the k_distance
  sz=dis_np[N_k].shape
  N_k_len[i]=sz[0]-1##This shows the number of neighboring points with distance less than or equal to k_distance, remove self-distance

summ=0
Ird_k=[]
inv_Ird_k=[]

Dist=tf.Session().run(distance)
Dist_ind=tf.Session().run(distance_sorted_ind)
print('sample number\n',sample_num)


for i in range(sample_num):#for each point A
  var1=distance_sorted_ind[i]
  var2=tf.Session().run(var1[1:int(N_k_len[i])])#get the value of the indices in the neighborhood

  for i1 in range(len(var2)):
      var2[i1]=int(var2[i1])
  for j in var2:#for each point B with this index in the neighborhood
      L=k_distance_vec[j,0]
      reach=[Dist[i,j],L]
      reach_dis=np.max(reach)#computing the reachability distance betwenn A and all its neighbors
      summ=summ+reach_dis
      
  Val=np.round((summ/N_k_len[i])**(-1),5)  
  Ird_k=np.append(Ird_k, Val)
  inv_Ird_k=np.append(inv_Ird_k,1/Val)
  summ=0
  print('length of Ird_k',len(Ird_k),',sample number',i)
  print(Ird_k)
  
summ_b=0
lof_k=[]
for i2 in range(sample_num):#for each point A
  var1=distance_sorted_ind[i2]
  var2=tf.Session().run(var1[1:int(N_k_len[i2])])#get the value of the indices in the neighborhood
  for j2 in var2:#for each point B with this index in the neighborhood
        summ_b=summ_b+Ird_k[j2]
  lof_k=np.append(lof_k,summ_b/(N_k_len[i2]*Ird_k[i2]) )
  summ_b=0
  
  print(lof_k)
  
#Detecting the outliers, if lof_k_A>1, then it is considered as an outlier
K=0
outlier=[]
for i3 in range(sample_num):  
  if (lof_k[i3]>1):
    outlier=np.append(outlier,i3)
print('These samples are outliers:\n',outlier)
print('The number of the outliers is', len(outlier),' out of 506 samples when we choose k=',k,' for the k-distance')
