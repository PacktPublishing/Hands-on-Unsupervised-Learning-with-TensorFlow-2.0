#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar 28 11:13:04 2019

@author: mahsa
"""

import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

#sess=tf.InteractiveSession()
##Generating the points
sample_n=200
features_n=2
iteration=100
cluster_n=3
points=np.random.uniform(0,20,(sample_n, features_n))
samples=tf.constant(points)
initial_centroids=tf.slice(tf.random_shuffle(samples),[0,0],[cluster_n,-1])
initial_centroids[0]

x=samples[0]##This shows how we can access the first point in the dataset
S=x[1]

##Plotting the points and the centroids
X=points[:,0]
Y=points[:,1]
plt.scatter(X,Y)
##Now plot the initial centroids
sess = tf.Session(); 
with sess.as_default(): 
  cent=initial_centroids.eval()
  X1=cent[:,0]
  Y1=cent[:,1]
  plt.scatter(X1,Y1, s=100, color='red',marker=(5, 2))
  plt.xlabel('1st Feature')
  plt.ylabel('2nd Feature')
  plt.title('Dataset points and the initial centroids')
  plt.show()
  
  ##We need to use the feature of broadcasting to find the distances between the points and the centroids
##The dimesnions of two tensors mut be equal or one of the dimensions must be 1 so we set dimensions to 1 using exapnd_dims
## Let's check the shapes first
print('The sample shape is like:',samples.shape)
print('The shape of the centroids is like:',initial_centroids.shape)
samples_exp=tf.expand_dims(samples,0)
centroids_exp=tf.expand_dims(initial_centroids,1)
print('The expanded sample shape is like:',samples_exp.shape)
print('The expanded shape of the centroids is like:',centroids_exp.shape)

##Subtracting the points and the centroids
samples_exp.dtype
subb=tf.subtract(centroids_exp,samples_exp)
print(subb.shape)
sq=tf.square(subb)
print(sq.shape) ##3,200,2
distance=tf.reduce_sum(sq,2)
print(distance.shape)
IND1=tf.argmin(distance,0)
print(IND1.shape)

def update_centroids(samples, IND, cluster_n):
    # Updates the centroid to be the mean of all samples associated with it.
    IND2=tf.cast(IND,'int32')
    partitions = tf.dynamic_partition(samples, IND2,cluster_n)#this partitions the data according to the dimensions in IND1
    new_centroids = tf.concat([tf.expand_dims(tf.reduce_mean(partition, 0), 0) for partition in partitions], 0)
    return new_centroids

iteration=1
while(True): 
  
  if iteration==1:
    centroids_exp=tf.expand_dims(initial_centroids,1)
  else:
    centroids_exp=tf.expand_dims(Centroids,1)
    
  subb=tf.subtract(centroids_exp,samples_exp)
  sq=tf.square(subb)
  distance=tf.reduce_sum(sq,2)
  IND1=tf.argmin(distance,0)
  updated_centroids=update_centroids(samples,IND1,cluster_n)
  model = tf.global_variables_initializer()
  with tf.Session() as session:
        sample_values = session.run(samples)
        updated_centroid_value = session.run(updated_centroids)
        print(updated_centroid_value)
      
  Centroid_prev=tf.squeeze(centroids_exp) 
  diff0=tf.subtract(Centroid_prev[0], updated_centroid_value[0])  
  Diff0=tf.norm(diff0,ord='euclidean')
  
  diff1=tf.subtract(Centroid_prev[1], updated_centroid_value[1])  
  Diff1=tf.norm(diff1,ord='euclidean')
  
  diff2=tf.subtract(Centroid_prev[2], updated_centroid_value[2])  
  Diff2=tf.norm(diff2,ord='euclidean')
  
  with sess.as_default():
    
    Diff0=Diff0.eval()
    Diff1=Diff1.eval()
    Diff2=Diff2.eval()
  
    Diff=Diff0+Diff1+Diff2
    print(Diff)
    if Diff<10**(-1):
        Index=IND1;
        break;
    else:
        Centroids=updated_centroid_value
        iteration=iteration+1
print('The Total number of iterations is:',iteration)
 

