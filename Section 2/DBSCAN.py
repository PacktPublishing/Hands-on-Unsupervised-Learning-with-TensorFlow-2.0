#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar 28 10:55:49 2019

@author: mahsa
"""
#importing the required libraries
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

#Let's generate some points on the 1/4 th of the perimeter of a circle
radius=[2,3,4,5]
type(radius)
len(radius)
y=np.zeros([len(radius),200])

radius=[2,3,4,5]
y=np.zeros([len(radius),200])
domain=np.linspace(0,5,200)
k=0
Y=[]
for r in radius:
  j=0
  for i in domain:  
    z=np.sqrt(r**2-i**2)
    y[k,j]=z
    j=j+1
  k=k+1
  
for i in range(0,k):
  plt.scatter(domain, y[i,:])    
plt.xlabel('x coordinate')
plt.ylabel('y coordinate')
plt.title('Dataset')

def Neighborhood(x,y,X,Y,min_pts):#X includes all the points but x is just one sample
  sh1=X.shape
  l_x=sh1[1]##this shows 800 in this example
  cluster_x=[]
  cluster_y=[]
  for i in range (0,l_x):
        x1=np.absolute(x-X[0,i])
        y1=np.absolute(y-Y[0,i])
        if (x1 <= min_pts and y1 <= min_pts):
          cluster_x.append(i)#indices of the neighbors
          cluster_y.append(i)       
  return (cluster_x , cluster_y)

###Testing the Neighbourhood function
#x=1
#y=1
#X=np.array([1.5,2.5,3,0.5])
#Y=np.array([1,2.5,1.5,0.5])
#X=X.reshape(1,len(X))
#Y=Y.reshape(1,len(Y))
#min_pts=0.7
#sh1=X.shape
#l_x=sh1[1]##this shows 800 in this example
#cluster_x=[]
#cluster_y=[]
#for i in range (0,l_x):
        #x1=np.absolute(x-X[0,i])
        #y1=np.absolute(y-Y[0,i])
        #print("index:",i,x1)
        #print("index:",i,y1)
        #if (x1 <= min_pts and y1 <= min_pts):
          #cluster_x.append(i)#indices of the neighbors
          #cluster_y.append(i) 
#print(cluster_x, cluster_y)

min_pts=0.3#the radius to find the neighbourhood of the points
#Let's re-write the "x" values such that they are detremined according to the "y" values
X=np.array([domain,domain,domain,domain])
X=np.reshape(X, (1,4*200))#X and Y would be a one-dimensional array, they include the coordinates of all points
Y=np.reshape(y, (1,4*200))

sh1=X.shape
l_x=sh1[1]
sh2=Y.shape
l_y=sh2[1]
Labels=np.zeros(4*200)##Save the cluster labels here
Labels[0]=1##the first point we start with should belong to cluster number 1
K=1##Cluster count

for i in range (0,l_x):
    [x2,y2]=Neighborhood(X[0,i],Y[0,i],X,Y,min_pts)
    if i==0:#The cluster label of the first point is chosen as 1
      Labels[x2]=1
      print("first point assigned to group 1")
    else:
      
      if Labels[i] !=0:
          Labels[x2]=Labels[i]
          print("already assigned to a group\t","Number of neighbors:",len(x2),"index:",i, "cluster is:",Labels[i])
      else:
          if len(x2) !=0:##If it is not an outlier
              K=K+1
              Labels[i]=K
              Labels[x2]=K
              print("New point, not an outlier","Number of neighbors:",len(x2), "index:",i, "cluster is:",Labels[i])
          else:#This means that the point is an outlier with no neighboring points so we can call the class with a big number 1000
              Labels[i]=1000
              print("New point, outlier","Number of neighbors:",len(x2), "index:",i, "cluster is:",Labels[i])

              
print(Labels)
             
