#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 27 22:09:04 2019

@author: mahsa
"""

import numpy as np
import tensorflow as tf
from sklearn.datasets import load_iris
import matplotlib.pyplot as plt
import seaborn as sns

### Loading the iris dataset
total_data = load_iris()
X=total_data.data
N=X.shape[0]##number of samples
K=X.shape[1]##number of features
print("The number of samples is:",N) 
print("The number of features is:", K)
Data=np.zeros((N,K))

def TF_Zscore(X):
  Mean=tf.math.reduce_mean(X,0)##column-wise mean, mean for each feature
  Std=tf.math.reduce_std(X,0)
  sess = tf.InteractiveSession()  
  for i in range (0,K):
    for j in range (0,N):
      a=(X[j,i]-Mean[i])/Std[i]
      a=sess.run(a)##changing the tensor to numpy array
      Data[j,i]=a
  sess.close()    
  return Data 

###Finding the covariance matrix
X_n=TF_Zscore(X)##normalized data
sess2 = tf.InteractiveSession()
Cov_mat=tf.matmul(tf.transpose(X_n),X_n)## Covariance matrix has the dimensionality K*K, 4*4 here in this example
#Cov_mat=sess2.run(Cov_mat)
Sigma, Eig_u, Eig_v=tf.svd(Cov_mat)
Sigma=sess2.run(Sigma)#numpy arrays
Eig_u=sess2.run(Eig_u)
Eig_v=sess2.run(Eig_v)
ind=np.argsort(-Sigma)#this gives the indices of the sorting in descending order
##Find out if the eigen values are equal and repetetive or not, if yes eliminate them since not orthogonal
for j in range (len(Sigma)):
    U=np.where(Sigma[j]==Sigma)
    L=len(U)
    if L!=1:#if we have eigenvalues that are equal, find their indices
        INDEX=U
    else:
        INDEX=[]
        
###eliminate the repetetive eigen values and form the modified Sigma matrix
Sigma_m=np.delete(Sigma, INDEX)
IND_m=np.delete(ind, INDEX)
Eig_u_m=np.delete(Eig_u, (INDEX), axis=0)##we must delete the eigenvectors corresponding to repetetive eigenvalues        
###Now we can choose "p" principal components (P), here I choose the 2 most influential features
p=2
IND=list(IND_m[0:p])
P=Eig_u_m[IND]## principal components matrix P has the following dimensions p*K
Y=tf.matmul(P, tf.transpose(X))
Y=tf.transpose(Y)###this new representation has the dimensionality of N*p
Y=sess2.run(Y)
