"""
Created on Mon May 27 13:47:04 2019

@author: mahsa lotfi
"""

import csv
import numpy as np
import tensorflow as tf
from matplotlib import pyplot as plt


cancer_data=[]
with open ('cancer_data.csv') as csvfile:
    data_csv = csv.reader(csvfile, delimiter=',')
    for row in data_csv:
        #print(row)#showing the samples in this dataset
        cancer_data.append(row) # convert the csv data to a list
        #print(cancer_data)
cancer_data=np.array(cancer_data)# convert the list to a Numpy Array
col=len(row)# this will show us 32
row=len(cancer_data)# this will show us 570
      
#There must be 569 samples and 30 features, we must delete the first two columns 
#(ID and labels like 'M:malignant' and 'B: Benign') and the first row that determines 
#the names of the features 

Data=np.zeros((569,30))
labels=np.zeros((569,1))
for i in range(row-1):
    C=cancer_data[i+1]
    if C[1]=='M':# we can also extract the Malignant (1) and Benign (0) status\label from the 2nd column
        labels[i]=1
    else:
        labels[i]=0
    for j in range(col-2):        
        Data[i][j]=C[j+2]# Thus data will have 569 * 30 dimensions
data_dim=Data.shape
n_row=data_dim[0]
n_col=data_dim[1]    
##parameters
n_iterations=100
init_learning_rate=0.01
eta_null=init_learning_rate
map_size_x=100#the dimensions of the final feature map
map_size_y=100
n_weights=map_size_x*map_size_y*n_col#each cell in the output is connected to all features in the input  
network_dimensions = np.array([5, 5])#neighbors in the feature map
sigma_null = max(network_dimensions[0], network_dimensions[1]) / 2#initial neighborhood radius
time_constant=n_iterations/np.log(sigma_null)#T_sigma in formulas

##Normalizing the data to have all the values in (0,1)
Data_normed=Data/Data.max(axis=0)#column normalized
##Initialization: Initializing the weights randomly
W=np.random.random_sample((map_size_x*map_size_y,n_col))#each cell in the feature map has a vector with the similar size to the input
init_W=W
sess=tf.Session()

def BMU(x,w):#This function computes the distance between single input sample and all the cells in the feature map to find the "best matching unit" (winner weight) 
    #inputs must be numpy arrays
    x_t=tf.convert_to_tensor(x)
    x_t=tf.to_float(x_t)
    x_tt=tf.expand_dims(x_t,0)
    w_t=tf.convert_to_tensor(w)
    w_t=tf.to_float(w_t)
    w_tt=tf.expand_dims(w_t,1)
    subb=tf.subtract(x_tt,w_tt)
    sq=tf.square(subb)
    distance=tf.reduce_sum(sq,2)#this shows the distance of every point from the rest of the points
    distance=tf.sqrt(distance)
    dist=sess.run(distance)
    ##find the minium distance and its index
    ind = np.unravel_index(np.argmin(dist, axis=None), dist.shape)
    return [ind, dist[ind]]
        
def find_index(ncol, index):
    IND=index[0]
    r=int(np.floor(IND/ncol)); #row of the minimum value in the feature map, if we assume that the vectorization was row-wise
    c=int(np.mod(IND,ncol));
    return [r,c]
       

def neighborhood_decrease(sigma_null, t, time_constant):
    Sigma=sigma_null * np.exp(-t / time_constant)
    return Sigma

def learning_rate_decrease(eta_null, t, n_iterations):
    eta=eta_null * np.exp(-t / n_iterations)
    return eta

def topological_neighborhood(distance_S, Sigma):
    return np.exp(-distance_S / (2* (Sigma**2)))


#Main part of the code
for t in range (n_iterations): #for each iteration
    print('iteration: ',t)
    for j in range (n_row):#for each sample of the dataset
        print('sample: ',j)
        s=Data_normed[j]#each sample has 30 features (columns)
        [index,min_distance]=BMU(Data_normed[j],W)
        [ro,co]=find_index(map_size_y, index)#we can find the 2Dindex of the minimum value
        # decrease the SOM parameters to modify the weights 
        Sigma = neighborhood_decrease(sigma_null, t, time_constant)
        Eta = learning_rate_decrease(eta_null, t, n_iterations)    
        # update the weights
        r=Sigma
        for k1 in range (map_size_x):
           for k2 in range (map_size_y):
               if abs(k1-ro)<=r and abs(k2-co)<=r:##finding the neighbors for the winner
                       indd=int(map_size_y*(k1-1)+k2)#the vectorized index of the weight
                       distance_S=np.sqrt(np.sum((np.array([k1,k2])-np.array([ro,co]))**2))
                       T=topological_neighborhood(distance_S, Sigma)
                       new_w = W[indd] + (Eta * T * (Data_normed[j]- W[indd]))
                       W[indd]=new_w
    
##Plot the SOM  
Weight_mat=np.zeros((map_size_x,map_size_y,30))                       
for k3 in range (len(W)):
    ro1=int(np.floor(k3/map_size_y)); #row of the minimum value in the feature map, if we assume that the vectorization was row-wise
    col1=int(np.mod(k3,map_size_y));
    Weight_mat[ro1,col1]=W[k3]
Mn=tf.reduce_mean(Weight_mat,axis=2)
Meann=sess.run(Mn)
plt.imshow(Meann, cmap='hot')  
plt.show() 
   
