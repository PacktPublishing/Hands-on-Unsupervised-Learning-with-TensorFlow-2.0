"""
Created on Wed May  1 22:22:54 2019
Three layer Auto-encoder
@author: mahsa lotfi
"""

#Importing the required libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
import tensorflow.examples.tutorials.mnist.input_data as input_data
mnist_data=input_data.read_data_sets("MNIST")

#checking the shapes of the data
print('The shape of the Fashion MNIST training data is:\t',mnist_data.train.images.shape)
print('The shape of the Fashion MNIST testing data is:\t',mnist_data.test.images.shape)
sample_size=mnist_data.train.images.shape[0]

#Displaying some of the samples
x1=mnist_data.train.images[100].reshape(28,28)
x2=mnist_data.train.images[360].reshape(28,28)
plt.figure(1)
plt.imshow(x1,cmap='Greys')
plt.figure(2)
plt.imshow(x2,cmap='Greys')

#let's make the three layer Auto-encoder model
input_size=784#images are 28*28
hid_size=196
output_size=784
act_func=tf.nn.relu#Defining the activation function
initializer=tf.variance_scaling_initializer()
init=tf.global_variables_initializer()

AE_inputs=tf.placeholder(tf.float32,shape=[None,input_size])#the inputs of the AE are te images of size 784 bas a vector or 28*28 as a matrix
weights=tf.Variable(initializer([input_size,hid_size]),dtype=tf.float32)
weights_2=tf.Variable(initializer([hid_size, output_size]),dtype=tf.float32)
bias=tf.Variable(tf.zeros(hid_size))
bias2=tf.Variable(tf.zeros(input_size))

#Hidden layer
mult=tf.matmul(AE_inputs, weights)
hid=act_func(mult+bias)

#Output layer
mult2=tf.matmul(hid,weights_2)
output=act_func(mult2+bias2)

#Back-propagation and predicting the weights
learning_rate=0.01
loss=tf.reduce_mean(tf.square(output-AE_inputs))#computing the mean square error as the loss function
optimizer=tf.train.AdamOptimizer(learning_rate)
train=optimizer.minimize(loss)
init=tf.global_variables_initializer()

#Running the algorithm on different data batches
epoch_n=10
batch_size=100
num_test=10
batch_num=sample_size/batch_size
print(batch_num)
with tf.Session() as sess:
    sess.run(init)
    for epoch in range(epoch_n):
      for i in range (int(batch_num)):
        X_b,y_b=mnist_data.train.next_batch(batch_size)
        sess.run(train,feed_dict={AE_inputs:X_b})
        train_loss=loss.eval(feed_dict={AE_inputs:X_b})
        print("The training loss is:\t",train_loss,"\n")
    results=output.eval(feed_dict={AE_inputs:mnist_data.test.images[:num_test]})
    results2=hid.eval(feed_dict={AE_inputs:mnist_data.test.images[:num_test]})

f,a=plt.subplots(2,10,figsize=(20,4))
for i in range(num_test):
  a[0][i].imshow(np.reshape(mnist_data.test.images[i],(28,28)),cmap='Greys')
  a[1][i].imshow(np.reshape(results[i],(28,28)),cmap='Greys')
g,b=plt.subplots(1,10,figsize=(10,4),squeeze=False)
for i in range(num_test):
  b[0][i].imshow(np.reshape(results2[i],(14,14)),cmap='Greys')    