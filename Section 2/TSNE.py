"""
TSNE using Scikit-learn library and the MNIST dataset
Created on Wed Mar  6 11:33:46 2019
@author: MAHSA
"""
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patheffects as PathEffects
import seaborn as sns
import tensorflow as tf
from tensorflow.python.framework import ops
from tensorflow.examples.tutorials.mnist import input_data
from sklearn.manifold import TSNE
import time


# Import MNIST
data_mnist = input_data.read_data_sets('input/data', one_hot=True)
# Shapes of training set
print("Training set (images) shape: {shape}".format(shape=data_mnist.train.images.shape))
print("Training set (labels) shape: {shape}".format(shape=data_mnist.train.labels.shape))

# Shapes of test set
print("Test set (images) shape: {shape}".format(shape=data_mnist.test.images.shape))
print("Test set (labels) shape: {shape}".format(shape=data_mnist.test.labels.shape))

# Create dictionary of target classes
label_dict = {
 0: '0',
 1: '1',
 2: '2',
 3: '3',
 4: '4',
 5: '5',
 6: '6',
 7: '7',
 8: '8',
 9: '9'
}
# Get 28x28 image
X=data_mnist.train.images
X.shape ##This shows the size of the dataset
Y=data_mnist.train.labels

sample_1 = data_mnist.train.images[49].reshape(28,28)
# Get corresponding integer label from one-hot encoded data
sample_label_1 = np.where(data_mnist.train.labels[49] == 1)[0][0]
# Plot sample
print("y = {label_index} ({label})".format(label_index=sample_label_1, label=label_dict[sample_label_1]))
plt.imshow(sample_1, cmap='Greys')

import seaborn as sns
sns.set_style('darkgrid')
sns.set_palette('muted')
sns.set_context("notebook", font_scale=1.5,
                rc={"lines.linewidth": 2.5})

def data_scatter(x, colors):
    # choose a color palette with seaborn.
    num_classes = len(np.unique(colors))
    palette = np.array(sns.color_palette("hls", num_classes))

    # create a scatter plot.
    f = plt.figure(figsize=(8, 8))
    ax = plt.subplot(aspect='equal')
    sc = ax.scatter(x[:,0], x[:,1], lw=0, s=40, c=palette[colors.astype(np.int)])
    plt.xlim(-25, 25)
    plt.ylim(-25, 25)
    ax.axis('off')
    ax.axis('tight')

    # add the labels for each digit corresponding to the label
    txts = []

    for i in range(num_classes):

        # Position of each label at median of data points.

        xtext, ytext = np.median(x[colors == i, :], axis=0)
        txt = ax.text(xtext, ytext, str(i), fontsize=24)
        txt.set_path_effects([
            PathEffects.Stroke(linewidth=5, foreground="w"),
            PathEffects.Normal()])
        txts.append(txt)

    return f, ax, sc, txts

x_subset = X[0:1000] ##This gives us a subset of the dataset
y_subset = Y[0:1000]

##Finding the indices of one-hpt vector labels
def one_hot2num(labels):###The input is a multi-dimensional numpy array
    sample_number=len(labels)
    Labels=np.zeros(sample_number)
    print('This shows what a one-hot vector looks like:',labels[0])##You can show one sample
    print('This hsows the length of the one-hot vector',len(labels[0]))##This shows the length of the one-hot vector
    for i in range (sample_number):
        L=labels[i]
        LL=L
       # LL=[int(j) for j in L]##cast the values in the labels vector into the integer
        INDEX=np.where(LL==1)
        print(INDEX)
        Labels[i]=INDEX[0]###To change a one-element array to an integer
    return Labels    

y_subset=np.array(y_subset)            
Y_indices=one_hot2num(y_subset)
time_start = time.time()
mnist_tsne = TSNE(random_state=123).fit_transform(x_subset)

print('t-SNE done! Time elapsed: {} seconds'.format(time.time()-time_start))

data_scatter(mnist_tsne, Y_indices)

