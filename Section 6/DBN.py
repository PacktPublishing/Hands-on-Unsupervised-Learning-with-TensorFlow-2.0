## Citing the code
#BibTex reference format:

#        @misc{DBNAlbert,
#        title={A Python implementation of Deep Belief Networks built upon NumPy and TensorFlow with scikit-learn compatibility},
#        author={albertbup},
#        year={2017}}

import pickle
import matplotlib.pyplot as plt
from sklearn import linear_model, metrics
from sklearn.pipeline import Pipeline
from dbn.models import UnsupervisedDBN 
import tensorflow.examples.tutorials.mnist.input_data as input_data

##Load Data
mnist_data=input_data.read_data_sets("MNIST")
X_train=mnist_data.train.images
X_test=mnist_data.test.images
Y_train=mnist_data.train.labels
Y_test=mnist_data.test.labels
plt.imshow(X_train[200].reshape(28,28))
n_epochs_rbm = 100  
 
logistic_inverse_reg = 50.0  #regularization strength, the smaller, the stronger
logistic_inverse_reg_2 = 1   

##Models we will use
logistic = linear_model.LogisticRegression(solver='newton-cg',multi_class='auto',C=logistic_inverse_reg)#logistic regression 
dbn = UnsupervisedDBN(hidden_layers_structure=[256, 512],batch_size=100,learning_rate_rbm=0.01,
                      n_epochs_rbm=n_epochs_rbm,
                      activation_function='sigmoid')

classifier = Pipeline(steps=[('dbn', dbn),('logistic', logistic)])

##Training RBM-logistic pipeline
classifier.fit(X_train, Y_train)

##Training logistic regression
logistic_classifier = linear_model.LogisticRegression(solver='newton-cg',multi_class='auto',C=logistic_inverse_reg_2)
logistic_classifier.fit(X_train, Y_train)

##Save model
with open('logistic.pkl', 'wb') as wf:
    pickle.dump(classifier, wf)

##Evaluation
print("\nreport of the evaluation:\n%s\n" % (metrics.classification_report(Y_test,classifier.predict(X_test))))
