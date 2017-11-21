
# coding: utf-8

# In[2]:


from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from sklearn.model_selection import train_test_split
import gzip
import os
import sys
import time
import imageio
from six.moves import urllib
from six.moves import xrange  # pylint: disable=redefined-builtin
from scipy.misc import imsave
import tensorflow as tf
import numpy as np
import csv
import pickle
import pandas as pd


# In[3]:


#Function to transform data into the format required
#Takes data of the form [60000 x 28 x 28 x 1] and returns [60000 x 784]
def transform_data(data):
    a = np.reshape(data,newshape=(data.shape[0],data.shape[1]*data.shape[1]))
    return a
def load_mnist_data(filename = 'mnist_data.pkl',standardize = True):
    mnist_data = pickle.load(open('mnist_data.pkl','rb'))
    train_data = transform_data(mnist_data["train_data"])
    train_labels = mnist_data["train_labels"].reshape(mnist_data["train_labels"].shape[0],-1)
    test_data = transform_data(mnist_data["test_data"])
    test_labels = mnist_data["test_labels"].reshape(mnist_data["test_labels"].shape[0],-1)
    #Form validation set
    train_data, valid_data, train_labels, valid_labels = train_test_split(train_data, 
                                                                          train_labels, 
                                                                          test_size=0.16, 
                                                                          random_state=42)
    #Standardize the data - divide by 255
    if standardize:
        train_data, valid_data, test_data = train_data/255., valid_data/255., test_data/255.
    train_data, valid_data, test_data = train_data.T, valid_data.T, test_data.T
    
    return train_data,train_labels,valid_data,valid_labels, test_data, test_labels
train_data,train_labels,valid_data, valid_labels, test_data,test_labels = load_mnist_data()

#One hot encode the labels
train_labels = np.array(pd.get_dummies(train_labels.flatten())).T
valid_labels = np.array(pd.get_dummies(valid_labels.flatten())).T
test_labels = np.array(pd.get_dummies(test_labels.flatten())).T


print("Data loaded.")
print("Shapes of data are as follows: ")
print("Train data shape: "+ str(train_data.shape))
print("Train labels shape: "+ str(train_labels.shape))
print("Test data shape: "+ str(test_data.shape))
print("Test labels shape: "+ str(test_labels.shape))
print("Validation data shape: "+ str(valid_data.shape))
print("Validation labels shape: "+ str(valid_labels.shape))


# In[12]:


def relu(z):
    return np.maximum(z, 0)


# In[13]:


def softmax(output_array):
    logits_exp = np.exp(output_array)
    return logits_exp / np.sum(logits_exp, axis = 1, keepdims = True)


# In[23]:


def cross_entropy_softmax(softmax_array, y):
    inds = np.argmax(y, axis = 1).astype(int)
    pred_proba = softmax_array[np.arange(len(softmax_array)), inds]
    log_preds = np.log(pred_proba)
    loss = -1.0 * np.sum(log_preds) / len(log_preds)
    return loss


# In[29]:


def regularizeL2(reg_lambda, weight1, weight2):
    weight1_loss = 0.5 * reg_lambda * np.sum(weight1 * weight1)
    weight2_loss = 0.5 * reg_lambda * np.sum(weight2 * weight2)
    return weight1_loss + weight2_loss


# In[ ]:



def accuracy(probs, test_labels):
    result = np.zeros([len(test_labels),1])
    for i in range(len(test_labels)):
        if np.argmax(probs[i,:]) == np.argmax(test_labels[i,:]):
            result[i] = 1
    return np.mean(result) * 100

x_train = train_data
y_train = train_labels

hidden_nodes = 5
num_labels = y_train.shape[1]
num_features = x_train.shape[1]
learning_rate = .01
reg_lambda = .01


w1 = np.random.normal(0, 1, [num_features, hidden_nodes]) 
w2 = np.random.normal(0, 1, [hidden_nodes, num_labels]) 

b1 = np.zeros((1, hidden_nodes))
b2 = np.zeros((1, num_labels))

#Initialize lists to store the validation accuracy,
#test accuracy and loss after every 50 epochs
valid_accs = []
test_accs = []
losses = []
train_accs = []

epochs = 5001
for epoch in range(epochs):

    inputL = np.dot(x_train, w1)
    hiddenL = relu(inputL + b1)
    outputL = np.dot(hiddenL, w2) + b2
    output_probs = softmax(outputL)
    
    loss = cross_entropy_softmax(output_probs, y_train)
    #Add regularization
    loss += regularizeL2(reg_lambda, w1, w2)

    output_delta = (output_probs - y_train) / output_probs.shape[0]
    
    hidden_delta = np.dot(output_delta, w2.T) 
    hidden_delta[hiddenL <= 0] = 0
    
    dw2 = np.dot(hiddenL.T, output_delta)
    db2 = np.sum(output_delta, axis = 0, keepdims = True)
    
    dw1 = np.dot(x_train.T, hidden_delta)
    db1 = np.sum(hidden_delta, axis = 0, keepdims = True)

    dw2 += reg_lambda * w2
    dw1 += reg_lambda * w1

    w1 -= learning_rate * dw1
    b1 -= learning_rate * db1
    w2 -= learning_rate * dw2
    b2 -= learning_rate * db2
    
    if epoch % 50 == 0:
            print('Loss at step %d: %.3f' %(epoch, loss))
            losses.append(loss)
            
            #Calculate accuracy on the train set
            inputL = np.dot(x_train, w1)
            hiddenL = relu(inputL + b1)
            scores = np.dot(hiddenL, w2) + b2
            probs = softmax(scores)
            #Save validation accuracy for plotting later
            train_accuracy = accuracy(probs, y_train)
            train_accs.append(train_accuracy)
            
            
            x_validation = valid_data.T
            y_validation = valid_labels.T
            inputL = np.dot(x_validation, w1)
            hiddenL = relu(inputL + b1)
            scores = np.dot(hiddenL, w2) + b2
            probs = softmax(scores)
            #Save validation accuracy for plotting later
            validation_accuracy = accuracy(probs, y_validation)
            valid_accs.append(validation_accuracy)
            
            #Calculate test accuracy as well
            x_test = test_data
            y_test = test_labels
            inputL = np.dot(x_test, w1)
            hiddenL = relu(inputL + b1)
            scores = np.dot(hiddenL, w2) + b2
            probs = softmax(scores)
            test_accuracy = accuracy(probs, y_test)
            test_accs.append(test_accuracy)
            
            

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

get_ipython().run_line_magic('matplotlib', 'inline')
rows = 1
cols = 2
x_plot = list(range(int(epochs/50)))
fig, axs = plt.subplots(rows,cols,figsize=(10,5))
red_patch = mpatches.Patch(color='red', label='MNIST train accuracy')
blue_patch = mpatches.Patch(color='blue', label='MNIST validation accuracy')
green_patch = mpatches.Patch(color='green', label='MNIST test accuracy')
#yellow_patch = mpatches.Patch(color='yellow', label='USPS accuracy')


axs[0].legend(handles=[red_patch,blue_patch, green_patch])
axs[0].plot(x_plot, train_accs,'r',x_plot, valid_accs, 'b', x_plot, test_accs, 'g')
axs[0].set_title("Accuracies on MNIST")

red_patch = mpatches.Patch(color='red', label='Loss on MNIST')
axs[1].legend(handles=[red_patch])
axs[1].plot(x_plot, losses,'r')
axs[1].set_title("Loss on MNIST")


