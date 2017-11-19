
# coding: utf-8

# In[194]:


from tensorflow.examples.tutorials.mnist import input_data
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


# In[195]:


#Let's load the MNIST dataset and store it as numpy arrays in a pickle file for later use.

SOURCE_URL = 'http://yann.lecun.com/exdb/mnist/'
WORK_DIRECTORY = 'data'
IMAGE_SIZE = 28
NUM_CHANNELS = 1
PIXEL_DEPTH = 255
NUM_LABELS = 10

def maybe_download(filename):
    """Download the data from Yann's website, unless it's already here."""
    if not tf.gfile.Exists(WORK_DIRECTORY):
        tf.gfile.MakeDirs(WORK_DIRECTORY)
    filepath = os.path.join(WORK_DIRECTORY, filename)
    if not tf.gfile.Exists(filepath):
        filepath, _ = urllib.request.urlretrieve(SOURCE_URL + filename, filepath)
    with tf.gfile.GFile(filepath) as f:
        size = f.size()
    print('Successfully downloaded', filename, size, 'bytes.')
    return filepath


def extract_data(filename, num_images):
    """Extract the images into a 4D tensor [image index, y, x, channels].
    Values are rescaled from [0, 255] down to [-0.5, 0.5].
    """
    print('Extracting', filename)
    with gzip.open(filename) as bytestream:
        bytestream.read(16)
        buf = bytestream.read(IMAGE_SIZE * IMAGE_SIZE * num_images)
        data = np.frombuffer(buf, dtype=np.uint8).astype(np.float32)
        #data = (data - (PIXEL_DEPTH / 2.0)) / PIXEL_DEPTH
        data = data.reshape(num_images, IMAGE_SIZE, IMAGE_SIZE, 1)
    return data


def extract_labels(filename, num_images):
    """Extract the labels into a vector of int64 label IDs."""
    print('Extracting', filename)
    with gzip.open(filename) as bytestream:
        bytestream.read(8)
        buf = bytestream.read(1 * num_images)
        labels = np.frombuffer(buf, dtype=np.uint8).astype(np.int64)
    return labels

def save_mnist_data():
    train_data_filename = maybe_download('train-images-idx3-ubyte.gz')
    train_labels_filename = maybe_download('train-labels-idx1-ubyte.gz')
    test_data_filename = maybe_download('t10k-images-idx3-ubyte.gz')
    test_labels_filename = maybe_download('t10k-labels-idx1-ubyte.gz')

    # Extract it into np arrays.
    train_data = extract_data(train_data_filename, 60000)
    train_labels = extract_labels(train_labels_filename, 60000)
    test_data = extract_data(test_data_filename, 10000)
    test_labels = extract_labels(test_labels_filename, 10000)

    #Pickle these variables
    mnist_data = {
        "train_data": train_data,
        "train_labels": train_labels,
        "test_data": test_data,
        "test_labels": test_labels}

    pickle.dump(obj = mnist_data, file=open('mnist_data.pkl','wb'),protocol=3)
    print("MNIST dumped into mnist_data.pkl")




            
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
                                                                          test_size=0.08, 
                                                                          random_state=42)
    #Standardize the data - divide by 255
    if standardize:
        train_data, valid_data, test_data = train_data/255., valid_data/255., test_data/255.
    train_data, valid_data, test_data = train_data, valid_data, test_data
    
    return train_data,train_labels,valid_data,valid_labels, test_data, test_labels
train_data,train_labels,valid_data, valid_labels, test_data,test_labels = load_mnist_data()

#One hot encode the labels
train_labels = np.array(pd.get_dummies(train_labels.flatten())).T
valid_labels = np.array(pd.get_dummies(valid_labels.flatten())).T
test_labels = np.array(pd.get_dummies(test_labels.flatten())).T


# In[196]:


def data_iterator(features, labels,batch_size):
    """ A simple data iterator """
    batch_idx = 0
    while True:
        # shuffle labels and features
        idxs = np.arange(0, len(features))
        np.random.shuffle(idxs)
        shuf_features = features[idxs]
        shuf_labels = labels[idxs]
        for batch_idx in range(0, len(features), batch_size):
            images_batch = shuf_features[batch_idx:batch_idx+batch_size] / 255.
            images_batch = images_batch.astype("float32")
            labels_batch = shuf_labels[batch_idx:batch_idx+batch_size]
            yield images_batch, labels_batch


save_mnist_data()