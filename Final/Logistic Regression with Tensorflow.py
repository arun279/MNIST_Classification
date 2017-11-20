
# coding: utf-8

# In[21]:



# coding: utf-8

# In[1]:


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


# In[2]:


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
save_mnist_data()


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
    train_data, valid_data, test_data = train_data, valid_data, test_data
    
    return train_data,train_labels,valid_data,valid_labels, test_data, test_labels

#train_data,train_labels,valid_data, valid_labels, test_data,test_labels = load_mnist_data()

#One hot encode the labels
#train_labels = np.array(pd.get_dummies(train_labels.flatten())).T
#valid_labels = np.array(pd.get_dummies(valid_labels.flatten())).T
#test_labels = np.array(pd.get_dummies(test_labels.flatten())).T

print("Data loaded.")
print("Shapes of data are as follows: ")
print("Train data shape: "+ str(train_data.shape))
print("Train labels shape: "+ str(train_labels.shape))
print("Test data shape: "+ str(test_data.shape))
print("Test labels shape: "+ str(test_labels.shape))
print("Validation data shape: "+ str(valid_data.shape))
print("Validation labels shape: "+ str(valid_labels.shape))


# In[15]:


#Load USPS Data
import numpy as np
import random as ran
import tensorflow as tf
import cv2
import os
import pandas as pd
from IPython.display import Image

'''
In this code, I try to retrieve USPS data from a directory structure
where name of the directort is the label for the files in that directory
'''

path = './images/Numerals/'

# Initialize 2 lists for labels and images
imlist = []
lblist = []
dirs = list(os.walk(path))[0]
dirs = dirs[1]
for d in dirs:
    #print(d)
    for files in d:
        directory = os.path.join(path,files)
        #print(directory)
        for fname in os.listdir(directory):
            if(fname.endswith(".png")):    
                image = cv2.imread(os.path.join(path,d,fname))
                im_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
                im_resize = cv2.resize(im_gray, (28, 28))
                #Letters are colored in the opposite way when compared to MNIST
                imlist.append(255 - np.ravel(im_resize)) 
                lblist.append(int(d))

# Normalize the values in the image
images = np.array(imlist)/255.
labels = np.reshape(np.array(lblist),[-1,1]).flatten()
# Get one-hot representation
labels = np.array(pd.get_dummies(labels))

# Set a random seed to get reproducible results
np.random.seed(42)

# Randomize the input data
ids = list(range(0,len(labels)))
np.random.shuffle(ids)
usps_images_shuf = images[ids]
usps_labels_shuf = labels[ids]
print("USPS Data loaded.")
print("USPS dataset shape: ", str(usps_images_shuf.shape))
print("USPS labels shape: ", str(usps_labels_shuf.shape))


# In[18]:


import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

#set up params
sess = tf.Session()
seed = 2
tf.set_random_seed(2)
batch_size = 100
learning_rate = 1e-4
m = 784
epochs = 50
n = train_data.shape[1]
k = 10 #Number of classes

x = tf.placeholder(tf.float32, [None, m])
y_orig = tf.placeholder(tf.float32, [None, k])


W = tf.Variable(tf.zeros([m, k]))
b = tf.Variable(tf.zeros([k]))
print("W.shape: " + str(W.shape))
print("b.shape: " + str(b.shape))
#let Y = Wx + b with a softmax activiation function
y = tf.nn.softmax(tf.matmul(x, W) + b)


cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_orig * tf.log(y), reduction_indices=[1]))
optimizer = tf.train.AdamOptimizer().minimize(cross_entropy)

losses = []
train_accs = []
valid_accs = []
test_accs = []
usps_accs = []
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    for epoch in range(epochs):
        epoch_loss = 0
        for _ in range(int(mnist.train.num_examples/batch_size)):
            current_x, current_y = mnist.train.next_batch(batch_size)
            _,c = sess.run([optimizer,cross_entropy], feed_dict={x: current_x, y_orig: current_y})
            epoch_loss += c
        print("Loss at epoch %d: %.3f" %(epoch,epoch_loss))
        losses.append(epoch_loss)
        correct_prediction = tf.equal(tf.argmax(y,1), tf.argmax(y_orig,1))
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
        train_accuracy_mnist = accuracy.eval({x:mnist.train.images, y_orig: mnist.train.labels})
        test_accuracy_mnist = accuracy.eval({x:mnist.test.images, y_orig: mnist.test.labels})
        valid_accuracy_mnist = accuracy.eval({x:mnist.validation.images, y_orig: mnist.validation.labels})
        usps_accuracy = accuracy.eval({x:usps_images_shuf, y_orig: usps_labels_shuf})
        print('Train accuracy:',train_accuracy_mnist)
        print('Validation accuracy:',valid_accuracy_mnist)
        print('Test accuracy:',test_accuracy_mnist)
        print('USPS Accuracy:',usps_accuracy)
        train_accs.append(train_accuracy_mnist)
        test_accs.append(test_accuracy_mnist)
        valid_accs.append(valid_accuracy_mnist)
        usps_accs.append(usps_accuracy)
        
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
rows = 1
cols = 2
x_plot = list(range(epochs))
fig, axs = plt.subplots(rows,cols,figsize=(10,5))
red_patch = mpatches.Patch(color='red', label='MNIST train accuracy')
blue_patch = mpatches.Patch(color='blue', label='MNIST validation accuracy')
green_patch = mpatches.Patch(color='green', label='MNIST test accuracy')
axs[0].legend(handles=[red_patch, blue_patch, green_patch])
axs[0].plot(x_plot, train_accs,'r',x_plot, valid_accs, 'b', x_plot, test_accs, 'g')
axs[0].set_title("Accuracies on MNIST")

yellow_patch = mpatches.Patch(color='yellow', label='USPS accuracy')
axs[1].legend(handles = [yellow_patch])
axs[1].plot(x_plot, usps_accs, 'y')
axs[1].set_title("Accuracies on USPS")

