
# coding: utf-8

# In[2]:


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
                imlist.append(np.ravel(im_resize))
                lblist.append(int(d))

# Normalize the values in the image
images = np.array(imlist)/255.
labels = np.reshape(np.array(lblist),[-1,1]).flatten()
# Get one-hot representation
labels = np.array(pd.get_dummies(labels))

# Set a random seed to get reproducible results
np.random.seed(42)


# In[5]:



# Randomize the input data
ids = list(range(0,len(labels)))
np.random.shuffle(ids)
usps_images_shuf = images[ids]
usps_labels_shuf = labels[ids]


# In[12]:


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
n = mnist.train.num_examples
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
train_step = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)

sess = tf.InteractiveSession()
tf.global_variables_initializer().run()
for _ in range(1000):
    batch_xs, batch_ys = mnist.train.next_batch(100)
    sess.run(train_step, feed_dict={x: batch_xs, y_orig: batch_ys})
#have a look at the results
correct_prediction = tf.equal(tf.argmax(y,1), tf.argmax(y_orig,1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

"""
iter_ = data_iterator(train_data,train_labels.T,batch_size=100)

for _ in range(1000):
    images_batch, labels_batch = next(iter_)
    train_step.run(feed_dict={x: images_batch, y_: labels_batch})
correct_prediction = tf.equal(tf.argmax(y,1), tf.argmax(y_,1))
"""
print(accuracy.eval(feed_dict={x: usps_images_shuf, y_orig: usps_labels_shuf}))


# In[13]:




