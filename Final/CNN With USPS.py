
# coding: utf-8

# In[1]:


from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)
import numpy as np
import random as ran
import tensorflow as tf


# In[2]:


import os
import cv2
import numpy as np
import pandas as pd
path = './datafiles/Numerals/'

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
images_shuf = images[ids]
labels_shuf = labels[ids]


# In[3]:


# comment
#learning_rate = 0.01
#training_epochs = 25
#batch_size = 100
#display_step = 1

# x = tf.placeholder(tf.float32, [None, 784])
# y = tf.placeholder(tf.float32, [None, 10])

x = tf.placeholder(tf.float32, [None, 784])
W = tf.Variable(tf.zeros([784, 10]))
b = tf.Variable(tf.zeros([10]))
y = tf.nn.softmax(tf.matmul(x, W) + b)

# W = tf.Variable(tf.zeros([784,10]))
# b = tf.Variable(tf.zeros([10]))
y_ = tf.placeholder(tf.float32, [None, 10])

# pred = tf.nn.softmax(tf.matmul(x, W) + b)
cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(y), reduction_indices=[1]))

# cost = tf.reduce_mean(-tf.reduce_sum(y*tf.log(pred), reduction_indices=1))
train_step = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)

# optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(cost)
sess = tf.InteractiveSession()

# init = tf.global_variables_initializer()
tf.global_variables_initializer().run()


# In[4]:


for _ in range(1000):
    batch_xs, batch_ys = mnist.train.next_batch(100)
    sess.run(train_step, feed_dict={x: batch_xs, y_:batch_ys})

correct_prediction = tf.equal(tf.argmax(y,1), tf.argmax(y_,1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
print(sess.run(accuracy, feed_dict={x: mnist.test.images, y_: mnist.test.labels}))


# In[5]:


# Weight initialization
def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)

def bias_variable(shape):
    initial = tf.constant(0.1, shape = shape)
    return tf.Variable(initial)

# Convolution and Pooling
def conv2d(x, W):
    return tf.nn.conv2d(x, W, strides=[1,1,1,1], padding='SAME')

def max_pool_2x2(x):
    return tf.nn.max_pool(x, ksize=[1,2,2,1], strides=[1,2,2,1], padding='SAME')

###### First Convolutional Layer
W_conv1 = weight_variable([5,5,1,32])
b_conv1 = bias_variable([32])
x_image = tf.reshape(x,[-1,28,28,1])

h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)
h_pool1 = max_pool_2x2(h_conv1)

# Second Convolutional Layer
W_conv2 = weight_variable([5,5,32,64])
b_conv2 = bias_variable([64])

h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)
h_pool2 = max_pool_2x2(h_conv2)

# Densley Connected Layer
W_fc1 = weight_variable([7 * 7 * 64, 1024])
b_fc1 = bias_variable([1024])

h_pool2_flat = tf.reshape(h_pool2, [-1, 7*7*64])
h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)

# Dropout
keep_prob = tf.placeholder(tf.float32)
h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

# Readout Layer
W_fc2 = weight_variable([1024, 10])
b_fc2 = bias_variable([10])

y_conv = tf.matmul(h_fc1_drop, W_fc2) + b_fc2

# Train and Evaluate Model
cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y_conv))
train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)
correct_prediction = tf.equal(tf.argmax(y_conv, 1), tf.argmax(y_, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))


# In[6]:


with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    for i in range(20000):
        batch = mnist.train.next_batch(50)
        if i % 100 == 0:
            train_accuracy = accuracy.eval(feed_dict={x: batch[0], y_: batch[1], keep_prob: 1.0})
            print('step %d, training accuracy %g' % (i, train_accuracy))
        train_step.run(feed_dict={x: batch[0], y_: batch[1], keep_prob:0.5})
    print('test accuracy %g' % accuracy.eval(feed_dict={x: mnist.test.images, y_: mnist.test.labels, keep_prob: 1.0}))
    print('test accuracy %g' % accuracy.eval(feed_dict={x: images_shuf[0:10000,:], y_: labels_shuf[0:10000,:], keep_prob: 1.0}))


# In[ ]:




