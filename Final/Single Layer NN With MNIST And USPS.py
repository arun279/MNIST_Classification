

import os
import cv2
import numpy as np
import pandas as pd
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


# In[2]:


#DNN using MNIST data
import tensorflow as tf


from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("tmp/data/",one_hot=True)

n_nodes_hl1 = 50
n_nodes_hl2 = 5
n_nodes_hl3 = 5

n_classes = 10
batch_size = 100
x = tf.placeholder('float',[None, 784]) #flatten to 1 x 784
y = tf.placeholder('float') #Labels

def NN_model(data):
    hidden_1_layer = {'weights':tf.Variable(tf.Variable(tf.random_normal([784,n_nodes_hl1]))), 'biases': tf.Variable(tf.random_normal([n_nodes_hl1]))}
    #hidden_2_layer = {'weights':tf.Variable(tf.Variable(tf.random_normal([n_nodes_hl1,n_nodes_hl2]))),'biases': tf.Variable(tf.random_normal([n_nodes_hl2]))}
    #hidden_3_layer = {'weights':tf.Variable(tf.Variable(tf.random_normal([n_nodes_hl2,n_nodes_hl3]))),'biases': tf.Variable(tf.random_normal([n_nodes_hl3]))}
    output_layer   = {'weights':tf.Variable(tf.Variable(tf.random_normal([n_nodes_hl1,n_classes]))), 'biases': tf.Variable(tf.random_normal([n_classes]))}

    l1 = tf.add(tf.matmul(data,hidden_1_layer['weights']), hidden_1_layer['biases'])
    l1 = tf.nn.relu(l1)

    output = tf.matmul(l1, output_layer['weights']) + output_layer['biases']

    return output

losses = []
train_accs = []
valid_accs = []
test_accs = []
usps_accs = []
epochs = 10
    
def train_neural_network(x):
    prediction = NN_model(x)
    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits = prediction,labels = y))

    #Minimize cost
    optimizer = tf.train.AdamOptimizer().minimize(cost)

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        for epoch in range(epochs):
            epoch_loss = 0
            for _ in range(int(mnist.train.num_examples/batch_size)):
                current_x,current_y = mnist.train.next_batch(batch_size)
                _,c = sess.run([optimizer,cost], feed_dict = {x:current_x,y:current_y})
                epoch_loss += c
            
            losses.append(epoch_loss)
            print('Epoch', epoch, 'done out of', epochs,'loss:',epoch_loss)
            correct = tf.equal(tf.argmax(prediction,1),tf.argmax(y,1))
            accuracy = tf.reduce_mean(tf.cast(correct,'float'))
            train_accuracy_mnist = accuracy.eval({x:mnist.train.images, y: mnist.train.labels})
            test_accuracy_mnist = accuracy.eval({x:mnist.test.images, y: mnist.test.labels})
            valid_accuracy_mnist = accuracy.eval({x:mnist.validation.images, y: mnist.validation.labels})
            usps_accuracy = accuracy.eval({x:usps_images_shuf, y: usps_labels_shuf})
            print('Train accuracy:',train_accuracy_mnist)
            print('Validation accuracy:',valid_accuracy_mnist)
            print('Test accuracy:',test_accuracy_mnist)
            print('USPS Accuracy:',usps_accuracy)
            train_accs.append(train_accuracy_mnist)
            test_accs.append(test_accuracy_mnist)
            valid_accs.append(valid_accuracy_mnist)
            usps_accs.append(usps_accuracy)
        

train_neural_network(x)
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

%matplotlib inline
rows = 1
cols = 2
x_plot = list(range(epochs))
fig, axs = plt.subplots(rows,cols,figsize=(10,5))
red_patch = mpatches.Patch(color='red', label='MNIST train accuracy')
blue_patch = mpatches.Patch(color='blue', label='MNIST validation accuracy')
green_patch = mpatches.Patch(color='green', label='MNIST test accuracy')
yellow_patch = mpatches.Patch(color='yellow', label='USPS accuracy')


axs[0].legend(handles=[red_patch])
axs[0].plot(x_plot, train_accs,'r',x_plot, valid_accs, 'b', x_plot, test_accs, 'g',
           x_plot, usps_accs, 'y')
axs[0].set_title("Accuracies on MNIST")

red_patch = mpatches.Patch(color='red', label='Loss on MNIST')


axs[1].legend(handles=[red_patch])
axs[1].plot(x_plot, losses,'r')
axs[1].set_title("Loss on MNIST")
