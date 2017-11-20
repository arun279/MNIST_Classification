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
                imlist.append(255-np.ravel(im_resize))
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