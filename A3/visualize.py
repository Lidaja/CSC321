from pylab import *
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cbook as cbook
import time
from scipy.misc import imread
from scipy.misc import imresize
import matplotlib.image as mpimg
from scipy.ndimage import filters
import urllib
from numpy import random
import glob
import cPickle

import os
from scipy.io import loadmat
import tensorflow as tf

def get_faces(n, W0, W1=None):
    r = np.random.randint(0, W0.shape[1], n)
    for i in r:
        figure(i)
        image = np.reshape(W0[:,i],(32,32,3))
        image = np.sum(image,axis=2)
        imshow(image, cmap = cm.coolwarm, interpolation='nearest')
        if not type(W1) == type(None):
            figtext(.5, .02, repr(W1[i,:]), ha='center')
        axis('off')
    show()

def get_face(i,W0,W1=None):
    figure(i)
    image = np.reshape(W0[:,i],(32,32,3))
    image = np.sum(image,axis=2)
    imshow(image, cmap = cm.coolwarm, interpolation='nearest')
    if not type(W1) == type(None):
        figtext(.5, .02, repr(W1[i,:]), ha='center')
    axis('off')
    show()

init = tf.initialize_all_variables()
sess = tf.Session()
sess.run(init)

snapshot = cPickle.load(open("snapshots1/new_snapshotpart3_800.pkl"))
W0800 = snapshot["W0"]
W11 = snapshot['W1']
snapshot = cPickle.load(open("snapshots1/new_snapshotpart3_300.pkl"))
W0300 = snapshot["W0"]
W12 = snapshot['W1']
W01 = (W0800 - W0800.min())
W01 = W01/W01.max()
W02 = (W0300 - W0300.min())
W02 = W02/W02.max()
#300:
#   295, 118, 123
#800:
#   310, 466