from pylab import *
from numpy import *
import matplotlib.pyplot as plt
import matplotlib.cbook as cbook
import time
from scipy.misc import imread
from scipy.misc import imresize
import matplotlib.image as mpimg
from scipy.ndimage import filters
import urllib
from numpy import random


import cPickle

import os
from scipy.io import loadmat

os.chdir('/h/u4/g5/00/g4jackso/Desktop/CSC321')
#Load the MNIST digit data
M = loadmat("mnist_all.mat")

#Display the 150-th "5" digit from the training set
#imshow(M["train5"][150].reshape((28,28)), cmap=cm.gray)
#show()


def softmax(y):
    '''Return the output of the softmax function for the matrix of output y. y
    is an NxM matrix where N is the number of outputs for a single case, and M
    is the number of cases'''
    return exp(y)/tile(sum(exp(y),0), (len(y),1))
    
def tanh_layer(y, W, b):    
    '''Return the output of a tanh layer for the input matrix y. y
    is an NxM matrix where N is the number of inputs for a single case, and M
    is the number of cases'''
    return tanh(dot(W.T, y)+b)

def forward(x, W0, b0, W1, b1):
    L0 = tanh_layer(x, W0, b0)
    L1 = tanh_layer(L0, W1, b1)
    output = softmax(L1)
    return L0, L1, output
    
def cost(y, y_):
    return -sum(y_*log(y)) 

def deriv_multilayer(W0, b0, W1, b1, x, L0, L1, y, y_):
    '''Incomplete function for computing the gradient of the cross-entropy
    cost function w.r.t the parameters of a neural network'''
    dCdL1 =  y - y_
    dCdW1 =  dot(L0, ((1- L1**2)*dCdL1).T )
    
def P(x,w,b):
    return softmax(dot(w.T,x)+b)

def gradient(x,y,w,b):
    p = P(x,w,b)
    w = vstack([b.T,w])
    x = vstack([1,x])
    print(p.shape)
    print(y.shape)
    print(x.shape)
    return dot(x,(p-y).T)

def testGradient(x,y,w,b):
    h = 0.000001
    w = vstack([b.T,w])
    x = vstack([1,x])
    g = gradient(x,y,w,b)
    for i in range(w.shape[0]):
        for j in range(w.shape[1]):
            w_1 = np.copy(w)
            w_2 = np.copy(w)
            w_1[i,j] += h
            w_2[i,j] -= h
            P1 = P(x,w_1,b)
            P2 = P(x,w_2,b)
            Cost1 = cost(P1,y)
            Cost2 = cost(P2,y)
            print (Cost1-Cost2)/(2*h)
            
#Load sample weights for the multilayer neural network
snapshot = cPickle.load(open("snapshot50.pkl"))
W0 = snapshot["W0"]
b0 = snapshot["b0"].reshape((300,1))
W1 = snapshot["W1"]
b1 = snapshot["b1"].reshape((10,1))


x = M["train5"][148:149].T/255.
b = random.random_sample((10,1))/100.0
w = random.random_sample((784,10))/100.0
p = P(x,w,b)
y = array([[1],[0],[0],[0],[0],[0],[0],[0],[0],[0]])

"""
#Load one example from the training set, and run it through the
#neural network
x = M["train5"][148:149].T    
L0, L1, output = forward(x, W0, b0, W1, b1)
#get the index at which the output is the largest
y = argmax(output)

################################################################################
#Code for displaying a feature from the weight matrix mW
#fig = figure(1)
#ax = fig.gca()    
#heatmap = ax.imshow(mW[:,50].reshape((28,28)), cmap = cm.coolwarm)    
#fig.colorbar(heatmap, shrink = 0.5, aspect=5)
#show()
################################################################################
"""