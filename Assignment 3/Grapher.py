from pylab import *
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cbook as cbook
from scipy.misc import imread
from scipy.misc import imresize
import matplotlib.image as mpimg
import cPickle

import os

#os.pathdchd('/Desktop/CSC321/Assignment3')
trainAccuracy = []
testAccuracy = []
xRange = range(1000)
for i in xRange:
    #snapshot = cPickle.load(open("snapshot"+str(i)+".pkl"))
    #W0 = tf.Variable(snapshot['W0'])
    #b0 = tf.Variable(snapshot['b0'])
    #W1 = tf.Variable(snapshot['W1'])
    #b1 = tf.Variable(snapshot['b1'])
    #RUN AND GET ACCURACY
    trainA = i/100.00
    trainAccuracy.append(trainA)
    testA = i/100.00
    testAccuracy.append(testA)
figure(1)
plot(xRange,trainAccuracy)
ylabel('Accuracy (%)')
xlabel('Number of Iterations')
title('Accuracy of Neural Net on Training Set')
savefig('trainAccuracy.pdf')
figure(2)
plot(xRange,trainAccuracy)
ylabel('Accuracy (%)')
xlabel('Number of Iterations')
title('Accuracy of Neural Net on Training Set')
show()
savefig('trainAccuracy.pdf')

    