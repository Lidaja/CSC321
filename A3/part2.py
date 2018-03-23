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

act = ["bracco", "butler", "gilpin", "harmon", "radcliffe", "vartan"]
t = int(time.time())
#t = 1454219613
print "t=", t
random.seed(t)
mat = loadmat("activations.mat")
M = {}
M["train"] = []
M["test"] = []
M["validate"] = []
for a in range(len(act)):
    train = [mat['train'][0,a][i,:,:,:,:].flatten()/255. for i in range(mat['train'][0,a].shape[0])]
    test = [mat['test'][0,a][i,:,:,:,:].flatten()/255. for i in range(mat['test'][0,a].shape[0])]
    validate = [mat['validate'][0,a][i,:,:,:,:].flatten()/255. for i in range(mat['validate'][0,a].shape[0])]
    M['train'].append(vstack(train))
    M['test'].append(vstack(test))
    M['validate'].append(vstack(validate))




def get_train_batch(M, N):
    n = N/6
    batch_xs = zeros((0, 13*13*384))
    batch_y_s = zeros((0, 6))

    #train_size = len(M[train_k[0]])
    train_size = 70
    
    for a in range(6):
        train_size = len(M["train"][a])
        idx = array(random.permutation(train_size)[:n])
        batch_xs = vstack((batch_xs, ((M["train"][a])[idx])))
        one_hot = zeros(6)
        one_hot[a] = 1
        batch_y_s = vstack((batch_y_s,tile(one_hot,(n,1))))
    return batch_xs, batch_y_s
    

def get_test(M):
    batch_xs = zeros((0, 13*13*384))
    batch_y_s = zeros((0, 6))
    
    for a in range(6):
        batch_xs = vstack((batch_xs, ((M["test"][a]))))
        one_hot = zeros(6)
        one_hot[a] = 1
        batch_y_s = vstack((batch_y_s,   tile(one_hot, (len(M["test"][a]), 1))   ))
    return batch_xs, batch_y_s


def get_train(M):
    batch_xs = zeros((0, 13*13*384))
    batch_y_s = zeros((0, 6))
    
    for a in range(6):
        batch_xs = vstack((batch_xs, ((M["train"][a]))))
        one_hot = zeros(6)
        one_hot[a] = 1
        batch_y_s = vstack((batch_y_s,   tile(one_hot, (len(M["train"][a]), 1))   ))
    return batch_xs, batch_y_s
        
def get_validate(M):
    batch_xs = zeros((0, 13*13*384))
    batch_y_s = zeros((0, 6))
    
    for a in range(6):
        batch_xs = vstack((batch_xs, ((M["validate"][a]))))
        one_hot = zeros(6)
        one_hot[a] = 1
        batch_y_s = vstack((batch_y_s,   tile(one_hot, (len(M["validate"][a]), 1))   ))
    return batch_xs, batch_y_s
        

x = tf.placeholder(tf.float32, (None,13*13*384))


# nhid = 800
# W0 = tf.Variable(tf.random_normal([32*32*3,nhid], stddev=0.01))
# b0 = tf.Variable(tf.random_normal([nhid], stddev=0.01))
# 
# W1 = tf.Variable(tf.random_normal([nhid, 6], stddev=0.01))
# b1 = tf.Variable(tf.random_normal([6], stddev=0.01))

# snapshot = cPickle.load(open("new_snapshot.pkl"))
# W0 = tf.Variable(snapshot["W0"])
# b0 = tf.Variable(snapshot["b0"])
# W1 = tf.Variable(snapshot["W1"])
# b1 = tf.Variable(snapshot["b1"])


# layer1 = tf.nn.relu(tf.matmul(x, W0)+b0)
# layer2 = tf.matmul(layer1, W1)+b1

W = tf.Variable(tf.random_normal([13*13*384, 6], stddev=1e-10))
b = tf.Variable(tf.random_normal([6], stddev=1e-10))
layer = tf.matmul(x, W)+b

y = tf.nn.softmax(layer)
y_ = tf.placeholder(tf.float32, [None, 6])



lam = 1e-10
decay_penalty = lam*tf.reduce_sum(tf.square(W))
NLL = -tf.reduce_sum(y_*tf.log(y))+decay_penalty

train_step = tf.train.GradientDescentOptimizer(4e-6).minimize(NLL)

init = tf.initialize_all_variables()
sess = tf.Session()
sess.run(init)

correct_prediction = tf.equal(tf.argmax(y,1), tf.argmax(y_,1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
test_x, test_y = get_test(M)
train_x, train_y = get_train(M)
validate_x, validate_y = get_validate(M)

test_accuracies = []
train_accuracies = []
validate_accuracies = []

for i in range(501):
    #print i 
    batch_xs, batch_ys = get_train_batch(M, 60)
    sess.run(train_step, feed_dict={x: batch_xs, y_: batch_ys})
    test_accuracies.append(sess.run(accuracy, feed_dict={x: test_x, y_: test_y}))
    train_accuracies.append(sess.run(accuracy, feed_dict={x: train_x, y_: train_y}))
    validate_accuracies.append(sess.run(accuracy, feed_dict={x: validate_x, y_: validate_y}))
    if i % 50 == 0:
        print "i=",i    
        print "Test:", test_accuracies[-1]
        print "Train:", train_accuracies[-1]
        print "Validate:", validate_accuracies[-1]
        print "Penalty:", sess.run(decay_penalty)
        print "NLL", sess.run(NLL, feed_dict={x: batch_xs, y_:batch_ys})

snapshot = {}
snapshot["W"] = sess.run(W)
snapshot["b"] = sess.run(b)
cPickle.dump(snapshot,  open("snapshots1/new_snapshot_part2.pkl", "w"))
    
figure(1)
plot(test_accuracies,'r',label='test')
plot(train_accuracies,'b',label='train')
plot(validate_accuracies,'g',label='validate')
legend()
ylabel('Accuracy (%)')
xlabel('Number of Iterations')
title('Accuracy of AlexNet')
savefig('Part2Accuracy.pdf')
close()
