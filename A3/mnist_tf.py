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
M = {}
M["train"] = []
M["test"] = []
M["validate"] = []
for actor in act:
    train = [imread(p).flatten()/255. for p in glob.glob("training/"+actor+"*") if imread(p).shape == (32,32,3)]
    test = [imread(p).flatten()/255. for p in glob.glob("testing/"+actor+"*") if imread(p).shape == (32,32,3)]
    validate = [imread(p).flatten()/255. for p in glob.glob("validating/"+actor+"*") if imread(p).shape == (32,32,3)]
    M['train'].append(vstack(train))
    M['test'].append(vstack(test))
    M['validate'].append(vstack(validate))



def get_train_batch(M, N):
    n = N/6
    batch_xs = zeros((0, 32*32*3))
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
    batch_xs = zeros((0, 32*32*3))
    batch_y_s = zeros((0, 6))
    
    for a in range(6):
        batch_xs = vstack((batch_xs, ((M["test"][a]))))
        one_hot = zeros(6)
        one_hot[a] = 1
        batch_y_s = vstack((batch_y_s,   tile(one_hot, (len(M["test"][a]), 1))   ))
    return batch_xs, batch_y_s


def get_train(M):
    batch_xs = zeros((0, 32*32*3))
    batch_y_s = zeros((0, 6))
    
    for a in range(6):
        batch_xs = vstack((batch_xs, ((M["train"][a]))))
        one_hot = zeros(6)
        one_hot[a] = 1
        batch_y_s = vstack((batch_y_s,   tile(one_hot, (len(M["train"][a]), 1))   ))
    return batch_xs, batch_y_s
        



x = tf.placeholder(tf.float32, (None,32*32*3))


nhid = 800
W0 = tf.Variable(tf.random_normal([32*32*3,nhid], stddev=0.01))
b0 = tf.Variable(tf.random_normal([nhid], stddev=0.01))

W1 = tf.Variable(tf.random_normal([nhid, 6], stddev=0.01))
b1 = tf.Variable(tf.random_normal([6], stddev=0.01))

# snapshot = cPickle.load(open("new_snapshot.pkl"))
# W0 = tf.Variable(snapshot["W0"])
# b0 = tf.Variable(snapshot["b0"])
# W1 = tf.Variable(snapshot["W1"])
# b1 = tf.Variable(snapshot["b1"])


layer1 = tf.nn.relu(tf.matmul(x, W0)+b0)
layer2 = tf.matmul(layer1, W1)+b1

# W = tf.Variable(tf.random_normal([784, 10], stddev=0.01))
# b = tf.Variable(tf.random_normal([10], stddev=0.01))
# layer = tf.matmul(x, W)+b

y = tf.nn.softmax(layer2)
y_ = tf.placeholder(tf.float32, [None, 6])



lam = 1e-10
decay_penalty =lam*tf.reduce_sum(tf.square(W0))+lam*tf.reduce_sum(tf.square(W1))
NLL = -tf.reduce_sum(y_*tf.log(y)-decay_penalty)

train_step = tf.train.GradientDescentOptimizer(0.0001).minimize(NLL)

init = tf.initialize_all_variables()
sess = tf.Session()
sess.run(init)

correct_prediction = tf.equal(tf.argmax(y,1), tf.argmax(y_,1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
test_x, test_y = get_test(M)

test_accuracies = []
train_accuracies = []

for i in range(330):
    #print i  
    batch_xs, batch_ys = get_train_batch(M, 60)
    sess.run(train_step, feed_dict={x: batch_xs, y_: batch_ys})
    print "i=",i
    test_accuracies.append(sess.run(accuracy, feed_dict={x: test_x, y_: test_y}))
    print "Test:", test_accuracies[-1]
    batch_xs, batch_ys = get_train(M)
    train_accuracies.append(sess.run(accuracy, feed_dict={x: batch_xs, y_: batch_ys}))
    print "Train:", train_accuracies[-1]
    print "Penalty:", sess.run(decay_penalty)
    # snapshot = {}
    # snapshot["W0"] = sess.run(W0)
    # snapshot["W1"] = sess.run(W1)
    # snapshot["b0"] = sess.run(b0)
    # snapshot["b1"] = sess.run(b1)
    # cPickle.dump(snapshot,  open("snapshots1/new_snapshot"+str(i)+".pkl", "w"))
    
figure(1)
plot(test_accuracies)
ylabel('Accuracy (%)')
xlabel('Number of Iterations')
title('Accuracy of Neural Net on Testing Set')
savefig('testAccuracy.pdf')
close()
figure(2)
plot(train_accuracies)
ylabel('Accuracy (%)')
xlabel('Number of Iterations')
title('Accuracy of Neural Net on Training Set')
savefig('trainAccuracy.pdf')
close()


snapshot = {}
snapshot["W0"] = sess.run(W0)
snapshot["W1"] = sess.run(W1)
snapshot["b0"] = sess.run(b0)
snapshot["b1"] = sess.run(b1)
cPickle.dump(snapshot,  open("snapshots1/new_snapshot.pkl", "w"))