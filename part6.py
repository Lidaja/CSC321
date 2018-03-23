################################################################################
#Michael Guerzhoy, 2016
#AlexNet implementation in TensorFlow, with weights
#Details: 
#http://www.cs.toronto.edu/~guerzhoy/tf_alexnet/
#
#With code from https://github.com/ethereon/caffe-tensorflow
#Model from  https://github.com/BVLC/caffe/tree/master/models/bvlc_alexnet
#Weights from Caffe converted using https://github.com/ethereon/caffe-tensorflow
#
#
################################################################################

from numpy import *
import os
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
from scipy.io import savemat
import tensorflow as tf

from caffe_classes import class_names
import cPickle

# train_x = zeros((1, 227,227,3)).astype(float32)
# train_y = zeros((1, 1000))
# xdim = train_x.shape[1:]
# ydim = train_y.shape[1]
# 
# 
# 
# ################################################################################
# #Read Image
# 
# x_dummy = (random.random((1,)+ xdim)/255.).astype(float32)
# i = x_dummy.copy()
# i[0,:,:,:] = (imread("poodle.png")[:,:,:3]).astype(float32)
# i = i-mean(i)

################################################################################

# (self.feed('data')
#         .conv(11, 11, 96, 4, 4, padding='VALID', name='conv1')
#         .lrn(2, 2e-05, 0.75, name='norm1')
#         .max_pool(3, 3, 2, 2, padding='VALID', name='pool1')
#         .conv(5, 5, 256, 1, 1, group=2, name='conv2')
#         .lrn(2, 2e-05, 0.75, name='norm2')
#         .max_pool(3, 3, 2, 2, padding='VALID', name='pool2')
#         .conv(3, 3, 384, 1, 1, name='conv3')
#         .conv(3, 3, 384, 1, 1, group=2, name='conv4')
#         .conv(3, 3, 256, 1, 1, group=2, name='conv5')
#         .fc(4096, name='fc6')
#         .fc(4096, name='fc7')
#         .fc(1000, relu=False, name='fc8')
#         .softmax(name='prob'))


net_data = load("bvlc_alexnet.npy").item()

act = ["bracco", "butler", "gilpin", "harmon", "radcliffe", "vartan"]
M = {}
M["train"] = []
M["test"] = []
M["validate"] = []
for a in range(len(act)):
    M['train'].append([])
    for p in glob.glob("big/training/"+act[a]+"*"):
        if imread(p).shape == (227,227,3):
            im = np.reshape(imread(p),(1,227,227,3))
            im = im - im.mean()
            M['train'][a].append(im)
    M['test'].append([])
    for p in glob.glob("big/testing/"+act[a]+"*"):
        if imread(p).shape == (227,227,3):
            im = np.reshape(imread(p),(1,227,227,3))
            im = im - im.mean()
            M['test'][a].append(im)
    M['validate'].append([])
    for p in glob.glob("big/validating/"+act[a]+"*"):
        if imread(p).shape == (227,227,3):
            im = np.reshape(imread(p),(1,227,227,3))
            im = im - im.mean()
            M['validate'][a].append(im)
        
for actor in act:
    train = [imread(p)-imread(p).mean() for p in glob.glob("training/"+actor+"*") if imread(p).shape == (32,32,3)]
    test = [imread(p).flatten()-imread(p).flatten().mean() for p in glob.glob("testing/"+actor+"*") if imread(p).shape == (32,32,3)]
    validate = [imread(p).flatten()-imread(p).flatten().mean() for p in glob.glob("validating/"+actor+"*") if imread(p).shape == (32,32,3)]
    M['train'].append(vstack(train))
    M['test'].append(vstack(test))
    M['validate'].append(vstack(validate))

init = tf.initialize_all_variables()
sess = tf.Session()
sess.run(init)

def conv(input, kernel, biases, k_h, k_w, c_o, s_h, s_w,  padding="VALID", group=1):
    '''From https://github.com/ethereon/caffe-tensorflow
    '''
    c_i = input.get_shape()[-1]
    assert c_i%group==0
    assert c_o%group==0
    convolve = lambda i, k: tf.nn.conv2d(i, k, [1, s_h, s_w, 1], padding=padding)
    
    
    if group==1:
        conv = convolve(input, kernel)
    else:
        input_groups = tf.split(3, group, input)
        kernel_groups = tf.split(3, group, kernel)
        output_groups = [convolve(i, k) for i,k in zip(input_groups, kernel_groups)]
        conv = tf.concat(3, output_groups)
    return  tf.reshape(tf.nn.bias_add(conv, biases), conv.get_shape().as_list())

def get_train_batch(M, N):
    n = N/6
    batch_xs = []
    batch_y_s = []

    #train_size = len(M[train_k[0]])
    train_size = 70
    
    for a in range(6):
        train_size = len(M["train"][a])
        idx = array(random.permutation(train_size)[:n])
        one_hot = zeros((1,6))
        one_hot[0,a] = 1
        batch_y_s += [one_hot]*n           
        for i in idx:
            batch_xs.append(M["train"][a][i])
    return batch_xs, batch_y_s
    

def get_test(M):
    batch_xs = []
    batch_y_s = []
    
    for a in range(6):
        one_hot = zeros((1,6))
        one_hot[0,a] = 1
        batch_y_s += [one_hot]*len(M["test"][a])
        batch_xs += M["test"][a]
    return batch_xs, batch_y_s


def get_train(M):
    batch_xs = []
    batch_y_s = []
    
    for a in range(6):
        one_hot = zeros((1,6))
        one_hot[0,a] = 1
        batch_y_s += [one_hot]*len(M["train"][a])       
        batch_xs += M["train"][a]
    return batch_xs, batch_y_s

    
def get_validate(M):
    batch_xs = []
    batch_y_s = []
    
    for a in range(6):
        one_hot = zeros((1,6))
        one_hot[0,a] = 1
        batch_y_s += [one_hot]*len(M["validate"][a])       
        batch_xs += M["validate"][a]
    return batch_xs, batch_y_s


# x = tf.Variable(i)
x = tf.placeholder(float32, [1,227,227,3])

#conv1
#conv(11, 11, 96, 4, 4, padding='VALID', name='conv1')
k_h = 11; k_w = 11; c_o = 96; s_h = 4; s_w = 4
conv1W = tf.Variable(net_data["conv1"][0])
conv1b = tf.Variable(net_data["conv1"][1])
conv1_in = conv(x, conv1W, conv1b, k_h, k_w, c_o, s_h, s_w, padding="SAME", group=1)
conv1 = tf.nn.relu(conv1_in)

#lrn1
#lrn(2, 2e-05, 0.75, name='norm1')
radius = 2; alpha = 2e-05; beta = 0.75; bias = 1.0
lrn1 = tf.nn.local_response_normalization(conv1,
                                                  depth_radius=radius,
                                                  alpha=alpha,
                                                  beta=beta,
                                                  bias=bias)

#maxpool1
#max_pool(3, 3, 2, 2, padding='VALID', name='pool1')
k_h = 3; k_w = 3; s_h = 2; s_w = 2; padding = 'VALID'
maxpool1 = tf.nn.max_pool(lrn1, ksize=[1, k_h, k_w, 1], strides=[1, s_h, s_w, 1], padding=padding)


#conv2
#conv(5, 5, 256, 1, 1, group=2, name='conv2')
k_h = 5; k_w = 5; c_o = 256; s_h = 1; s_w = 1; group = 2
conv2W = tf.Variable(net_data["conv2"][0])
conv2b = tf.Variable(net_data["conv2"][1])
conv2_in = conv(maxpool1, conv2W, conv2b, k_h, k_w, c_o, s_h, s_w, padding="SAME", group=group)
conv2 = tf.nn.relu(conv2_in)


#lrn2
#lrn(2, 2e-05, 0.75, name='norm2')
radius = 2; alpha = 2e-05; beta = 0.75; bias = 1.0
lrn2 = tf.nn.local_response_normalization(conv2,
                                                  depth_radius=radius,
                                                  alpha=alpha,
                                                  beta=beta,
                                                  bias=bias)

#maxpool2
#max_pool(3, 3, 2, 2, padding='VALID', name='pool2')                                                  
k_h = 3; k_w = 3; s_h = 2; s_w = 2; padding = 'VALID'
maxpool2 = tf.nn.max_pool(lrn2, ksize=[1, k_h, k_w, 1], strides=[1, s_h, s_w, 1], padding=padding)

#conv3
#conv(3, 3, 384, 1, 1, name='conv3')
k_h = 3; k_w = 3; c_o = 384; s_h = 1; s_w = 1; group = 1
conv3W = tf.Variable(net_data["conv3"][0])
conv3b = tf.Variable(net_data["conv3"][1])
conv3_in = conv(maxpool2, conv3W, conv3b, k_h, k_w, c_o, s_h, s_w, padding="SAME", group=group)
conv3 = tf.nn.relu(conv3_in)

#conv4
#conv(3, 3, 384, 1, 1, group=2, name='conv4')
k_h = 3; k_w = 3; c_o = 384; s_h = 1; s_w = 1; group = 2
conv4W = tf.Variable(net_data["conv4"][0])
conv4b = tf.Variable(net_data["conv4"][1])
conv4_in = conv(conv3, conv4W, conv4b, k_h, k_w, c_o, s_h, s_w, padding="SAME", group=group)
conv4 = tf.nn.relu(conv4_in)


snapshot = cPickle.load(open("snapshots1/new_snapshot_part2.pkl"))
W = tf.Variable(snapshot["W"])
b = tf.Variable(snapshot["b"])

layer = tf.matmul(tf.reshape(conv4,[1,13*13*384])/255., W)+b

y = tf.nn.softmax(layer)
y_ = tf.placeholder(tf.float32, [None, 6])



lam = 1e-10
decay_penalty = lam*tf.reduce_sum(tf.square(W))
NLL = -tf.reduce_sum(y_*tf.log(y))+decay_penalty

train_step = tf.train.GradientDescentOptimizer(4e-5).minimize(NLL)

init = tf.initialize_all_variables()
sess = tf.Session()
sess.run(init)
test_x, test_y = get_test(M)
train_x, train_y = get_train(M)
validate_x, validate_y = get_validate(M)
test_accuracies = []
train_accuracies = []
validate_accuracies = []


for i in range(200):
    print "started"
    correct = 0
    total = 0
    for j in range(len(train_x)):
        im = train_x[j]
        if argmax(sess.run(layer, feed_dict={x:im})) == argmax(train_y[j]):
            correct += 1
        total += 1
    train_accuracies.append(correct/float(total))
    correct = 0
    total = 0
    for j in range(len(test_x)):
        im = test_x[j]
        if argmax(sess.run(layer, feed_dict={x:im})) == argmax(test_y[j]):
            correct += 1
        total += 1
    test_accuracies.append(correct/float(total))
    correct = 0
    total = 0
    for j in range(len(validate_x)):
        im = validate_x[j]
        if argmax(sess.run(layer, feed_dict={x:im})) == argmax(validate_y[j]):
            correct += 1
        total += 1
    validate_accuracies.append(correct/float(total))
    print "i=",i
    print "Train:", train_accuracies[-1]
    print "Validate:", validate_accuracies[-1]
    print "Test:", test_accuracies[-1]
    print "Penalty:", sess.run(decay_penalty)
    batch_xs, batch_ys = get_train_batch(M, 60)
    count = 0
    for j in np.random.permutation(len(train_x)):
        sess.run(train_step, feed_dict={x: train_x[j], y_: train_y[j]})
        count += 1
        if count%10 == 0:
            print count,
    print ""

figure(1)
plot(test_accuracies,'r',label='test')
plot(train_accuracies,'b',label='train')
plot(validate_accuracies,'g',label='validate')
legend(loc=4)
ylabel('Accuracy (%)')
xlabel('Number of Iterations')
title('Accuracy of Neural Net')
savefig('Part6Accuracy2.pdf')
close()
