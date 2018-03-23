from pylab import *
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cbook as cbook
from scipy.misc import imread
from scipy.misc import imresize
import matplotlib.image as mpimg
import cPickle

import os

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


#os.pathdchd('/Desktop/CSC321/Assignment3')
trainAccuracy = []
testAccuracy = []
xRange = range(200)

x = tf.placeholder(tf.float32, (None,32*32*3))


y = tf.nn.softmax(layer2)
y_ = tf.placeholder(tf.float32, [None, 6])


layer1 = tf.nn.tanh(tf.matmul(x, W0)+b0)
layer2 = tf.matmul(layer1, W1)+b1

init = tf.initialize_all_variables()
sess = tf.Session()
sess.run(init)


for i in xRange:
    snapshot = cPickle.load(open("snapshhots1/new_snapshot"+str(i)+".pkl"))
    W0 = tf.Variable(snapshot['W0'])
    b0 = tf.Variable(snapshot['b0'])
    W1 = tf.Variable(snapshot['W1'])
    b1 = tf.Variable(snapshot['b1'])
    
    trainA = sess.run(accuracy, feed_dict={x: test_x, y_: test_y})
    batch_xs, batch_ys = get_train(M)
    
    print "Train:", sess.run(accuracy, feed_dict={x: batch_xs, y_: batch_ys})
    print "Penalty:", sess.run(decay_penalty)
    
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

    