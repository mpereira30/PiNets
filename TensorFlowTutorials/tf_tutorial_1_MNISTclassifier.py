#!/usr/bin/env python

import numpy as np 
import tensorflow as tf

from tensorflow.examples.tutorials.mnist import input_data

# A one-hot vector is a vector which is 0 in most dimensions, and 1 in a single dimension
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

# MNIST images are 28x28 pixels which are flattened into a 784-dimensional vector 
x = tf.placeholder(tf.float32, [None, 784]) # none means that the dimension can be of any length which allows to input ANY number of images

W = tf.Variable(tf.zeros([784,10])) # input : 784-dim vector of pixels, output : 10 classes of numbers from 0 - 9 
b = tf.Variable(tf.zeros([10]))

'''
RAW FORMULATION OF CROSS ENTROPY (can be numerically unstable)

y = tf.nn.softmax(tf.matmul(x,W) + b)

# targets for training:
y_ = tf.placeholder(tf.float32, [None,10])

# using the cross entropy loss function:
cross_entropy = tf.reduce_mean( -tf.reduce_sum(y_ * tf.log(y), reduction_indices=[1]) )

# tf.reduce_sum adds elements in the second dimension of y [None,10] given by reduction_indices=[1]
# tf.reduce_mean finds the mean over all samples in one batch

'''

y = tf.matmul(x,W) + b
y_ = tf.placeholder(tf.float32, [None,10])
cross_entropy = tf.reduce_mean( tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y) )
train_step = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy) # learning rate = 0.05
# This does 1 step of gradient descent training

sess = tf.InteractiveSession()

# First initialize the created variables above:
tf.global_variables_initializer().run()

# Perform 1000 iterations:
for _ in range(1000):
    batch_xs, batch_ys = mnist.train.next_batch(100)
    sess.run(train_step, feed_dict={x: batch_xs, y_: batch_ys})

# Evaluation:
# Using tf.argmax which gives the INDEX of the highest entry in a tensor along some axis

correct_prediction = tf.equal( tf.argmax(y,1), tf.argmax(y_,1) ) # tf.equal gives a tensor of booleans which we cast into floats below and then take the average. If both vectors are equal, the average should equal 1 or 100% accuracy
accuracy = tf.reduce_mean( tf.cast(correct_prediction, tf.float32) )

print( sess.run(accuracy, feed_dict={x:mnist.test.images, y_:mnist.test.labels}) )







