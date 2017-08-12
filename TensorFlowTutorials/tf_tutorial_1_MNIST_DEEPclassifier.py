#!/usr/bin/env python
import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
import numpy as np 
import tensorflow as tf

from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

x = tf.placeholder(tf.float32, [None, 784])
y_ = tf.placeholder(tf.float32, [None,10])

# Initialize weights and biases with a small amount of noise:
def weight_variable(shape):
	initial = tf.truncated_normal(shape, stddev=0.1)
	return tf.Variable(initial)
	
# When using ReLU neurons, initialize with a slightly positive bias in order to avoid dead neurons	
def bias_variable(shape):
	initial = tf.constant(0.1, shape=shape)
	return tf.Variable(initial)

# By doing zero padding we ensure that the output of convolution has same size as the input:
def conv2d(x, W):
	return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

# Using simple max pooling over 2x2 blocks:	
def max_pool_2x2(x):
	return tf.nn.max_pool(x, ksize=[1,2,2,1], strides=[1,2,2,1], padding='SAME')

	
# Lets define the layers of the ConvNet:
W_conv1 = weight_variable([5,5,1,32])
b_conv1 = bias_variable([32])

# Reshape the input image into a 4d tensor in order to apply it to the first ConV layer above:
x_image = tf.reshape(x ,[-1,28,28,1])

h_conv1 = tf.nn.relu(conv2d(x_image,W_conv1) + b_conv1)
h_pool1 = max_pool_2x2(h_conv1) # This reduces the image size to 14x14

# Second convolutional layer:
W_conv2 = weight_variable([5,5,32,64])
b_conv2 = bias_variable([64])
h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)
h_pool2 = max_pool_2x2(h_conv2) # This reduces the image size to 7x7

# Fully connected layer of 1024 neurons:
W_fc1 = weight_variable([7*7*64, 1024])
b_fc1 = bias_variable([1024])

# reshape the tensor from the 2nd pooling layer into a batch of vectors:
h_pool2_flat = tf.reshape(h_pool2, [-1, 7*7*64])
h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)

# Add a dropout layer to prevent overfitting:
keep_prob = tf.placeholder(tf.float32) # the probability that a neuron's output is kept during dropout
h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

# Final layer:
W_fc2 = weight_variable([1024,10])
b_fc2 = bias_variable([10])

y_conv = tf.matmul(h_fc1_drop, W_fc2) + b_fc2

# Now train the model:
cross_entropy = tf.reduce_mean( tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y_conv) )
train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy) 
correct_prediction = tf.equal( tf.argmax(y_conv,1), tf.argmax(y_,1) )
accuracy = tf.reduce_mean( tf.cast(correct_prediction, tf.float32) )

with tf.Session() as sess:
	sess.run(tf.global_variables_initializer())
	for i in range(20000):
		batch = mnist.train.next_batch(50)	
		if i%100 == 0:
			train_accuracy = accuracy.eval(feed_dict={x:batch[0], y_:batch[1], keep_prob:0.5})
			print('step %d, training accuracy %g' % (i, train_accuracy))
		train_step.run(feed_dict={x:batch[0], y_:batch[1], keep_prob:0.5})
	print('test accuracy %g' % accuracy.eval(feed_dict={x:mnist.test.images, y_:mnist.test.labels, keep_prob:1.0})) # keep_prob = 1.0 means no dropout

