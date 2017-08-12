#!/usr/bin/env python

import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'

import numpy as np 
import tensorflow as tf

from tensorflow.examples.tutorials.mnist import input_data
from tensorflow.contrib import rnn 

mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

hm_epochs = 10 # how many epochs?
n_classes = 10
batch_size = 128

chunk_size = 28
n_chunks = 28

rnn_size = 128

x = tf.placeholder('float', [None, n_chunks, chunk_size])
y = tf.placeholder('float')

# Define the model (computational graph)
def RNN_model(x):
	layer = {'weights':tf.Variable(tf.random_normal([rnn_size, n_classes])), 'biases':tf.Variable(tf.random_normal([n_classes]))}

	x = tf.transpose(x, [1,0,2])
	'''
	The above performs transpose such that the resulting tensor first two dimensions and keeps
	the third as it is
	eg. x = np.ones(1,2,3)
		x = np.transpose(x, [1,0,2])
		x.shape => (2,1,3)
	This is done to match the input requirements of the rnn_cell 
	
	What I think: In previous MNIST examples, the x placeholder has dim - [None, 784] so this is like (1,784,...)
	When we swap the 1st 2 dimensions we would get (784,1,...)
	'''
	x = tf.reshape(x, [-1, chunk_size]) 
	'''
	If a dimension is given as -1 in a reshaping operation, the other dimensions are automatically calculated.
	So here, the above (784,1) vector will be reshaped into a matrix such that the columns will be 28. I feel 
	that the rows would end up becoming 28 too i.e. a 28 x 28 pixel image
	'''
	x = tf.split(x, n_chunks, 0)	
	'''
	Splits a tensor into sub tensors
	We want to now split this into 28 chunks (or 28 arrays or row vectors) because we are splitting along the
	0th axis -> rows
	'''	
	
	lstm_cell = rnn.BasicLSTMCell(rnn_size) 
	'''
	__init__(num_units, forget_bias=1.0, input_size=None, state_is_tuple=True, activation=tf.tanh)
	This will initialize the basic LSTM cell
	In tensorflow, 1 cell is not 1 object but an array of objects or units.
	This creates an instance of a RNNCell which is passed to the function below to create a simple RNN
	'''
	outputs, states = rnn.static_rnn(lstm_cell, x, dtype=tf.float32)
	
	ouput = tf.matmul(outputs[-1], layer['weights']) + layer['biases'] # Use the final ouput
	
	return ouput

def train_neural_network(x):
	prediction = RNN_model(x)
	cost = tf.reduce_mean( tf.nn.softmax_cross_entropy_with_logits(logits=prediction, labels=y) )
	optimizer = tf.train.AdamOptimizer().minimize(cost)				

	with tf.Session() as sess:
		sess.run(tf.global_variables_initializer())
		
		for epoch in range(hm_epochs):
			epoch_loss = 0
			for _ in range(int(mnist.train.num_examples/batch_size)):
				epoch_x, epoch_y = mnist.train.next_batch(batch_size)
				epoch_x = epoch_x.reshape(batch_size,n_chunks,chunk_size) # CHANGED from previous code
				'''
				epoch_x, epoch_y = mnist.train.next_batch(batch_size) gives (128,784) to epoch_x
				where 128 = batch_size
				
				So, when we call reshape on epoch_x, the number of rows remains the same = 128, but instead of a
				row vector of 784 pixels, we convert it into a 2D image - 28x28 (in 2nd and 3rd dim of epoch_x)
				'''
				_, c = sess.run([optimizer, cost], feed_dict={x:epoch_x, y:epoch_y})
				epoch_loss += c
			print('Epoch', epoch, 'completed out of', hm_epochs, 'loss:', epoch_loss)	
		
		correct = tf.equal( tf.argmax(prediction,1), tf.argmax(y,1) )
		accuracy = tf.reduce_mean( tf.cast(correct, 'float') )
		print('Accuracy:', accuracy.eval({x:mnist.test.images.reshape((-1,n_chunks,chunk_size)), y:mnist.test.labels}) )
		
	
train_neural_network(x)
	
	