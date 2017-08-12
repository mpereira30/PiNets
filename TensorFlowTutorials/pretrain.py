#!/usr/bin/env python

from __future__ import print_function, division
import numpy as np
import tensorflow as tf
import csv
import matplotlib.pyplot as plt

truncated_backprop_length = 15
num_batches = 40
layer_size = 32 # number of RNN neurons
num_targets = 2
num_layers = 2
n = 2 # number of state variables
batch_size = 1
num_epochs = 50

# Train the network with 1 trajectory at a time:
X_train = tf.placeholder(tf.float64, [batch_size, truncated_backprop_length-1]) # control input is 1x1
Y_train = tf.placeholder(tf.float64, [truncated_backprop_length-1, n*batch_size]) # target is next state (2x1)

# For output layer:
W2 = tf.Variable(np.random.rand(layer_size, num_targets), dtype=tf.float64) 
b2 = tf.Variable(np.zeros((1,num_targets)), dtype=tf.float64) 

inputs_series = tf.split(X_train, truncated_backprop_length-1, 1)
target_series = tf.unstack(Y_train, axis=0)

# For initial state layers:
initial_layer = tf.placeholder(tf.float64, [2, batch_size, n]) # 2: cell_state and hidden_state
initial_layer_unstack = tf.unstack(initial_layer, axis=0) # Unstack the cell and hidden values 
initial_layer_list = [ tf.contrib.rnn.LSTMStateTuple(initial_layer_unstack[0], initial_layer_unstack[1]) ] # LSTMStateTuple stores 2 elements: (c, h) in that order

# For higher dimensional layers:
inner_layers = tf.placeholder(tf.float64, [num_layers, 2, batch_size, layer_size])
inner_layers_unstack = tf.unstack(inner_layers, axis=0)
inner_layers_list = [ tf.contrib.rnn.LSTMStateTuple( inner_layers_unstack[idx][0], inner_layers_unstack[idx][1] ) for idx in range(num_layers) ] 
rnn_tuple = tuple( initial_layer_list + inner_layers_list )

rnn_initial_layer = tf.contrib.rnn.BasicLSTMCell(n)
stack_rnn = [rnn_initial_layer]
for _ in range(num_layers):
	stack_rnn.append(tf.contrib.rnn.BasicLSTMCell(layer_size))

stacked_cell = tf.contrib.rnn.MultiRNNCell(stack_rnn, state_is_tuple = True)
states_series, current_state = tf.contrib.rnn.static_rnn(cell=stacked_cell, inputs=inputs_series, initial_state=rnn_tuple)

# State series is a collection of states of the final layer only for each time step which are multiplied by the weights of the last layer (linear activation):
logits_series = [tf.matmul(state, W2) + b2 for state in states_series] #Broadcasted addition
losses = [tf.nn.l2_loss(logits-targets) for logits, targets in zip(logits_series,target_series)]

total_loss = tf.reduce_mean(losses)
train_step = tf.train.AdamOptimizer().minimize(total_loss)	

with tf.Session() as sess:
	sess.run(tf.global_variables_initializer())
	loss_list = []

	for epoch_idx in range(num_epochs):
		print("Epoch - ", epoch_idx+1)

		for i in range(1, num_batches+1):
			fname = 'Data_InvPend/data_' + str(i)
			raw_data = np.genfromtxt(fname, delimiter=",")
			num_timesteps = raw_data.shape[1]
			num_mini_batches = num_timesteps//truncated_backprop_length

			for bidx in range(num_mini_batches):
				
				_inner_layers = np.zeros((num_layers, 2, batch_size, layer_size))
				
				start_idx = bidx * truncated_backprop_length
				end_idx = start_idx + truncated_backprop_length

				u = np.zeros((batch_size, truncated_backprop_length-1))
				u[:] = raw_data[2, start_idx:end_idx-1]
				y = np.transpose(raw_data[0:2, (start_idx+1):end_idx])
				x0 = np.zeros( (2, batch_size, n) )
				x0[0] = x0[1] = np.array( [ raw_data[0,start_idx], raw_data[1,start_idx] ] )

				_total_loss, _train_step = sess.run(
					[total_loss, train_step],
					feed_dict={
					X_train: u,
					Y_train: y,
					initial_layer: x0,
					inner_layers: _inner_layers,
					})
				
				loss_list.append(_total_loss)
				plt.figure(1)
				plt.plot(loss_list[:])
				plt.draw()
				plt.pause(0.0001)
				plt.show(block=False)



