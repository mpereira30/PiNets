#!/usr/bin/env python

from __future__ import print_function, division
import numpy as np
import tensorflow as tf
import csv
import matplotlib.pyplot as plt
import os

os.system('clear') # clears the screen on the terminal

'''
First, we fetch the optimal trajectory:
Now, batch_size = number of rollouts and truncated_backprop_length = length of sequence
'''
fname = 'DDP_Marcus/savedData/data_1'
raw_data = np.genfromtxt(fname, delimiter=",")
truncated_backprop_length = raw_data.shape[1]# length of sequence 
batch_size = 100 # number of rollouts
num_epochs = 200
rnn_layer_size = 32 # number of RNN neurons
fnn_layer_size = 32 # number of FNN neurons
num_targets = 2
num_layers = 3
n = 2 # number of state variables

# placeholders:
X_train = tf.placeholder(tf.float64, [truncated_backprop_length, batch_size, truncated_backprop_length]) # random initial control input 
Y_train = tf.placeholder(tf.float64, [1, truncated_backprop_length]) # target is the optimal control input 


inputs_series = tf.split(X_train, truncated_backprop_length, 1)
target_series = tf.split(Y_train, truncated_backprop_length, 1)

#================================== RNN for dynamics ================================================================================================================

# For initial state layers:
initial_layer = tf.placeholder(tf.float64, [2, batch_size, n]) # 2 because of cell_state and hidden_state matrices
initial_layer_unstack = tf.unstack(initial_layer, axis=0) # Unstack the cell and hidden values 
initial_layer_list = [ tf.contrib.rnn.LSTMStateTuple(initial_layer_unstack[0], initial_layer_unstack[1]) ] # LSTMStateTuple stores 2 elements: (c, h) in that order

# For higher dimensional layers:
inner_layers = tf.placeholder(tf.float64, [num_layers, 2, batch_size, rnn_layer_size])
inner_layers_unstack = tf.unstack(inner_layers, axis=0)
inner_layers_list = [ tf.contrib.rnn.LSTMStateTuple( inner_layers_unstack[idx][0], inner_layers_unstack[idx][1] ) for idx in range(num_layers) ] 
rnn_tuple = tuple( initial_layer_list + inner_layers_list )

# For RNN output layer:
rnn_output_W = tf.Variable(np.random.rand(rnn_layer_size, num_targets), dtype=tf.float64) 
rnn_output_b = tf.Variable(np.zeros((1,num_targets)), dtype=tf.float64)

keep_prob = tf.placeholder(tf.float64, shape=())

rnn_initial_layer = tf.contrib.rnn.BasicLSTMCell(n)
stack_rnn = [rnn_initial_layer]
for _ in range(num_layers):
	stack_rnn.append( tf.contrib.rnn.DropoutWrapper(tf.contrib.rnn.BasicLSTMCell(rnn_layer_size), output_keep_prob = keep_prob) )

stacked_cell = tf.contrib.rnn.MultiRNNCell(stack_rnn, state_is_tuple = True)
states_series, current_state = tf.contrib.rnn.static_rnn(cell=stacked_cell, inputs=inputs_series, initial_state=rnn_tuple)

# State series is a collection of states of the final layer only for each time step which are multiplied by the weights of the last layer (linear activation):
rollout_data_list = [tf.matmul(state, rnn_output_W) + rnn_output_b for state in states_series] #Broadcasted addition

'''
The output of the RNN will be the rollouts:- 
	A python list whose length = truncated_backprop_length = length of control sequence 
	Dimension of each element in this list will be [batch_size] x [num_targets] for each time step
'''

#======================================================================================================================================================================

#================================== FNN for Cost ======================================================================================================================

def Cost_nn(data):
	layer_1 = {'weights': tf.Variable( tf.random_normal(shape=[num_targets, fnn_layer_size], dtype=tf.float64 ) ),
			'biases': tf.Variable( tf.random_normal(shape=[fnn_layer_size], dtype=tf.float64) ) }

	layer_2 = {'weights': tf.Variable( tf.random_normal([fnn_layer_size, fnn_layer_size], dtype=tf.float64) ),
			'biases': tf.Variable( tf.random_normal([fnn_layer_size], dtype=tf.float64) )}

	output_layer = {'weights': tf.Variable( tf.random_normal([fnn_layer_size, 1], dtype=tf.float64) ),
			'biases': tf.Variable( tf.random_normal([fnn_layer_size], dtype=tf.float64) )}

	# The output is the cost for the kth trajectory (or batch) for the ith timestep. 

	l1 = tf.matmul(data, layer_1['weights']) + layer_1['biases']
	l1 = tf.tanh(l1)

	l2 = tf.matmul(l1, layer_2['weights']) + layer_2['biases']
	l2 = tf.nn.tanh(l2)

	output = tf.matmul(l2, output_layer['weights']) + output_layer['biases']

	return output

rollout_cost_list = [Cost_nn(roll_data) for roll_data in rollout_data_list] 
# This should be a list of length = sequence length = truncated_backprop_length. Each element of this list has dimensions - [batch_size] x 1

#=======================================================================================================================================================================

#================================== Control Update =====================================================================================================================


noise_placeholder = tf.placeholder(tf.float64, [batch_size, truncated_backprop_length])
noise_list = tf.split(noise_placeholder, truncated_backprop_length, 1)
control_seq_placeholder = tf.placeholder(tf.float64, [1,truncated_backprop_length])
control_seq = tf.split(control_seq_placeholder, truncated_backprop_length, 1)

def delta_Us(S, N):
	lambda_param = tf.constant(-0.01, dtype=tf.float64)	
	S = tf.multiply(S, lambda_param)
	expS = tf.exp(S)
	denominator = tf.reduce_sum(expS)
	numerator = tf.reduce_sum( tf.multiply(expS,N) )
	return tf.div(numerator, denominator)
	
updates = [delta_Us(roll_cost, noise) for roll_cost, noise in zip(rollout_cost_list, noise_list)]
u_star = [u + del_u for u, del_u in zip(control_seq, updates)]
loss = tf.losses.mean_squared_error(target_series[0], u_star[0])

loss_placeholder = tf.placeholder(tf.float64, [1,truncated_backprop_length])
total_loss = tf.reduce_mean(loss_placeholder)




















