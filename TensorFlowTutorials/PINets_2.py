#!/usr/bin/env python

from __future__ import print_function, division
import numpy as np
import tensorflow as tf
import csv
import matplotlib.pyplot as plt
import os
import time 

np.random.seed(None)
# os.system('clear') # clears the screen on the terminal

'''
First, we fetch the optimal trajectory:
Now, batch_size = number of rollouts and truncated_backprop_length = length of sequence
'''
fname = 'Data_InvPend/data_1'
raw_data = np.genfromtxt(fname, delimiter=",")
truncated_backprop_length = raw_data.shape[1]# length of sequence 
batch_size = 100 # number of rollouts
num_epochs = 200
rnn_layer_size = 32 # number of RNN neurons
fnn_layer_size = 32 # number of FNN neurons
num_targets = 2
num_layers = 1
n = 2 # number of state variables

# Create all placeholders:
X_train = tf.placeholder(tf.float64, [truncated_backprop_length, batch_size, truncated_backprop_length])  
Y_train = tf.placeholder(tf.float64, [1, truncated_backprop_length])  
noise_placeholder = tf.placeholder(tf.float64, [truncated_backprop_length, batch_size, truncated_backprop_length])
control_seq_placeholder = tf.placeholder(tf.float64, [truncated_backprop_length, truncated_backprop_length])
initial_layer = tf.placeholder(tf.float64, [truncated_backprop_length, 2, batch_size, n]) 
# 2 because of cell_state and hidden_state matrices
inner_layers = tf.placeholder(tf.float64, [num_layers, 2, batch_size, rnn_layer_size])
keep_prob = tf.placeholder(tf.float64, shape=())

# Unstack the placeholders:
X_ = tf.unstack(X_train, axis=0)
noise_ = tf.unstack(noise_placeholder, axis=0)
control_seq_ = tf.split(control_seq_placeholder, truncated_backprop_length, axis=0)
initial_layer_ = tf.unstack(initial_layer, axis=0)

# Further split/unstack:
target_series = tf.split(Y_train, truncated_backprop_length, 1)



#---------------Rollout Generation-----------------------------------------------------------------------------------------------------



def dynamics_RNN(input_, init_layer_):
	
	inputs_series = tf.split(input_, truncated_backprop_length, 1)

	# For initial state layers:
	initial_layer_unstack = tf.unstack(init_layer_, axis=0) # Unstack the cell and hidden values 
	initial_layer_list = [ tf.contrib.rnn.LSTMStateTuple(initial_layer_unstack[0], initial_layer_unstack[1]) ] # LSTMStateTuple stores 2 elements: (c, h) in that order

	# For higher dimensional layers:
	inner_layers_unstack = tf.unstack(inner_layers, axis=0)
	inner_layers_list = [ tf.contrib.rnn.LSTMStateTuple( inner_layers_unstack[idx][0], inner_layers_unstack[idx][1] ) for idx in range(num_layers) ] 
	rnn_tuple = tuple( initial_layer_list + inner_layers_list )

	# For RNN output layer:
	rnn_output_W = tf.Variable(np.random.rand(rnn_layer_size, num_targets), dtype=tf.float64) 
	rnn_output_b = tf.Variable(np.zeros((1,num_targets)), dtype=tf.float64)

	rnn_initial_layer = tf.contrib.rnn.BasicLSTMCell(n)
	stack_rnn = [rnn_initial_layer]
	for _ in range(num_layers):
		stack_rnn.append( tf.contrib.rnn.DropoutWrapper(tf.contrib.rnn.BasicLSTMCell(rnn_layer_size), output_keep_prob = keep_prob) )

	stacked_cell = tf.contrib.rnn.MultiRNNCell(stack_rnn, state_is_tuple = True)
	states_series, current_state = tf.contrib.rnn.static_rnn(cell=stacked_cell, inputs=inputs_series, initial_state=rnn_tuple)

	# State series is a collection of states of the final layer only for each time step which are multiplied by the weights of the last layer (linear activation):
	return [tf.matmul(state, rnn_output_W) + rnn_output_b for state in states_series] #Broadcasted addition

	'''
	The output of the RNN will be the rollouts:- 
		A python list whose length = truncated_backprop_length = length of control sequence 
		Dimension of each element in this list will be [batch_size] x [num_targets] for each time step
	'''

giant_rollout_data_list = [None] * truncated_backprop_length
# This is a list of LISTS of length truncated_backprop_length, wherein each list is as described in the comment above.

with tf.variable_scope("RNN") as scope_rnn:
	for u_i in range(truncated_backprop_length):
		giant_rollout_data_list[u_i] = dynamics_RNN(X_[u_i], initial_layer_[u_i]) 
		scope_rnn.reuse_variables()



#---------------Cost Calculation-------------------------------------------------------------------------------------------------------



def cost_FNN(data):

	L1_W = tf.get_variable( name="layer_1_weights", shape=[num_targets, fnn_layer_size], dtype=tf.float64, initializer=tf.random_normal_initializer() )
	L1_B = tf.get_variable( name="layer_1_biases", shape=[fnn_layer_size], dtype=tf.float64, initializer=tf.zeros_initializer() )

	L2_W = tf.get_variable( name="layer_2_weights", shape=[fnn_layer_size, fnn_layer_size], dtype=tf.float64, initializer=tf.random_normal_initializer() )
	L2_B = tf.get_variable( name="layer_2_biases", shape=[fnn_layer_size], dtype=tf.float64, initializer=tf.zeros_initializer() )	

	O_W = tf.get_variable( name="output_layer_weights", shape=[fnn_layer_size, 1], dtype=tf.float64, initializer=tf.random_normal_initializer() )
	O_B = tf.get_variable( name="output_layer_biases", shape=[1,1], dtype=tf.float64, initializer=tf.zeros_initializer() )	

	l1 = tf.matmul(data, L1_W) + L1_B
	l1 = tf.tanh(l1)

	l2 = tf.matmul(l1, L2_W) + L2_B
	l2 = tf.nn.tanh(l2)

	output = tf.matmul(l2, O_W) + O_B	

	return output

giant_rollout_cost_list = [None] * truncated_backprop_length

with tf.variable_scope("FNN") as scope_fnn:
	
	for u_i in range(truncated_backprop_length):
		
		rollout_data_list = giant_rollout_data_list[u_i]
		rollout_cost_list = [None] * truncated_backprop_length

		for indx in range(truncated_backprop_length):

			rollout_cost_list[indx] = cost_FNN( rollout_data_list[indx] ) 
			'''
			The first time cost_FNN is called, get_variable will behave as tf.Variable and creates the variables for the weights and biases. All subsequent calls will try to fetch the same variables. This is when it will check if the variables are to be resued or not.			
			'''
			scope_fnn.reuse_variables()

		giant_rollout_cost_list[u_i] = rollout_cost_list



#---------------Control Update-------------------------------------------------------------------------------------------------------



def delta_Us(S, N):
	lambda_param = tf.constant(-0.01, dtype=tf.float64)	
	S = tf.multiply(S, lambda_param)
	expS = tf.exp(S)
	denominator = tf.reduce_sum(expS)
	numerator = tf.reduce_sum( tf.multiply(expS,N) )
	return tf.div(numerator, denominator)

loss_list = [None] * truncated_backprop_length
u_star = [None] * truncated_backprop_length

for i in range(truncated_backprop_length):
	rollout_cost_list = giant_rollout_cost_list[i]
	noise_list = tf.split(noise_[i], truncated_backprop_length, 1)
	control_seq = tf.split(control_seq_[i], truncated_backprop_length, 1)

	updates = [delta_Us(roll_cost, noise) for roll_cost, noise in zip(rollout_cost_list, noise_list)]
	u_star[i] = [u + del_u for u, del_u in zip(control_seq, updates)]
	loss_list[i] = tf.losses.mean_squared_error(target_series[i], u_star[i][0])

u_new = u_star

total_loss = tf.reduce_sum(loss_list)
train_step = tf.train.RMSPropOptimizer(0.001).minimize(total_loss)



#--------------Actual Implementation---------------------------------------------------------------------------------------------------

with tf.Session() as sess:
	sess.run(tf.global_variables_initializer())

	ui = np.zeros((truncated_backprop_length, batch_size, truncated_backprop_length))
	y = np.zeros((1,truncated_backprop_length))
	u = np.zeros((truncated_backprop_length, truncated_backprop_length))
	x0 = np.zeros((truncated_backprop_length, 2, batch_size, n))
		
	# Target optimal MPC control:
	y[0,:] = raw_data[2,:]

	init_u = np.random.randn(1, truncated_backprop_length) # Initial control sequence

	for i in range(truncated_backprop_length): 

		# All rollouts will have the same starting point
		x0[i,0,:,0], x0[i,0,:,1] = raw_data[0,i], raw_data[1,i] # for cell state
		x0[i,1,:,0], x0[i,1,:,1] = raw_data[0,i], raw_data[1,i] # for hidden state

		u[i, 0:(truncated_backprop_length-i)] = init_u[0, i:truncated_backprop_length]
		
	for epoch_idx in range(num_epochs):
		
		_inner_layers = np.zeros((num_layers, 2, batch_size, rnn_layer_size))
		
		E = 0.005 * np.random.randn(truncated_backprop_length, batch_size, truncated_backprop_length)
		for i in range(truncated_backprop_length):
			ui[i] =  E[i] + u[i,:] # broadcasted addition

		
		_total_loss, _train_step, _u_star = sess.run([total_loss, train_step, u_new],
			feed_dict={
			X_train: ui,
			Y_train: y,
			initial_layer: x0,
			inner_layers: _inner_layers,
			keep_prob: 1.0,
			noise_placeholder: E,
			control_seq_placeholder: u
			})	

		for i in range(truncated_backprop_length):
			u[i,:] = np.reshape(np.asarray( _u_star[i][0:truncated_backprop_length] ), (1, truncated_backprop_length)) 
		
		np.reshape(u, (truncated_backprop_length,truncated_backprop_length))	
			
		print("Epoch:",epoch_idx+1,",Loss:",_total_loss)









