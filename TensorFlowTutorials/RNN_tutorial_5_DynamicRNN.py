from __future__ import print_function, division
import numpy as np
import tensorflow as tf

import matplotlib.pyplot as plt

num_epochs = 100
total_series_length = 50000
truncated_backprop_length = 15
state_size = 4
num_classes = 2
echo_step = 3
batch_size = 5
num_batches = total_series_length//batch_size//truncated_backprop_length

num_layers = 3

# Function to generate data:
def generateData():
	x = np.array(np.random.choice(2, total_series_length, p=[0.5, 0.5]))
	y = np.roll(x, echo_step)
	y[0:echo_step] = 0
	x = x.reshape((batch_size, -1))  
	y = y.reshape((batch_size, -1))
	return (x, y)


# Tensorflow placeholders for input, output and RNN state:
batchX_placeholder = tf.placeholder(tf.float32, [batch_size, truncated_backprop_length])
batchY_placeholder = tf.placeholder(tf.int32, [batch_size, truncated_backprop_length])
init_state = tf.placeholder(tf.float32, [num_layers, 2, batch_size, state_size])

# Tensorflow variables for weights and biases:
W2 = tf.Variable(np.random.rand(state_size, num_classes),dtype=tf.float32) #output layer
b2 = tf.Variable(np.zeros((1,num_classes)), dtype=tf.float32) #output layer

# Construct the multi-layered RNN:
state_per_layer_list = tf.unstack(init_state, axis=0)
rnn_tuple_state = tuple( [ tf.contrib.rnn.LSTMStateTuple( state_per_layer_list[idx][0], state_per_layer_list[idx][1] ) for idx in range(num_layers) ] )

def lstm_cell():
	return tf.contrib.rnn.BasicLSTMCell(state_size, state_is_tuple=True)

stacked_cell = tf.contrib.rnn.MultiRNNCell( [lstm_cell() for _ in range(num_layers)], state_is_tuple=True )

'''
 NOTE: We have not declared anything that can be assigned to inputs yet. Like previous tutorials we do not unstack or split the batchX_placeholder or batchY_placeholer. Instead we use the dynamic RNN API which performs fully dynamic unrolling of inputs. 

On using tf.expand_dims:
tf.expand_dims(input, axis)
This function INSERTS a "1" in the dimension given by the axis key. If axis is given a negative value, it is counted from the end. 
Examples:
If t is a tensor of shape [2, 3, 5], 
expand_dims(t, 0) EXPANDS the dimension of the tensor t to [1, 2, 3, 5]
expand_dims(t, 2) => [2, 3, 1, 5]
expand_dims(t, 3) => [2, 3, 5, 1]

Now, dimensionality of batchX_placeholder is [batch_size, truncated_backprop_length]. So, expand_dims with axis = -1 should result in a dimensionality of [batch_size, truncated_backprop_length, 1]

Why is this to be done?
The dynamic_rnn function takes the batch inputs of shape [batch_size, truncated_backprop_length, input_size]. In this case the input size is 1. Therefore, expand_dims will do the job. 

'''
states_series, current_state = tf.nn.dynamic_rnn(cell=stacked_cell, inputs=tf.expand_dims(batchX_placeholder, -1), initial_state=rnn_tuple_state)
states_series = tf.reshape(states_series, [-1, state_size])
'''
The original shape of the output state_series is [batch_size, truncated_backprop_length, state_size] which is then reshaped to 
[batch_size * truncated_backprop_length, input_size]
This ONLY CONTAINS the hidden_state of the LAST LAYER across all time-steps. Now we dont really care about other layers as we only need the last layer's output to compute the logits at every step. 	

'''
logits = tf.matmul(states_series, W2) + b2 # Not using list comprehension anymore 
# In previous tutorials state_series was basically a list of 5x4 tensors (one for each time step along the truncated_backprop_length dimension). Therefore, we used list comprehension to form the logits by iterating through the state_series list, each time multiplying (5x4) state times (4x2) W2 weights matrix. In the case above, although the matrix multiplicating is compatible, ( (5x15) x4 ) x (4x2), we consider all time steps togther (5x15)

labels = tf.reshape(batchY_placeholder, [-1])
# The above reshaping will change the size from [batch_size, truncated_backprop_length] to [batch_size * truncated_backprop_length]

losses = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits, labels=labels)
# Why can we do this??
# The explanation is in the pciture of this tutorial. Both logits and labels are of shape [batch_size * truncated_backprop_length] and therefore can be directly compared using the cross entropy formula. 

# In all the implementation above we have replaced python lists by tensors. 

# We will only use lists for visualizing and plotting below. Author says this is a quick and dirty solution. So we will have to use unstack like we did in previous tutorials


logits_series = tf.unstack( tf.reshape(logits, [batch_size, truncated_backprop_length, num_classes]), axis=1 )
predictions_series = [tf.nn.softmax(logits) for logits in logits_series]



total_loss = tf.reduce_mean(losses)
train_step = tf.train.AdagradOptimizer(0.3).minimize(total_loss)

# Visualization:
def plot(loss_list, predictions_series, batchX, batchY):
	plt.subplot(2, 3, 1)
	plt.cla()
	plt.plot(loss_list)

	for batch_series_idx in range(5):
		one_hot_output_series = np.array(predictions_series)[:, batch_series_idx, :]
		single_output_series = np.array([(1 if out[0] < 0.5 else 0) for out in one_hot_output_series])
		plt.subplot(2, 3, batch_series_idx + 2)
		plt.cla()
		plt.axis([0, truncated_backprop_length, 0, 2])
		left_offset = range(truncated_backprop_length)
		plt.bar(left_offset, batchX[batch_series_idx, :], width=1, color="blue")
		plt.bar(left_offset, batchY[batch_series_idx, :] * 0.5, width=1, color="red")
		plt.bar(left_offset, single_output_series * 0.3, width=1, color="green")
	
	plt.draw()
	plt.pause(0.0001)

with tf.Session() as sess:
	sess.run(tf.global_variables_initializer())
	plt.ion()
	plt.figure()
	plt.show()
	loss_list = []

	for epoch_idx in range(num_epochs):
		x,y = generateData()

		_current_state = np.zeros((num_layers, 2, batch_size, state_size))
		# Replacing _current_cell_state and _current_hidden_state because we want to use 1 big tensor for the entire multi-layer LSTM network. The "2" refers to the 2 states, cell_state and hidden_state. So for each layer and each sample in a batch, we have both a cell_state and a hidden_state vector of size state_size. 

		print("New data, epoch", epoch_idx)

		for batch_idx in range(num_batches):
			start_idx = batch_idx * truncated_backprop_length
			end_idx = start_idx + truncated_backprop_length
			
			batchX = x[:,start_idx:end_idx]
			batchY = y[:,start_idx:end_idx]

			_total_loss, _train_step, _current_state, _predictions_series = sess.run(
				[total_loss, train_step, current_state, predictions_series],
				feed_dict={
				batchX_placeholder:batchX,
				batchY_placeholder:batchY,
				init_state: _current_state, # The state is now stored in a single tensor
				})
			
			loss_list.append(_total_loss)

			if batch_idx%100 == 0:
				print("Step",batch_idx, "Loss", _total_loss)
				plot(loss_list, _predictions_series, batchX, batchY)
		plt.ioff()
		plt.show()


















