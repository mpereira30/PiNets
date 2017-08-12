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

# Tensorflow variables for weights and biases:
W2 = tf.Variable(np.random.rand(state_size, num_classes),dtype=tf.float32) #output layer
b2 = tf.Variable(np.zeros((1,num_classes)), dtype=tf.float32) #output layer

inputs_series = tf.split(batchX_placeholder, truncated_backprop_length, 1)
# This splits the batchX matrix into a list of batch_size column vectors, because we are unstacking along the column axis. The RNN will simultaneously be training on different parts in the time-series (each row is small piece from different place in the timeseries). Hence the name input_series -> not 1 sequence but series.
labels_series = tf.unstack(batchY_placeholder, axis=1)

# Forward pass
#======== LSTM specific declarations =============================================================================

# We DON'T USE LSTM_TUPPLE for each state now. Because we have multiple layers. Instead we use 1 big tensor to hold all states for efficient store complexity. 
init_state = tf.placeholder(tf.float32, [num_layers, 2, batch_size, state_size])

# But now we cannot directly use the same API as in the previous 1 layer LSTM code. Since the TF Multilayer-LSTM-API accepts the state as a tuple of LSTMtuples, therefore the above init_state structure will have to be unpacked before inputing into the RNN network. Then just like in the previous program, we create an LSTMtuple for each layer. 

state_per_layer_list = tf.unstack(init_state, axis=0)

# using list comprehension to extract the cell_state and hidden_state and form the LSTMtuples. Then we form a tuple of these LSTMtuples using the outermost tuple():
# LSTMStateTuple stores 2 elements: (c, h) in that order
rnn_tuple_state = tuple( [ tf.contrib.rnn.LSTMStateTuple( state_per_layer_list[idx][0], state_per_layer_list[idx][1] ) for idx in range(num_layers) ] )

'''
def lstm_cell():
	return tf.contrib.rnn.BasicLSTMCell(state_size, state_is_tuple=True)

# Now create a stacked cell by using lsit comprehension:
stacked_cell = tf.contrib.rnn.MultiRNNCell( [lstm_cell() for _ in range(num_layers)], state_is_tuple=True )
'''

# Alternative way to stack RNN cells:
rnn_cell_1 = tf.contrib.rnn.BasicLSTMCell(state_size)
rnn_cell_2 = tf.contrib.rnn.BasicLSTMCell(state_size)
rnn_cell_3 = tf.contrib.rnn.BasicLSTMCell(state_size)
stack_rnn = [rnn_cell_1]
stack_rnn.append(rnn_cell_2)
stack_rnn.append(rnn_cell_3)
stacked_cell = tf.contrib.rnn.MultiRNNCell(stack_rnn, state_is_tuple = True)

states_series, current_state = tf.contrib.rnn.static_rnn(cell=stacked_cell, inputs=inputs_series, initial_state=rnn_tuple_state)

#=================================================================================================================

logits_series = [tf.matmul(state, W2) + b2 for state in states_series] #Broadcasted addition
predictions_series = [tf.nn.softmax(logits) for logits in logits_series]
# This prediction mayb to just visualize the output at every step in the sequence

losses = [tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits, labels=labels) for logits, labels in zip(logits_series,labels_series)]
# Here softmax is calculated internally before doing cross entropy loss calculation
# Also, here I would guess losses would be a list where each elements is a (5 x 1) column vector. In all it would be a matrix of size  (batch_size x truncated_backprop_length)

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

