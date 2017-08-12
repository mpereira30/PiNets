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
W = tf.Variable(np.random.rand(state_size+1, state_size), dtype=tf.float32)
b = tf.Variable(np.zeros((1,state_size)), dtype=tf.float32)
W2 = tf.Variable(np.random.rand(state_size, num_classes),dtype=tf.float32) #output layer
b2 = tf.Variable(np.zeros((1,num_classes)), dtype=tf.float32) #output layer

inputs_series = tf.unstack(batchX_placeholder, axis=1) 
# This splits the batchX matrix into a list of batch_size column vectors, because we are unstacking along the column axis. The RNN will simultaneously be training on different parts in the time-series (each row is small piece from different place in the timeseries). Hence the name input_series -> not 1 sequence but series.
labels_series = tf.unstack(batchY_placeholder, axis=1)

init_state = tf.placeholder(tf.float32, [batch_size, state_size])
# Because of the unstacking above and simulataneous training on batches of parts of the series, this requires us to save batch_size number of instances of states when propagating the RNN forward. Therefore, the initial state placeholder is of the size (batch_size x state_size). So we save one instance of the state per batch. 

# Forward pass
current_state = init_state
states_series = []
for current_input in inputs_series:
	current_input = tf.reshape(current_input, [batch_size, 1]) # I think this should already be of the shape batch_size x 1
	# However, tf.unstack simply returns a list of tensor objects. So, I guess we assert the shape of the tensor object above.
	input_and_state_concatenated = tf.concat([current_input, current_state], 1)  # Increasing number of columns
	next_state = tf.tanh(tf.matmul(input_and_state_concatenated, W) + b)  # Broadcasted addition of the biases
	states_series.append(next_state)
	current_state = next_state

'''
Note about broadcasting in Numpy:
The size of the resulting array is the maximum size along each dimension of the input arrays.
In this code, addition by broadcasting is done in 2 places:
	1. to compute the next state
	2. to compute the logit 
So dimension of,
	1. next_state is : batch_size x state_size -> (5 x 4)
	2. logit is : batch_size x num_classes -> (5 x 2)
The dimensions with size 1 are stretched or “copied” to match the other. So in case 1, (1x4) bias vector would become (5x4) matrix.
'''

logits_series = [tf.matmul(state, W2) + b2 for state in states_series] #Broadcasted addition
predictions_series = [tf.nn.softmax(logits) for logits in logits_series]
# This prediction maybe to just visualize the output at every step in the sequence

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
	sess.run(tf.initialize_all_variables())
	plt.ion()
	plt.figure()
	plt.show()
	loss_list = []

	for epoch_idx in range(num_epochs):
		x,y = generateData()
		_current_state = np.zeros((batch_size, state_size))

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
				init_state:_current_state
				})

			loss_list.append(_total_loss)

			if batch_idx%100 == 0:
				print("Step",batch_idx, "Loss", _total_loss)
				plot(loss_list, _predictions_series, batchX, batchY)
		plt.ioff()
		plt.show()

