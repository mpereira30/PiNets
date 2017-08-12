from __future__ import print_function, division
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

#====== Part that remains unchanged ========================================================================

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
init_state = tf.placeholder(tf.float32, [batch_size, state_size])

# Tensorflow variables for weights and biases:
W2 = tf.Variable(np.random.rand(state_size, num_classes),dtype=tf.float32) #output layer
b2 = tf.Variable(np.zeros((1,num_classes)), dtype=tf.float32) #output layer

#========================================================================================================================
#========================================================================================================================
#======================= New (Using TF API) =============================================================================


# Unpack columns: 
inputs_series = tf.split(batchX_placeholder, truncated_backprop_length, 1)
'''
Why use tf .split and not tf.unstack?
The tf.nn.rnn accepts a list of inputs of shape [batch_size, input_size], and the input_size is simply one in our case (input is just a series of scalars). Split does not remove the singular dimension, but unstack does. Therefore, here in the forward pass there would be no need to re-assert or reshape the input to the rnn.
'''
labels_series = tf.unstack(batchY_placeholder, axis=1)

# Forward passes
cell = tf.contrib.rnn.BasicRNNCell(state_size)
states_series, current_state = tf.contrib.rnn.static_rnn(cell=cell, inputs=inputs_series, initial_state=init_state)
'''
We don't have to implement the forward pass of the sequence using a for-loop !!!
The tf.nn.rnn unrolls the RNN and creates the graph automatically, so we can remove the for-loop. The function returns a series of previous states as well as the last state in the same shape
'''
#========================================================================================================================
#============== Back to previous code ===================================================================================

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



