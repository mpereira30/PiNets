#!/usr/bin/env python

from __future__ import print_function, division
import numpy as np
import csv
import matplotlib.pyplot as plt

num_trajectories = 40
truncated_backprop_length = 15
batch_size = 5
u_inputs = []
y_targets = []
initial_inputs = []
n = 2

for i in range(1, num_trajectories+1):
	
	fname = 'Data_InvPend/data_' + str(i)
	raw_data = np.genfromtxt(fname, delimiter=",")
	num_timesteps = raw_data.shape[1]
	num_mini_batches = num_timesteps//truncated_backprop_length

	for bidx in range(num_mini_batches):
		start_idx = bidx * truncated_backprop_length
		end_idx = start_idx + truncated_backprop_length

		u = np.zeros((1, truncated_backprop_length-1))
		u[:] = raw_data[2, start_idx:end_idx-1]
		y = raw_data[0:2, (start_idx+1):end_idx]
		x0 = np.zeros((1,n))
		x0[0,0], x0[0,1] = raw_data[0, start_idx], raw_data[0, start_idx] 
		u_inputs.append(u)
		y_targets.append(y)
		initial_inputs.append(x0)


print(u_inputs[0].shape)
print(y_targets[0].shape)
print(initial_inputs[0].shape)
print(len(u_inputs))
print(len(y_targets))
print(len(initial_inputs))
