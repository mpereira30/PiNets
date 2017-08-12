#!/usr/bin/env python

import theano
import numpy

# Input Variables: 
x = theano.tensor.fvector('x')
target =  theano.tensor.fscalar('target')

W = theano.shared(numpy.asarray([0.2, 0.7]), 'W')
y = (x * W).sum()
 
# Cost function:
cost = theano.tensor.sqr(target-y) 
# Gradients of cost wrt the weights:
gradients = theano.tensor.grad(cost,[W])
# Update the weights using gradient descent:
W_updated = W - (0.1*gradients[0])
# Create a list of updates - it is required to be a list of tuples:
updates = [(W, W_updated)]
 
# We have a list of 2 inputs - x and target, One output y 
f = theano.function([x,target], y, updates=updates)

for i in xrange(10):
	output = f([1.0, 1.0], 20.0)
	print output

	
