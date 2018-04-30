from numpy import *
from sigmoid import sigmoid

def computeGrad(theta, X, y):
	# Computes the gradient of the cost with respect to
	# the parameters.
	
	m = X.shape[0] # number of training examples
	
	grad = zeros(size(theta)) # initialize gradient
	
	for i in range(theta.shape[0]):
	    for j in range(m):
	        grad[i] += (sigmoid(dot(X[j,:],theta)) - y[j]) * X[j,i]
			
	grad /= m
	
	return grad