from numpy import *
from sigmoid import sigmoid

def computeCost(theta, X, y): 
	# Computes the cost of using theta as the parameter 
	# for logistic regression. 
    
	m = X.shape[0] # number of training examples
	
	J = 0
	
	for i in range(m):
		J += (-y[i] * log(sigmoid(dot(X[i,:],theta))) - (1 - y[i]) * log(1 - sigmoid(dot(X[i,:],theta))))
		
	J /= m
	
	return J