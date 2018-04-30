from numpy import *
from euclideanDistance import euclideanDistance

def kNN(k, X, labels, y):
    # Assigns to the test instance the label of the majority of the labels of the k closest 
	# training examples using the kNN with euclidean distance.

    m = X.shape[0] # number of training examples
    n = X.shape[1] # number of attributes

    distances = zeros(m)
    for i in range(m):
		distances[i] = euclideanDistance(X[i,:], y)
		
    # Sort distances and re-arrange labels based on the distance of the instances
    idx = distances.argsort()
    labels = labels[idx]
	
    c = zeros(max(labels)+1)
	
    # Compute the class labels of the k nearest neigbors
    for i in range(k):
		c[labels[i]] += 1

    # Return the label with the largest number of appearances
    label = argmax(c)
    
    return label

 
