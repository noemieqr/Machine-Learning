import numpy as np
from euclideanDistance import euclideanDistance
from simpleInitialization import simpleInitialization
#from kmeansppInitialization import kmeansppInitialization

# Returns True or False if k-means is done. K-means terminates either
# because it has run a maximum number of iterations OR the centroids
# stop changing.
def shouldStop(oldCentroids, centroids, iterations):
    if iterations > 100 or np.array_equal(oldCentroids,centroids): 
        return True
    else:
        return False

# K-Means is an algorithm that takes as input a dataset and a constant
# k and returns k centroids (which define clusters of data in the
# dataset which are similar to one another).
def kmeans(X, k):
    # Intialize centroids
    centroids = simpleInitialization(X, k)
    #centroids = kmeansppInitialization(X,k)
    
    # Initialize book keeping vars.
    iterations = 0
    oldCentroids = None
    
    # Run the main k-means algorithm
    while not shouldStop(oldCentroids, centroids, iterations):
        # Save old centroids for convergence test.
        oldCentroids = centroids
        iterations += 1
        
        # Compute distances from the centroid points
        distances = np.array([[euclideanDistance(Xi,centroid) for centroid in centroids]
                              for Xi in X])
        # Compute nearest centroid indices
        labels = np.array([np.argmin(distance) for distance in distances])
        # Find new centroids
        centroids = [[np.sum(col)/len(col) for col in np.transpose(X[labels==i])] 
                     for i in range(k)]
        
    return labels
