from numpy import *
from kmeans import kmeans

def spectralClustering(W, k):
	# Create degree matrix
	D = diag(sum(W, axis=0))
	# Create Laplacian matrix
	L = D - W
	eigval, eigvec = linalg.eig(L) # Calculate eigenvalues and eigenvectors
	eigval = eigval.real # Keep the real part
	eigvec = eigvec.real # Keep the real part
	idx = eigval.argsort() # Get indices of sorted eigenvalues
	eigvec = eigvec[:,idx] # Sort eigenvectors according to eigenvalues
	Y = eigvec[:,:k] # Keep the first k vectors
 	labels = kmeans(Y, k)

	return labels
