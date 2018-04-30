from numpy import *
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import pylab
from numpy import genfromtxt

# Load the data set (wine). Variable data stores the final data (178 x 13)
my_data = [[-1, -1], [0, 0], [1, 1]]
data = my_data

M = mean(data, 0)
C = data - M
print(C)
W = dot(C.T, C)
print(W)
eigval, eigvec = linalg.eig(W) #compute eigenvalues and eigenvectors of covariance matrix
idx = eigval.argsort()[::-1] #sorting of eigenvalues
eigvec = eigvec[:,idx] #sort eigenvetors according to eigenvalues

newData2 = dot(C,real(eigvec[:,:1])) # Project the data to the new space (2-D)

print(eigval)
print(eigvec)
print(newData2)
