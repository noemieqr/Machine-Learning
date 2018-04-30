from numpy import *
from math import e
from math import pow

def sigmoid(z):
	# Computes the sigmoid of z.

    g = 1.0 / (1 + pow(e,-z))
             
    return g
