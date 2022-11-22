import os, sys
import numpy as np
from sklearn import datasets

#import libraries as needed

def readDataLabels(): 
	# read in the data and the labels to feed into the ANN
	X, y = datasets.make_moons(n_samples=300, noise=0.08, shuffle=False)

	return X, y
