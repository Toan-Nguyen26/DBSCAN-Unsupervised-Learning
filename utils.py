import numpy as np
import math
import matplotlib.pyplot as plt


def plot(X, y):
    # Plot the dataset X and the corresponding labels y
    plt.scatter(X[:, 0], X[:, 1], c=y, s=50, cmap='viridis')
    plt.show()


def euclidean_distance(x1,x2):
    #TODO
    #calculates l2 distance between two vectors
    return
