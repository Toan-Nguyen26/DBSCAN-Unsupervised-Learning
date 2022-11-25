import numpy as np
import math
import matplotlib.pyplot as plt


def plot(X, y):
    # Plot the dataset X and the corresponding labels y
    plt.scatter(X[:, 0], X[:, 1], c=y, s=50, cmap='viridis')
    plt.show()


def euclidean_distance(x1,x2):
    #calculates l2 distance between two vectors
    return math.sqrt(pow(x1[0]-x2[0],2)+pow(x1[1]-x2[1],2))

def euclidean_distance_hyperplane(x1,x2):
    #calculates l2 distance between two vectors
    n = min(len(x1),len(x2))
    return math.sqrt(sum([pow(x1[i]-x2[i],2) for i in range(n)]))