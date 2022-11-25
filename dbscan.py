import os, sys
import numpy as np
import math

from data import readDataLabels
from utils import plot

from sklearn.cluster import DBSCAN as sklearnDBSCAN     # Only jerks will use this in assignment!


def test_sklearn(X):
    # Lets see how sklearn performs.
    db = sklearnDBSCAN(eps=0.2, min_samples=10).fit(X)
    plot(X, db.labels_)


class DBSCAN():

    """

        eps  - The radius within which samples are considered neighbors.
        min_samples - The minimum number of samples required by a point to
                        be considered core point. 
    """
    def __init__(self, eps=1.0, min_samples=5):
        self.eps = eps
        self.min_samples = min_samples


    def get_neighbors(self, sample_i):
        """ Returns a list of indices of neighoring samples
            A sample_a is considered a neighebor of sample_b if the distance 
            between them less than eps """
        neighbors = []
        idxs = np.arange(len(self.X))
        for i, _sample in enumerate(self.X):
            if i != sample_i:
                dist = euclidean_distance(self.X[sample_i],_sample)
                if dist < self.eps:
                    neighbors.append(i)
        return np.asarray(neighbors)


    def expand_cluster(self, sample_i, neighbors): #TODO
        """ Recursively expand cluster until the border of the dense area is reached
            Dense area is determined by eps and min_samples """
        if type(sample_i) is not list:
            cluster = [sample_i]
        else:
            cluster = sample_i
        # Iterate through neighbors
        for neighbor_i in neighbors:
            if not (neighbor_i in self.visited_samples):
                self.visited_samples.append(neighbor_i)

                # Fetch the sample's distant neighbors (neighbors of neighbor)
                dist_neighbors = self.get_neighbors(neighbor_i)
                
                # Make sure the neighbor's neighbors are more than min_samples
                if len(dist_neighbors) >= self.min_samples:
                    # (If this is true the neighbor is a core point) -> call this function to expand the cluster.
                    cluster = self.expand_cluster(cluster + [neighbor_i],dist_neighbors)
                else:
                    # Else, the neighbor is not a core point so just add it to the cluster
                    cluster.append(neighbor_i)

        return cluster


    def det_cluster_labels(self): #TODO
        """" Return the sample labels as the index of the cluster in which they are contained """
        # Set default value to number of clusters
        # This will make sure all outliers have same cluster label
        labels = np.full(shape=self.X.shape[0], fill_value=len(self.clusters))
        for i,cluster in enumerate(self.clusters):
            for sample in cluster:
                labels[sample] = i

        return labels


    def predict(self, X):
        self.X = X 
        self.clusters = []
        self.visited_samples = []
        self.neighbors = {}

        n_samples = np.shape(self.X)[0]

        for sample in range(n_samples):
            # If a saample is visited, then do not bother with it.
            if sample in self.visited_samples:
                continue

            # Iterate through un-visited samples and expand clusters from them
            # if they have more neighbors than self.min_samples
            self.neighbors[sample] = self.get_neighbors(sample)
            if len(self.neighbors[sample]) >= self.min_samples:
                # Core point is a sample that has more than min_samples neighbors. Its the beginning of a new cluster!
                # If core point => mark as visited
                self.visited_samples.append(sample)
                # Sample has more neighbors than self.min_samples => expand
                # cluster from sample
                new_cluster = self.expand_cluster(sample, self.neighbors[sample])
                # Add cluster to list of clusters
                self.clusters.append(new_cluster)
            pass


        cluster_labels = self.det_cluster_labels()
        return cluster_labels


def main():

    dbscan = DBSCAN() #add custom parameters for eps and min_samples

    X,y = readDataLabels()
    print(X[0])
    plot(X, y)

    # Run prediction over the data
    # Cluster the data using DBSCAN
    clf = DBSCAN(eps=0.2, min_samples=5)   # You can experiment with different parameters
    y_pred = clf.predict(X)

    # Plot and compare to ground truth
    plot(X,y_pred)

    # For reference... This is how sklearn performs!
    test_sklearn(X)


if __name__ == "__main__":
    main()