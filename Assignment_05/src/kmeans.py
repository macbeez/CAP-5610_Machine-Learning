import numpy as np
from scipy.spatial import distance
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.metrics.pairwise import cosine_distances

# np.random.seed(100)

def euclidean_distance(x1, x2):
    return distance.euclidean(x1, x2)

def manhattan_distance(x1, x2):
    return distance.cityblock(x1, x2)

def cosine_distance(x1, x2):
    return distance.cosine(x1, x2)

def generalized_jaccard(x1, x2):
    min_val = 0
    max_val = 0
    for i, j in zip(x1, x2):
        min_val += min(i, j)
        max_val += max(i, j)
    return 1-(min_val/max_val)

def SSE(y_test, y_pred):
    y_pred = y_pred
    sum = 0
    for i in range(len(y_test)):
        sum += (y_test[i] - y_pred[i])**2
    return sum

class KMeansTask1:
    def __init__(self, K = 2, max_iters = 100, centroids = None, dist_method="euclidean", plot_steps = False):
        self.K = K
        self.max_iters = max_iters
        self.plot_steps = plot_steps
        self.centroids = centroids
        self.dist_method = dist_method
        print("Initial Centroids: ", self.centroids)
        # list of sample indices for each cluster
        self.clusters = [[] for _ in range(self.K)] # initialize an empty list for clusters
        # Empty list for each mean feature vectors for each cluster
        # self.centroids = [] 

    def predict(self, X):
        self.X = X
        if (len(X.shape) > 1):
            self.n_samples, self.n_features = X.shape
        else:
            self.n_samples, self.n_features = X.shape[0], 1

        # initialize random centroids by picking K centers from n_samples when centroids are not given
        if self.centroids == None:
            random_sample_idxs = np.random.choice(self.n_samples, self.K, replace = False)
            self.centroids = [self.X[idx] for idx in random_sample_idxs] 

        # Optimization
        for i in range(self.max_iters):
            # update clusters
            self.clusters = self._create_clusters(self.centroids)
            if self.plot_steps:
                self.plot(title = "Wins in 2016 and 2017 (Iter: {0}, Current centroids)".format(i), x = "Wins in 2016", y = "Wins in 2017")
            # update centroids
            centroids_old = self.centroids
            self.centroids = self._get_centroids(self.clusters)
            if self.plot_steps:
                self.plot(title = "Wins in 2016 and 2017 (Iter: {0}, Updated Centroids)".format(i+1), x = "Wins in 2016", y = "Wins in 2017")
            # check if converged
            if self._is_converged(centroids_old, self.centroids):
                print("The final converging centroids are: ", self.centroids)
                break

        # return cluster labels
        return self._get_cluster_labels(self.clusters)

    def _get_cluster_labels(self, clusters):
        # These are not the actual labels of the cluster points
        # This is just the index of the cluster it was assigned to
        labels = np.empty(self.n_samples)
        for cluster_ids, cluster in enumerate(clusters):
            for sample_idx in cluster:
                labels[sample_idx] = cluster_ids
        return labels

    def _create_clusters(self, centroids):
        clusters = [[] for _ in range(self.K)]
        for idx, sample in enumerate(self.X):
            centroid_idx = self._closest_centroid(sample, centroids)
            # print("Clusters:", clusters)
            clusters[centroid_idx].append(idx)
        return clusters

    def _closest_centroid(self, sample, centroids):
        distances = [self.get_distance(sample, point) for point in centroids]
        closest_idx = np.argmin(distances)
        return closest_idx

    def _get_centroids(self, clusters):
        centroids = np.zeros((self.K, self.n_features)) # intialize centroids to zeros
        for cluster_idx, cluster in enumerate(clusters):
            cluster_mean = np.mean(self.X[cluster], axis = 0) # calculate the mean of the points in the current cluster
            centroids [cluster_idx] = cluster_mean
        return centroids # returning the new centroids

    def get_distance(self, x1, x2):
        if self.dist_method == "euclidean":
            return euclidean_distance(x1, x2)

        elif self.dist_method == "manhattan":
            return manhattan_distance(x1, x2)

    def _is_converged(self, centroids_old, centroids):
        distance = [self.get_distance(centroids_old[i], centroids[i]) for i in range(self.K)]
        return sum(distance) == 0

    def plot(self, title, x, y):
        fig, ax = plt.subplots(figsize = (8,6))
        ax.set_title(title)
        ax.set_xlabel(x)
        ax.set_ylabel(y)

        for i, index in enumerate(self.clusters):
            point = self.X[index].T
            ax.scatter(*point)

        for point in self.centroids:
            ax.scatter(*point, marker = "x", color = "black", linewidth = 2)
            text = "(%.2f, %.2f)" % (point[0], point[1])
            plt.text(point[0], point[1], text)

        plt.show()

class KMeansTask2:
    def __init__(self, K = 3, max_iters = 100, centroids = None, ytest=None, dist_method="euclidean", plot_steps = False, termination="centroid"):
        self.K = K
        self.max_iters = max_iters
        self.plot_steps = plot_steps
        self.old_centroids = []
        self.centroids = centroids
        self.old_sse = 0
        self.sse = 0
        self.dist_method = dist_method
        self.total_iterations = 0
        self.termination_method = termination
        # list of sample indices for each cluster
        self.clusters = [[] for _ in range(self.K)] # initialize an empty list for clusters

        self.ytest = ytest

    def predict(self, X):
        self.X = X
        if (len(X.shape) > 1):
            self.n_samples, self.n_features = X.shape
        else:
            self.n_samples, self.n_features = X.shape[0], 1

        if self.centroids == None:
            # initialize the centroids, randomly pick K centers from n_samples
            random_sample_idxs = np.random.choice(self.n_samples, self.K, replace = False)
            self.centroids = [self.X[idx] for idx in random_sample_idxs] # change this to assign your own centers

        # Optimization
        for i in range(self.max_iters):
            # update clusters
            self.old_sse = SSE(self.ytest, self._get_cluster_labels(self.clusters))
            self.clusters = self._create_clusters(self.centroids)
            # update centroids
            self.old_centroids = self.centroids
            self.centroids = self._get_centroids(self.clusters)
            # check if converged
            if self._is_converged():
                if self.termination_method == "centroids":
                    self.total_iterations = i
                    break
                if self.termination_method == "sse":
                    self.total_iterations = i
                    self.clusters = self._create_clusters(self.old_centroids)
                    break

        # return cluster labels
        return self._get_cluster_labels(self.clusters)

    def _get_cluster_labels(self, clusters):
        # These are not the actual labels of the cluster points
        # This is just the index of the cluster it was assigned to
        labels = np.empty(self.n_samples)
        for cluster_ids, cluster in enumerate(clusters):
            for sample_idx in cluster:
                labels[sample_idx] = cluster_ids
        return labels

    def _create_clusters(self, centroids):
        clusters = [[] for _ in range(self.K)]
        for idx, sample in enumerate(self.X):
            centroid_idx = self._closest_centroid(sample, centroids)
            clusters[centroid_idx].append(idx)
        return clusters

    def _closest_centroid(self, sample, centroids):
        distances = [self.get_distance(sample, point) for point in centroids]
        closest_idx = np.argmin(distances)
        return closest_idx

    def _get_centroids(self, clusters):
        centroids = np.zeros((self.K, self.n_features)) # intialize centroids to zeros
        for cluster_idx, cluster in enumerate(clusters):
            cluster_mean = np.mean(self.X[cluster], axis = 0) # calculate the mean of the points in the current cluster
            centroids [cluster_idx] = cluster_mean
        return centroids # returning the new centroids

    def get_distance(self, x1, x2):
        if self.dist_method == "euclidean":
            return euclidean_distance(x1, x2)

        elif self.dist_method == "manhattan":
            return manhattan_distance(x1, x2)

        elif self.dist_method == "cosine":
            return cosine_distance(x1,x2)

        elif self.dist_method == "jaccard":
            return generalized_jaccard(x1, x2)

    def _is_converged(self):
        if self.termination_method == "centroids":
            distance = [self.get_distance(self.old_centroids[i], self.centroids[i]) for i in range(self.K)]
            return sum(distance) == 0
        if self.termination_method == "sse":
            self.sse = SSE(self.ytest, self._get_cluster_labels(self.clusters))
            return self.sse >= self.old_sse








