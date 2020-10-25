###########################################################################################################
### TASK 1
###########################################################################################################

import numpy as np
from sklearn.datasets import make_blobs
import matplotlib.pyplot as plt
from kmeans import KMeansTask1

# create random cluster points 
X, y = make_blobs(centers = 2, n_samples=500, n_features = 2, shuffle = True, random_state=68)
# change K to clusters when centroids are initialized to "None" and cluster points are generated using makeblobs library
clusters = len(np.unique(y)) 
centroids = None

# Using given points
X, y = np.array([[3, 5], [3, 4], [2, 8], [2, 3], [6, 2], [6, 4], [7, 3], [7, 4], [8, 5], [7, 6]]), None
# print("X, y: ", X, y)
# print(X.shape)

centroids = (np.array([3, 2]), np.array([4, 8]))

k1 = KMeansTask1(K = len(centroids), 
         max_iters = 50, 
         centroids = centroids, 
         dist_method="manhattan", 
         plot_steps = True) # set plot_steps = False to see the final graph without iterations
y_pred = k1.predict(X)

# k1.plot(title = "Wins in 2016 and 2017 (Final clusters)", x = "Wins in 2016", y = "Wins in 2017")

##########################################################################################################
### TASK 2
##########################################################################################################

from sklearn import datasets
from kmeans import *
from sklearn import metrics
from sklearn.preprocessing import scale
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from statistics import mode
import time
from scipy.spatial import distance
import warnings
from matplotlib import rcParams
from tabulate import tabulate

rcParams.update({'figure.autolayout': True})

# Filter RuntimeWarnings
warnings.filterwarnings("ignore", "Mean of empty slice")
warnings.filterwarnings("ignore", "invalid value encountered in true_divide")
warnings.filterwarnings("ignore", "overflow encountered in double_scalars")
warnings.filterwarnings("ignore:", "invalid value encountered in double_scalars")

iris = datasets.load_iris()
X = iris.data
y = pd.DataFrame(iris.target)
y_i = iris.target
# print("Actual y values: \n\n", y_i)
clusters = len(np.unique(y))

# Set fixed centroids 
# centroids1 = (np.array([5, 3.4, 1.46, 0.25]), np.array([6.5, 2.96, 5.5, 1.99]), np.array([5.9, 2.8, 4.2, 1.3]))

# Set random centroids by equating centroids to None.
# centroids = None

# Set centroids to have maximum distance between each other
def farthest_distance_centroids(X_input, distance_method):
    X = X_input
    random_centroid_idx = np.random.choice(X.shape[0], 1)
    random_centroid = X[random_centroid_idx][0]
    farthest_dist = 0
    max_dist_centroids = []
    for i in range(0, len(X)):
        dist = distance_method(random_centroid, X[i])
        if dist > farthest_dist:
            farthest_dist = dist
            temp1 = X[i]
    max_dist_centroids.append(random_centroid) # centroid C1
    max_dist_centroids.append(temp1) # centroid C2

    # pick the third centroid to have the longest average distance between C1 and C2:
    longest_average = 0
    for i in range(0,len(X)):
        dist1 = distance_method(max_dist_centroids[0], X[i])
        dist2 = distance_method(max_dist_centroids[1], X[i])
        average_distance = (dist1 + dist2)/2
        if longest_average < average_distance:
            longest_average = average_distance
            temp2 = X[i]

    max_dist_centroids.append(temp2) # centroid C3
    centroids = max_dist_centroids
    return centroids

def SSE(y_real, y_predicted):
    y_real = y_real
    y_predicted = y_predicted
    sum = 0
    for i in range(len(y_i)):
        sum += (y_real[i] - y_predicted[i])**2
    return sum

def get_cluster_order(centroids):
    # Get 1st column 
    l = [i[0] for i in k2.centroids]
    # Get the indexes of the sorted list
    result = sorted(range(len(l)), key=lambda k: l[k])
    return result

def reassign_values(input_list, new_order):
    new_list = []
    for value in input_list:
        new_list.append(new_order.index(value))
    new_list = np.array(new_list, dtype=int)
    return new_list

#################################################################################################################
### EUCLIDEAN DISTANCE

start_time = time.time()
centroids = farthest_distance_centroids(X, euclidean_distance)
# print("Centroids ED: ", centroids)
k2 = KMeansTask2(K = clusters,
            max_iters = 100,
            centroids = centroids,
            dist_method = "euclidean",
            plot_steps = False,
            ytest = y_i,
            termination="centroids")

y_pred = np.array(k2.predict(X), dtype='int')

# Get the cluster order of the predicted results
cluster_order = get_cluster_order(k2.centroids)
euclidean_ypred = reassign_values(y_pred, cluster_order)

print("Euclidean --- %s seconds ---" % (time.time() - start_time))
print("Number of iterations using Euclidean distance: ", k2.total_iterations)
euclidean_accuracy = metrics.accuracy_score(y_i, euclidean_ypred) * 100
print("\nEuclidean accuracy: %.2f" % euclidean_accuracy, "%")
print("\nEuclidean SSE: ", SSE(y_i, euclidean_ypred))
print("Y_pred values with Euclidean distance: \n\n", euclidean_ypred)

## The the average over 100 iterations:
avg_SSE = []
avg_acc = []
avg_iter = []
avg_time = []
for a in range(0,100):
    start_time = time.time()
    centroids = farthest_distance_centroids(X, euclidean_distance)
    k2 = KMeansTask2(K = clusters,
            max_iters = 500,
            centroids = centroids,
            dist_method = "euclidean",
            plot_steps = False,
            ytest = y_i,
            termination="sse")
    y_pred = np.array(k2.predict(X), dtype='int')
    cluster_order = get_cluster_order(k2.centroids)
    euclidean_ypred = reassign_values(y_pred, cluster_order)
    avg_time.append((time.time() - start_time))
    avg_SSE.append(SSE(y_i, euclidean_ypred))
    acc = metrics.accuracy_score(y_i, euclidean_ypred) 
    avg_acc.append(acc)
    avg_iter.append(k2.total_iterations)

metrics_dict = {"time": avg_time, "SSE": avg_SSE, "accuracy":avg_acc, "iterations":avg_iter}

### Plot Histograms for Metrics

plt.tight_layout()
fig_hist, ax_hist = plt.subplots(2, 2, figsize=(7,7))

fig_hist.suptitle('KMeans - Euclidean Metrics')

metrics_dict["SSE"] = np.array(metrics_dict["SSE"])
metrics_dict["accuracy"] = np.array(metrics_dict["accuracy"])
metrics_dict["iterations"] = np.array(metrics_dict["iterations"])
metrics_dict["time"] = np.array(metrics_dict["time"])

ax_hist[0,0].hist(metrics_dict["SSE"], bins = 10)
ax_hist[0,0].set_xlabel("SSE")
ax_hist[0,0].set_ylabel("count")
sse_mean = metrics_dict["SSE"].mean()
min_ylim, max_ylim = ax_hist[0,0].get_ylim()
ax_hist[0,0].axvline(sse_mean, color='k', linestyle='dashed', linewidth=1)


ax_hist[0,1].hist(metrics_dict["accuracy"], bins = 10)
ax_hist[0,1].set_xlabel("Accuracy (%)")
ax_hist[0,1].set_ylabel("count")
acc_mean = metrics_dict["accuracy"].mean()
min_ylim, max_ylim = ax_hist[0,1].get_ylim()
ax_hist[0,1].axvline(acc_mean, color='k', linestyle='dashed', linewidth=1)


ax_hist[1,0].hist(metrics_dict["iterations"], bins = 10)
ax_hist[1,0].set_xlabel("Iterations")
ax_hist[1,0].set_ylabel("count")
iter_mean = metrics_dict["iterations"].mean()
min_ylim, max_ylim = ax_hist[1,0].get_ylim()
ax_hist[1,0].axvline(iter_mean, color='k', linestyle='dashed', linewidth=1)


ax_hist[1,1].hist(metrics_dict["time"], bins = 10)
ax_hist[1,1].set_xlabel("Completion Time (s)")
ax_hist[1,1].set_ylabel("count")
time_mean = metrics_dict["time"].mean()
min_ylim, max_ylim = ax_hist[1,1].get_ylim()
ax_hist[1,1].axvline(time_mean, color='k', linestyle='dashed', linewidth=1)

fig_hist.savefig('kmeans_cosine_metrics.png')

print("Euclidean Mean Values over 100 iterations:")
print(tabulate([[sse_mean, acc_mean, iter_mean, time_mean]], headers=["SSE", "Accuracy", "Iterations", "Time"], tablefmt='psql'))

plt.show()

##########################################################################################################################
### COSINE DISTANCE

start_time = time.time()
centroids = farthest_distance_centroids(X, cosine_distance)
# print("Centroids CD: ", centroids)
k2 = KMeansTask2(K = clusters,
            max_iters = 100,
            centroids = centroids,
            dist_method = "cosine",
            plot_steps = False,
            ytest = y_i,
            termination="centroids")


y_pred = np.array(k2.predict(X), dtype='int')
cluster_order = get_cluster_order(k2.centroids)
cosine_ypred = reassign_values(y_pred, cluster_order)

print("\nCosine --- %s seconds ---" % (time.time() - start_time))
print("Number of iterations using Cosine distance: ", k2.total_iterations)
cosine_accuracy = metrics.accuracy_score(y_i, cosine_ypred) * 100
print("\nCosine accuracy: %.2f" % cosine_accuracy, "%")
print("\nCosine SSE: ", SSE(y_i, cosine_ypred))
print("Y_pred values with Cosine distance: \n\n", cosine_ypred)

### The the average over 100 iterations:

avg_SSE = []
avg_acc = []
avg_iter = []
avg_time = []
for a in range(0,100):
    start_time = time.time()
    centroids = farthest_distance_centroids(X, cosine_distance)
    k2 = KMeansTask2(K = clusters,
            max_iters = 100,
            centroids = centroids,
            dist_method = "cosine",
            plot_steps = False,
            ytest = y_i,
            termination = "centroid")
    y_pred = np.array(k2.predict(X), dtype='int')
    cluster_order = get_cluster_order(k2.centroids)
    euclidean_ypred = reassign_values(y_pred, cluster_order)
    avg_time.append((time.time() - start_time))
    avg_SSE.append(SSE(y_i, cosine_ypred))
    acc = metrics.accuracy_score(y_i, cosine_ypred) 
    avg_acc.append(acc)
    # avg_iter.append(k2.total_iterations)

metrics_dict = {"time": avg_time, "SSE": avg_SSE, "accuracy":avg_acc, "iterations":avg_iter}

### Plot Histograms for Metrics

plt.tight_layout()
fig_hist, ax_hist = plt.subplots(2, 2, figsize=(7,7))

fig_hist.suptitle('KMeans - Cosine Metrics')

metrics_dict["SSE"] = np.array(metrics_dict["SSE"])
metrics_dict["accuracy"] = np.array(metrics_dict["accuracy"])
metrics_dict["iterations"] = np.array(metrics_dict["iterations"])
metrics_dict["time"] = np.array(metrics_dict["time"])

ax_hist[0,0].hist(metrics_dict["SSE"], bins = 10)
ax_hist[0,0].set_xlabel("SSE")
ax_hist[0,0].set_ylabel("count")
sse_mean = metrics_dict["SSE"].mean()
min_ylim, max_ylim = ax_hist[0,0].get_ylim()
ax_hist[0,0].axvline(sse_mean, color='k', linestyle='dashed', linewidth=1)


ax_hist[0,1].hist(metrics_dict["accuracy"], bins = 10)
ax_hist[0,1].set_xlabel("Accuracy (%)")
ax_hist[0,1].set_ylabel("count")
acc_mean = metrics_dict["accuracy"].mean()
min_ylim, max_ylim = ax_hist[0,1].get_ylim()
ax_hist[0,1].axvline(acc_mean, color='k', linestyle='dashed', linewidth=1)


ax_hist[1,0].hist(metrics_dict["iterations"], bins = 10)
ax_hist[1,0].set_xlabel("Iterations")
ax_hist[1,0].set_ylabel("count")
iter_mean = metrics_dict["iterations"].mean()
min_ylim, max_ylim = ax_hist[1,0].get_ylim()
ax_hist[1,0].axvline(iter_mean, color='k', linestyle='dashed', linewidth=1)


ax_hist[1,1].hist(metrics_dict["time"], bins = 10)
ax_hist[1,1].set_xlabel("Completion Time (s)")
ax_hist[1,1].set_ylabel("count")
time_mean = metrics_dict["time"].mean()
min_ylim, max_ylim = ax_hist[1,1].get_ylim()
ax_hist[1,1].axvline(time_mean, color='k', linestyle='dashed', linewidth=1)

fig_hist.savefig('kmeans_cosine_metrics.png')

print("Cosine Mean Values over 100 iterations:")
print(tabulate([[sse_mean, acc_mean, iter_mean, time_mean]], headers=["SSE", "Accuracy", "Iterations", "Time"], tablefmt='psql'))

plt.show()

######################################################################################################################
### GENERALIZED JACCARD

start_time = time.time()
centroids = farthest_distance_centroids(X, generalized_jaccard)
# print("Centroids GJ: ", centroids)
k2 = KMeansTask2(K = clusters,
            max_iters = 100,
            centroids = centroids,
            dist_method = "jaccard",
            plot_steps = False,
            ytest = y_i,
            termination="centroids")

y_pred = np.array(k2.predict(X), dtype='int')
cluster_order = get_cluster_order(k2.centroids)
jaccard_ypred = reassign_values(y_pred, cluster_order)

print("\nJaccard --- %s seconds ---" % (time.time() - start_time))
print("Number of iterations using Jaccard distance: ", k2.total_iterations)
jaccard_accuracy = metrics.accuracy_score(y_i, jaccard_ypred) * 100
print("\nJaccard accuracy: %.2f" % jaccard_accuracy, "%")
print("\nJaccard SSE: ", SSE(y_i,jaccard_ypred))
print("Y_pred values with jaccard similarity: \n\n", jaccard_ypred)

### The the average over 100 iterations:

avg_SSE = []
avg_acc = []
avg_iter = []
avg_time = []
for a in range(0,100):
    start_time = time.time()
    centroids = farthest_distance_centroids(X, generalized_jaccard)
    k2 = KMeansTask2(K = clusters,
            max_iters = 500,
            centroids = centroids,
            dist_method = "jaccard",
            plot_steps = False,
            ytest = y_i,
            termination="sse")
    y_pred = np.array(k2.predict(X), dtype='int')
    cluster_order = get_cluster_order(k2.centroids)
    euclidean_ypred = reassign_values(y_pred, cluster_order)
    avg_time.append((time.time() - start_time))
    avg_SSE.append(SSE(y_i, jaccard_ypred))
    acc = metrics.accuracy_score(y_i, jaccard_ypred) 
    avg_acc.append(acc)
    avg_iter.append(k2.total_iterations)

metrics_dict = {"time": avg_time, "SSE": avg_SSE, "accuracy":avg_acc, "iterations":avg_iter}

### Plot Histograms for Metrics

plt.tight_layout()
fig_hist, ax_hist = plt.subplots(2, 2, figsize=(7,7))

fig_hist.suptitle('KMeans - Jaccard Metrics')

metrics_dict["SSE"] = np.array(metrics_dict["SSE"])
metrics_dict["accuracy"] = np.array(metrics_dict["accuracy"])
metrics_dict["iterations"] = np.array(metrics_dict["iterations"])
metrics_dict["time"] = np.array(metrics_dict["time"])

ax_hist[0,0].hist(metrics_dict["SSE"], bins = 10)
ax_hist[0,0].set_xlabel("SSE")
ax_hist[0,0].set_ylabel("count")
sse_mean = metrics_dict["SSE"].mean()
min_ylim, max_ylim = ax_hist[0,0].get_ylim()
ax_hist[0,0].axvline(sse_mean, color='k', linestyle='dashed', linewidth=1)


ax_hist[0,1].hist(metrics_dict["accuracy"], bins = 10)
ax_hist[0,1].set_xlabel("Accuracy (%)")
ax_hist[0,1].set_ylabel("count")
acc_mean = metrics_dict["accuracy"].mean()
min_ylim, max_ylim = ax_hist[0,1].get_ylim()
ax_hist[0,1].axvline(acc_mean, color='k', linestyle='dashed', linewidth=1)


ax_hist[1,0].hist(metrics_dict["iterations"], bins = 10)
ax_hist[1,0].set_xlabel("Iterations")
ax_hist[1,0].set_ylabel("count")
iter_mean = metrics_dict["iterations"].mean()
min_ylim, max_ylim = ax_hist[1,0].get_ylim()
ax_hist[1,0].axvline(iter_mean, color='k', linestyle='dashed', linewidth=1)


ax_hist[1,1].hist(metrics_dict["time"], bins = 10)
ax_hist[1,1].set_xlabel("Completion Time (s)")
ax_hist[1,1].set_ylabel("count")
time_mean = metrics_dict["time"].mean()
min_ylim, max_ylim = ax_hist[1,1].get_ylim()
ax_hist[1,1].axvline(time_mean, color='k', linestyle='dashed', linewidth=1)

fig_hist.savefig('kmeans_cosine_metrics.png')

print("Jaccard Mean Values over 100 iterations:")
print(tabulate([[sse_mean, acc_mean, iter_mean, time_mean]], headers=["SSE", "Accuracy", "Iterations", "Time"], tablefmt='psql'))

plt.show()

############################################################################################################################
### Plot 2D graphs by taking 2 features at a time.

plot_variable = "jaccard"

if plot_variable == "jaccard":
    relabeled_ypred = jaccard_ypred
elif plot_variable == "cosine":
    relabeled_ypred = cosine_ypred
elif plot_variable == "euclidean":
    relabeled_ypred = euclidean_ypred

iris_df = pd.DataFrame(iris.data)
iris_df.columns = ['Sepal_Length', 'Sepal_Width', 'Petal_Length', 'Petal_Width']
y.columns = ['Target']
color_theme = np.array(['red', 'green', 'blue'])

fig_jaccard, axes_jaccard = plt.subplots(2,2)
fig_jaccard.suptitle('KMeans (using Jaccard Distance)')

axes_jaccard[0,0].scatter(x = iris_df.Petal_Length, y = iris_df.Petal_Width, c = color_theme[iris.target], s = 50)
axes_jaccard[0,0].set_title('Y_test points')
axes_jaccard[0,0].set_xlabel("Petal Length")
axes_jaccard[0,0].set_ylabel("Petal Width")

axes_jaccard[0,1].scatter(x = iris_df.Petal_Length, y = iris_df.Petal_Width, c = color_theme[jaccard_ypred], s = 50)
axes_jaccard[0,1].set_title("K-Means Clustering")
axes_jaccard[0,1].set_xlabel("Petal Length")
axes_jaccard[0,1].set_ylabel("Petal Width")

axes_jaccard[1,0].scatter(x = iris_df.Sepal_Length, y = iris_df.Sepal_Width, c = color_theme[iris.target], s = 50)
axes_jaccard[1,0].set_title('Y_test points')
axes_jaccard[1,0].set_xlabel("Sepal_Length")
axes_jaccard[1,0].set_ylabel("Sepal_Width")

axes_jaccard[1,1].scatter(x = iris_df.Sepal_Length, y = iris_df.Sepal_Width, c = color_theme[jaccard_ypred], s = 50)
axes_jaccard[1,1].set_title("K-Means Clustering")
axes_jaccard[1,1].set_xlabel("Sepal_Length")
axes_jaccard[1,1].set_ylabel("Sepal_Width")
plt.tight_layout()

fig_cosine, axes_cosine = plt.subplots(2,2)
fig_cosine.suptitle('KMeans (using Cosine Distance)')

axes_cosine[0,0].scatter(x = iris_df.Petal_Length, y = iris_df.Petal_Width, c = color_theme[iris.target], s = 50)
axes_cosine[0,0].set_title('Y_test points')
axes_cosine[0,0].set_xlabel("Petal Length")
axes_cosine[0,0].set_ylabel("Petal Width")

axes_cosine[0,1].scatter(x = iris_df.Petal_Length, y = iris_df.Petal_Width, c = color_theme[cosine_ypred], s = 50)
axes_cosine[0,1].set_title("K-Means Clustering")
axes_cosine[0,1].set_xlabel("Petal Length")
axes_cosine[0,1].set_ylabel("Petal Width")

axes_cosine[1,0].scatter(x = iris_df.Sepal_Length, y = iris_df.Sepal_Width, c = color_theme[iris.target], s = 50)
axes_cosine[1,0].set_title('Y_test points')
axes_cosine[1,0].set_xlabel("Sepal_Length")
axes_cosine[1,0].set_ylabel("Sepal_Width")

axes_cosine[1,1].scatter(x = iris_df.Sepal_Length, y = iris_df.Sepal_Width, c = color_theme[cosine_ypred], s = 50)
axes_cosine[1,1].set_title("K-Means Clustering")
axes_cosine[1,1].set_xlabel("Sepal_Length")
axes_cosine[1,1].set_ylabel("Sepal_Width")
plt.tight_layout()

fig_euclidean, axes_euclidean = plt.subplots(2,2)
fig_euclidean.suptitle('KMeans (using Euclidean Distance)')

axes_euclidean[0,0].scatter(x = iris_df.Petal_Length, y = iris_df.Petal_Width, c = color_theme[iris.target], s = 50)
axes_euclidean[0,0].set_title('Y_test points')
axes_euclidean[0,0].set_xlabel("Petal Length")
axes_euclidean[0,0].set_ylabel("Petal Width")

axes_euclidean[0,1].scatter(x = iris_df.Petal_Length, y = iris_df.Petal_Width, c = color_theme[euclidean_ypred], s = 50)
axes_euclidean[0,1].set_title("K-Means Clustering")
axes_euclidean[0,1].set_xlabel("Petal Length")
axes_euclidean[0,1].set_ylabel("Petal Width")

axes_euclidean[1,0].scatter(x = iris_df.Sepal_Length, y = iris_df.Sepal_Width, c = color_theme[iris.target], s = 50)
axes_euclidean[1,0].set_title('Y_test points')
axes_euclidean[1,0].set_xlabel("Sepal_Length")
axes_euclidean[1,0].set_ylabel("Sepal_Width")

axes_euclidean[1,1].scatter(x = iris_df.Sepal_Length, y = iris_df.Sepal_Width, c = color_theme[euclidean_ypred], s = 50)
axes_euclidean[1,1].set_title("K-Means Clustering")
axes_euclidean[1,1].set_xlabel("Sepal_Length")
axes_euclidean[1,1].set_ylabel("Sepal_Width")

plt.tight_layout()
plt.show()

###############################################################################################################
### TASK 4
###############################################################################################################

from scipy.spatial import distance
import numpy as np
import math

X = [[4.7, 3.2], [4.9, 3.1], [5, 3], [4.6, 2.9]] # cluster 1
Y = [[5.9, 3.2], [6.7, 3.1], [6, 3], [6.2, 2.8]] # cluster 2

farthest_distance = 0
farthest_points = []
for i in range(0, len(X)):
    for j in range(0, len(X)):
        dist = distance.euclidean(X[i], Y[j])
        if dist > farthest_distance:
            farthest_distance = dist
            farthest_points = X[i], Y[j]

print("\nThe two farthest_points are: ", farthest_points)
print("The farthest distance is: %.4f" % farthest_distance)

closest_distance = math.inf
closest_points = []
for i in range(0, len(X)):
    for j in range(0, len(X)):
        dist = distance.euclidean(X[i], Y[j])
        if dist < closest_distance:
            closest_distance = dist
            closest_points = X[i], Y[j]

print("\nThe two closest points are: ", closest_points)
print("The closest distance is: %.4f" % closest_distance)

points = [[4.7, 3.2], [4.9, 3.1], [5, 3], [4.6, 2.9], [5.9, 3.2], [6.7, 3.1], [6, 3], [6.2, 2.8]]
distances = []
for i in range(0, len(points)):
    for j in range(i + 1, len(points)):
        dist = distance.euclidean(points[i], points[j])
        distances.append(dist)
average_distance = sum(distances)/len(distances)

print("\nAverage distance between all pairs is: %.4f" % average_distance)



