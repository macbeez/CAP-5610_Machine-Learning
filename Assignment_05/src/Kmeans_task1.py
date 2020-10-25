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
