###########################################################################################################
### TASK 4
##########################################################################################################

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

print("The two farthest_points are: ", farthest_points)
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
