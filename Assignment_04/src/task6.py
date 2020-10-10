import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import axes3d 
import math
import numpy as np

# 3D plotting
fig = plt.figure()
ax = plt.axes(projection = "3d")

# Data to plot
x = [1, 1, 1]
y = [0, (math.sqrt(2) * -1), math.sqrt(2)]
z = [0, 1, 1]

# Labeling
ax.set_xlabel('X Axes = 1')
ax.set_ylabel('Y Axes = x * sqrt(2)')
ax.set_zlabel('Z Axes = x * x')
plt.title("3D plot of feature vectors = [1, x * sqrt(2), x * x]")

colors = ("blue", "orange", "orange")
ax.scatter(x,y,z, c = colors)

xx, yy = np.meshgrid(np.linspace(0.5, 1.5, 30), np.linspace(-2, 2, 30))
zz = xx*0 + 0.5

ax.plot_surface(xx, yy, zz)

# label the cordinate points
for X, Y, Z in zip(x, y, z):
	# text = str(X) + ', ' + str(Y) + ', ' + str(Z)
	text = "(%.3f, %.3f, %.3f)" % (X, Y, Z)
	ax.text(X, Y, Z, text)

plt.show()