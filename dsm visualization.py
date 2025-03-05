#!../mypythonenv/bin/python

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# round, fruity, eatable
# Define the points
new_vectors = {
    'apple': np.array([0.9, 0.9, 1]),
    'pear': np.array([0.4, 0.9, 1]),
    'basketball': np.array([0.9, -0.7, -0.9]),
    'cake': np.array([0.5, -0.1, 0.9]),
    'brick': np.array([-.3, -.9, -1]),
    'cookie': np.array([.8, -.8, .8])
}

# Extracting the keys and values
labels = list(new_vectors.keys())
points = np.array(list(new_vectors.values()))

# Define a color map
colors = ['red', 'green', 'blue', 'orange', 'pink', 'yellow']

# 2D Plot
plt.figure(figsize=(10, 5))

# Plotting the 2D projection (first two coordinates)
plt.subplot(1, 2, 1)
for i, label in enumerate(labels):
    plt.scatter(points[i, 0], points[i, 2], color=colors[i])  # Using roundness and edibility
    plt.annotate(label, (points[i, 0], points[i, 2]), textcoords="offset points", xytext=(0, 10), ha='center', fontsize=12)

plt.xlabel('roundness', fontsize=14)  # Increased font size
plt.ylabel('edibility', fontsize=14)   # Increased font size
plt.xlim(-1.1, 1.1)  # Enlarged bounds
plt.ylim(-1.1, 1.1)  # Enlarged bounds
plt.grid()

# Add an arrow from the 2D plot to the 3D plot
plt.annotate('', xy=(1.2, 0), xytext=(1, 0), 
             arrowprops=dict(arrowstyle='->', lw=1.5, color='black'))

# 3D Plot
ax = plt.subplot(1, 2, 2, projection='3d')
for i, label in enumerate(labels):
    ax.scatter(points[i, 0], points[i, 1], points[i, 2], color=colors[i])
    ax.text(points[i, 0], points[i, 1], points[i, 2], label, fontsize=12)  # Increased font size

ax.set_xlabel('roundness', fontsize=14)  # Increased font size
ax.set_ylabel('fruitiness', fontsize=14)  # Increased font size
ax.set_zlabel('edibility', fontsize=14)   # Increased font size
ax.set_xlim(-1.1, 1.1)  # Enlarged bounds
ax.set_ylim(-1.1, 1.1)  # Enlarged bounds
ax.set_zlim(-1.1, 1.1)  # Enlarged bounds

plt.tight_layout()
plt.show()
