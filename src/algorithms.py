#install matplotlib
#python -m pip install matplotlib
#python -m pip install sklearn

print(__doc__)

import time

import numpy as np
import matplotlib.pyplot as plt

from sklearn.cluster import MiniBatchKMeans, KMeans
from minisom import MiniSom 
from sklearn.metrics.pairwise import pairwise_distances_argmin
from sklearn.datasets.samples_generator import make_blobs

# Generate random data
np.random.seed(0)

batch_size = 45
centers = [[1, 1], [-1, -1], [1, -1]]
n_clusters = len(centers)
X, labels_true = make_blobs(n_samples=3000, centers=centers, cluster_std=0.7)

#K-Means
k_means = KMeans(init='k-means++', n_clusters=3, n_init=10)
t0 = time.time()
k_means.fit(X)
t_batch = time.time() - t0
#SOM
som = MiniSom(1, 3, 2, sigma=0.3, learning_rate=0.5) # initialization of 1x3 SOM
t0 = time.time()
som.train_random(X, 100) # trains the SOM with 100 iterations
t_batch2 = time.time() - t0


#PLOTS
fig = plt.figure(figsize=(8, 3))
fig.subplots_adjust(left=0.02, right=0.98, bottom=0.05, top=0.9)
colors = ['#4EACC5', '#FF9C34', '#4E9A06']


k_means_cluster_centers = np.sort(k_means.cluster_centers_, axis=0)
k_means_labels = pairwise_distances_argmin(X, k_means_cluster_centers)
som_labels_array = [None] * len(X)
for index in range(len(X)):
    som_labels_array[index] = som.winner(X[index])[1]
order = [None] * len(k_means_cluster_centers)
for index in range(len(k_means_cluster_centers)):
    order[som.winner(k_means_cluster_centers[index])[1]] = index
print(order)
som_labels = np.array(som_labels_array)

# KMeans
ax = fig.add_subplot(1, 3, 1)
print('k-means')
for k, col in zip(range(n_clusters), colors):
    my_members = k_means_labels == k
    cluster_center = k_means_cluster_centers[k]
    ax.plot(X[my_members, 0], X[my_members, 1], 'w',
            markerfacecolor=col, marker='.')
    ax.plot(cluster_center[0], cluster_center[1], 'o', markerfacecolor=col,
            markeredgecolor='k', markersize=6)
ax.set_title('KMeans')
ax.set_xticks(())
ax.set_yticks(())
plt.text(-3.5, 1.8,  'train time: %.2fs' % (
    t_batch))

print('SOM')
ax = fig.add_subplot(1, 3, 2)
for k, col in zip(range(n_clusters), colors):
    cluster_center = k_means_cluster_centers[k]
    my_members = som_labels == som.winner(cluster_center)[1]
    ax.plot(X[my_members, 0], X[my_members, 1], 'w',
            markerfacecolor=col, marker='.')
    ax.plot(cluster_center[0], cluster_center[1], 'o', markerfacecolor=col,
            markeredgecolor='k', markersize=6)
ax.set_title('SOM')
ax.set_xticks(())
ax.set_yticks(())
plt.text(-3.5, 1.8, 'train time: %.2fs' %
         (t_batch2))

# Initialise the different array to all False
different = (som_labels == 4)
ax = fig.add_subplot(1, 3, 3)

for k in range(n_clusters):
    different += ((k_means_labels == k) != (som_labels == som.winner(k_means_cluster_centers[k])[1]))

identic = np.logical_not(different)
ax.plot(X[identic, 0], X[identic, 1], 'w',
        markerfacecolor='#bbbbbb', marker='.')
ax.plot(X[different, 0], X[different, 1], 'w',
        markerfacecolor='m', marker='.')
ax.set_title('Difference')
ax.set_xticks(())
ax.set_yticks(())

plt.show()