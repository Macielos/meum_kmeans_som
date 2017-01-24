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

from algorithms.result_statistics import ResultStatistics

class AlgorithmsComparer(object):
    
    def __init__(self):
        return

    def compare(self, data, n_clusters, n_init, data_dimension, sigma, learning_rate, som_iterations):

        #K-means
        print('starting k-means')
        k_means = KMeans(init='k-means++', n_clusters=n_clusters, n_init=n_init)
        t0 = time.time()
        k_means.fit(data)
        t_batch = time.time() - t0
        print('k-means done in ' + str(t_batch))

        k_means_cluster_centers = np.sort(k_means.cluster_centers_, axis=0)
        k_means_clusters = pairwise_distances_argmin(data, k_means_cluster_centers)

        #SOM
        print('starting SOM')
        som = MiniSom(1, n_clusters, data_dimension, sigma, learning_rate)
        t0 = time.time()
        som.train_random(data, som_iterations)
        t_batch2 = time.time() - t0
        print('SOM done in ' + str(t_batch2))

        som_clusters_array = [None] * len(data)
        for index in range(len(data)):
            som_clusters_array[index] = som.winner(data[index])[1]
        som_clusters = np.array(som_clusters_array)

        #Counting
        k_means_clusters_size = [0]*n_clusters
        for element in k_means_clusters:
            k_means_clusters_size[element] = k_means_clusters_size[element] + 1

        som_clusters_size = [0]*n_clusters
        for element in som_clusters:
            transposed_cluster = som.winner(k_means_cluster_centers[element])[1]
            som_clusters_size[transposed_cluster] = som_clusters_size[transposed_cluster] + 1

        different = (som_clusters == -1)
        differentCount = 0
        for k in range(n_clusters):
            different += ((k_means_clusters == k) != (som_clusters == som.winner(k_means_cluster_centers[k])[1]))
        for cluster in different:
            for value in np.nditer(cluster):
                if value:
                    differentCount = differentCount + 1

        result = ResultStatistics(k_means_clusters_size, som_clusters_size, t_batch, t_batch2, differentCount)
        return result