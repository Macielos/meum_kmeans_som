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

    def compare(self, iteration, data_tuples, data_array, n_clusters, n_init, data_dimension, sigma, learning_rate, som_iterations):

        #K-means
        print('starting k-means')
        k_means = KMeans(init='k-means++', n_clusters=n_clusters, n_init=n_init)
        t0 = time.time()
        k_means.fit(data_tuples)
        t_batch = time.time() - t0
        k_means_time = str(t_batch)
        print('k-means done in ' + k_means_time)

        k_means_cluster_centers = np.sort(k_means.cluster_centers_, axis=0)
        k_means_clusters = pairwise_distances_argmin(data_tuples, k_means_cluster_centers)

        #SOM
        print('starting SOM')
        som = MiniSom(1, n_clusters, data_dimension, sigma, learning_rate)
        t0 = time.time()
        som.train_random(data_array, som_iterations)
        t_batch2 = time.time() - t0
        som_time = str(t_batch2)
        print('SOM done in ' + som_time)

        som_clusters_array = [None] * len(data_array)
        for index in range(len(data_array)):
            som_clusters_array[index] = som.winner(data_array[index])[1]
        som_clusters = np.array(som_clusters_array)

        #Counting
        k_means_clusters_size = [0]*n_clusters
        k_means_min = len(data_tuples)
        k_means_max = 0
        k_means_sum = 0
        for element in k_means_clusters:
            k_means_clusters_size[element] = k_means_clusters_size[element] + 1

        for size in k_means_clusters_size:
            if size < k_means_min:
                k_means_min = size
            if size > k_means_max:
                k_means_max = size
            k_means_sum = k_means_sum + size

        k_means_mean = k_means_sum/n_clusters
        k_means_std = np.std(np.array(k_means_clusters_size))

        som_clusters_size = [0]*n_clusters
        som_min = len(data_array)
        som_max = 0
        som_sum = 0
        for element in som_clusters:
            transposed_cluster = som.winner(k_means_cluster_centers[element])[1]
            som_clusters_size[transposed_cluster] = som_clusters_size[transposed_cluster] + 1

        for size in som_clusters_size:
            if size < som_min:
                som_min = size
            if size > som_max:
                som_max = size
            som_sum = som_sum + size

        som_mean = som_sum / n_clusters
        som_std = np.std(np.array(som_clusters_size))

        different = (som_clusters == -1)
        differentCount = 0
        for k in range(n_clusters):
            different += ((k_means_clusters == k) != (som_clusters == som.winner(k_means_cluster_centers[k])[1]))
        for cluster in different:
            for value in np.nditer(cluster):
                if value:
                    differentCount = differentCount + 1

        result = ResultStatistics(k_means_clusters_size, som_clusters_size, t_batch, t_batch2, differentCount, k_means_min, som_min, k_means_max, som_max, k_means_mean, som_mean, k_means_std, som_std)
        print('k-means')
        print(k_means_clusters_size)
        print('min - ' + str(k_means_min))
        print('max - ' + str(k_means_max))
        print('mean - ' + str(k_means_mean))
        print('std - ' + str(k_means_std))
        print('som')
        print(som_clusters_size)
        print('min - ' + str(som_min))
        print('max - ' + str(som_max))
        print('mean - ' + str(som_mean))
        print('std - ' + str(som_std))

        with open("results.csv", "a") as result_file:
            result_file.write(";".join([str(iteration), str(n_clusters), str(len(data_array)), str(learning_rate), str(som_iterations), " ", " ", str(k_means_time), str(k_means_clusters_size),str(k_means_min),str(k_means_max),str(k_means_mean),str(k_means_std), " ", " ", str(som_time), str(som_clusters_size),str(som_min),str(som_max),str(som_mean),str(som_std)])+"\n")
        return result
