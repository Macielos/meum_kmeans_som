"""

TODO
- wczytywanie danych - na wejściu plik, na wyjściu mapa/lista krotek nazwa->array danych
- odpalanie SOM i k-means - na wejsciu krotki jw, na wyjsciu lista klastrow
- statsy, grafy, itd.
"""

# plants
from data_loading.data_loader import SimpleDataLoader, SimpleWithDateDataLoader, LongTextDataLoader, FeatureExtractingDataLoader
from algorithms.algorithms import AlgorithmsComparer
from sklearn.datasets.samples_generator import make_blobs
import numpy as np

def print_data(loader):
    print('loaded ' + str(len(loader.records)) + ' records')
    #print('first 10 records: \n' + (', '.join(map(str, loader.records[0:10]))))
    #print('last 10 records: \n' + (', '.join(map(str, loader.records[-10:-1]))))

print('loading data...')
"""
plants_data_loader = FeatureExtractingDataLoader(1000, limit=1000)
plants_data_loader.load('../data/plants')
print_data(plants_data_loader)
"""
power_cons_data_loader = SimpleWithDateDataLoader(0, ';', True, 20000)
power_cons_data_loader.load('../data/power_consumption')
print_data(power_cons_data_loader)
"""
reuters_data_loader = LongTextDataLoader(2**10, ' ', False, 1000)
reuters_data_loader.load('../data/reuters/C50train')
print_data(reuters_data_loader)
"""
print('data loaded, now clustering magic begins')

data_loader = power_cons_data_loader

array = data_loader.get()
tuples = tuple(array)

comparer = AlgorithmsComparer()
# comparer.compare(tuples, array, 3, 100, len(array[0]), 0.3, 0.5, 100)

with open("results.csv", "a") as result_file:
    result_file.write(";".join(["i", "cluster_count", "records", "som_learning_rate", "som_iterations", " ", "SOM", "time", "clusters", "min", "max", "mean", "std", "K-MEANS", " ", "time", "clusters", "min", "max", "mean", "std"])+"\n")

for cluster_count in [3, 5, 10]:
    for i in range(0, 3):
        comparer.compare(i, tuples, array, cluster_count, 100, len(array[0]), 0.3, 0.5, 100)

for limit in [100, 1000, -1]:
    arrayLimited = array[:limit] if limit > 0 else array
    tupleLimited = tuple(arrayLimited)
    for i in range(0, 3):
        comparer.compare(i, tupleLimited, arrayLimited, 5, 100, len(arrayLimited[0]), 0.3, 0.5, 100)

for learning_rate in [0.1, 0.3, 0.6]:
    for i in range(0, 3):
        comparer.compare(i, tuples, array, 5, 100, len(array[0]), 0.3, learning_rate, 100)

for som_iterations in [10, 100, 200]:
    for i in range(0, 3):
        comparer.compare(i, tuples, array, 5, 100, len(array[0]), 0.3, 0.5, som_iterations)
