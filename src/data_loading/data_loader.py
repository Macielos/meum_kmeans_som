import os
import io
import numpy as np
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import HashingVectorizer
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import Normalizer

from data_loading.data_record import DataRecord

class SimpleDataLoader(object):
    records = []
    separator = ','
    limit = -1
    skip_header = True
    use_hashing = False

    def __init__(self, features=0, separator=',', skip_header=True, limit=-1):
        self.hasher = HashingVectorizer(n_features=features*100,
                               stop_words='english', non_negative=True,
                               norm=None, binary=False)

        if features > 0:
            self.use_hashing = True
            self.svd = TruncatedSVD(features)
            self.normalizer = Normalizer(copy=False)
            self.lsa = make_pipeline(self.svd, self.normalizer)

        self.separator = separator
        self.limit = limit
        self.skip_header = skip_header
        self.records = []

    def load(self, directory):
        if not os.path.isdir(directory):
            self.read_file(directory)
            return
        for file in os.listdir(directory):
            full_name = directory+os.sep+file
            if os.path.isdir(full_name):
                self.load(full_name)
            else:
                self.read_file(full_name)

    def read_file(self, filename):
        print('reading data from file: '+filename)
        with io.open(filename, "r", encoding="utf8") as file:
            i = 0
            first = True
            for line in file:
                if self.skip_header and first:
                    first = False
                    continue
                if i >= self.limit >= 0:
                    break
                # print(line)
                self.records.append(self.line_to_record(line))
                i = i+1

    def line_to_record(self, line):
        parts = line.split(self.separator)
        return DataRecord(parts[0], parts[1:])
    """
    def get_tuples(self):
        records = []
        for record in self.records:
            data = record.data
            if self.use_hashing:
                data = self.hasher.fit_transform(data)
                data = self.lsa.fit_transform(data)
            records.append(tuple(data))
        return np.array(records)
    """
    def get_array(self):
        return self.get(False)

    def get_tuples(self):
        return self.get(True)

    def get(self, is_tuple):
        if not self.use_hashing:
            records = []
            for record in self.records:
                records.append(record.data)
            return records

        records = []
        for record in self.records:
            records.append(" ".join(record.data))

        records = self.hasher.fit_transform(records)
        records = self.lsa.fit_transform(records)

        if is_tuple:
            return tuple(records)
        else:
            return records

    def to_float_array(self, array):
        number_array = []
        for element in array:
            if self.is_float(element):
                number_array.append(float(element))
            else:
                number_array.append(0.0)#TODO
        return number_array

    def is_float(self, value):
      try:
        float(value)
        return True
      except ValueError:
        return False

class SimpleWithDateDataLoader(SimpleDataLoader):

    def line_to_record(self, line):
        parts = line.split(self.separator)
        return DataRecord(parts[0] + parts[1], self.to_float_array(parts[2:]))

class LongTextDataLoader(SimpleDataLoader):
    #TODO znalezc lepsza liste i czytac ja z pliku
    stop_words = {'is', 'am', 'are', 'a', 'an', 'the', 'in', 'on', 'at', 'and', 'or'}

    def read_file(self, filename):
        print('reading data from file: '+filename)

        words = set()
        with io.open(filename, "r", encoding="utf8") as file:
            #TODO stop words, formy podst. itd
            for line in file:
                words.update(set(line.lower().replace('\s', ' ').split(self.separator)).difference(self.stop_words))
        self.records.append(DataRecord(filename, words))
