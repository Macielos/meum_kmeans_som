import os
import io
import numpy as np
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction import DictVectorizer
from sklearn.feature_extraction.text import HashingVectorizer
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import Normalizer

from data_loading.data_record import DataRecord

from sklearn.datasets import fetch_20newsgroups

class SimpleDataLoader(object):
    records = []
    separator = ','
    limit = -1
    skip_header = True
    use_hashing = False

    def __init__(self, features=0, separator=',', skip_header=True, limit=-1):
        self.hasher = HashingVectorizer(n_features=2**18,
                               stop_words='english', non_negative=True,
                               norm=None, binary=False)
        self.dictVectorizer = DictVectorizer()

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

    def postprocess(self, records):
        return records

    def get(self):
        records_to_process = []
        for record in self.records:
            records_to_process.append(record.data)
        return self.postprocess(records_to_process)

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


class FeatureExtractingDataLoader(SimpleDataLoader):

    def postprocess(self, records):
        records_to_process = []
        for record in records:
            records_to_process.append(" ".join(record).strip('\n'))
        records2 = self.hasher.transform(records_to_process)
        records3 = self.lsa.fit_transform(records2)
        return records3


class SimpleWithDateDataLoader(SimpleDataLoader):

    def line_to_record(self, line):
        parts = line.split(self.separator)
        return DataRecord(parts[0] + parts[1], self.to_float_array(parts[2:]))


class LongTextDataLoader(FeatureExtractingDataLoader):
    #TODO znalezc lepsza liste i czytac ja z pliku
    stop_words = {'is', 'am', 'are', 'a', 'an', 'the', 'in', 'on', 'at', 'and', 'or'}

    def read_file(self, filename):
        #print('reading data from file: '+filename)

        words = set()
        with io.open(filename, "r", encoding="utf8") as file:
            #TODO stop words, formy podst. itd
            for line in file:
                words.update(set(line.lower().replace('\s', ' ').split(self.separator)).difference(self.stop_words))
        self.records.append(DataRecord(filename, words))
