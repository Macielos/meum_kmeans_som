import os
import io

from data_loading.data_record import DataRecord


class SimpleDataLoader(object):
    separator = ','
    records = []

    def __init__(self, separator=','):
        self.separator = separator
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
            for line in file:
                # print(line)
                self.records.append(self.line_to_record(line))

    def line_to_record(self, line):
        parts = line.split(self.separator)
        return DataRecord(parts[0], parts[1:])


class SimpleWithDateDataLoader(SimpleDataLoader):

    def line_to_record(self, line):
        parts = line.split(self.separator)
        return DataRecord(parts[0]+parts[1], parts[2:])


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
