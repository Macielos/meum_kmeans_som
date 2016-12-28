class DataRecord(object):

    name = ""
    data = []

    def __init__(self, name, data):
        self.name = name
        self.data = data

    def __str__(self):
        return 'name: '+self.name+', data ('+str(len(self.data))+'): '+', '.join(map(str, self.data))
