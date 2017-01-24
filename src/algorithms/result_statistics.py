class ResultStatistics(object):

    kMeansClustersSizes = []
    somClustersSizes = []
    differentCount = 0
    kMeansTime = 0
    somTime = 0


    def __init__(self, kMeansClusterSizes, somClusterSizes, kMeansTime, somTime, differentCount):
        self.kMeansClusterSizes = kMeansClusterSizes
        self.somClusterSizes = somClusterSizes
        self.kMeansTime = kMeansTime
        self.somTime = somTime
        self.differentCount = differentCount
