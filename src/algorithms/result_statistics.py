class ResultStatistics(object):

    kMeansClustersSizes = []
    somClustersSizes = []
    differentCount = 0
    kMeansTime = 0
    somTime = 0
    kMeansMin = 0
    somMin = 0
    kMeansMax = 0
    somMax = 0
    kMeansMean = 0
    somMean = 0
    kMeansStdDev = 0
    somStdDev = 0


    def __init__(self, kMeansClusterSizes, somClusterSizes, kMeansTime, somTime, differentCount, kMeansMin,
                 somMin, kMeansMax, somMax, kMeansMean, somMean, kMeansStdDev, somStdDev):
        self.kMeansClusterSizes = kMeansClusterSizes
        self.somClusterSizes = somClusterSizes
        self.kMeansTime = kMeansTime
        self.somTime = somTime
        self.differentCount = differentCount
        self.kMeansMin = kMeansMin
        self.somMin = somMin
        self.kMeansMax = kMeansMax
        self.somMax = somMax
        self.kMeansMean = kMeansMean
        self.somMean = somMean
        self.kMeansStdDev = kMeansStdDev
        self.somStdDev = somStdDev
