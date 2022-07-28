# featureExtractor.py

import sys
import util
import numpy as np
import display

TRAIN_SET_SIZE = 5000
TEST_SET_SIZE = 5000
DIGIT_DATUM_WIDTH=28
DIGIT_DATUM_HEIGHT=28

USAGE_STRING = """
  USAGE:      python featureExtractor.py <options>
  EXAMPLES:   python featureExtractor.py -f kmeans -m s
                  - would run the KMeans classifier on TRAIN_SET_SIZE training examples
                  using the KMeansFeatureExtractorDigit class to get the features
                 """

def default(str):
    return str + ' [Default: %default]'
    

class BaseFeatureExtractor(object):
    def __init__(self):
        pass
    
    def fit(self, trainingData):
        """
        Train feature extractor given the training Data
        :param trainingData: in numpy format
        :return:
        """
        pass
    
    def extract(self, data):
        """
        Extract the feature of data
        :param data: in numpy format
        :return: features, in numpy format and len(features)==len(data)
        """
        pass
    
    def visualize(self, data):
        pass


class BasicFeatureExtractorDigit(BaseFeatureExtractor):
    """
    Just regard the value of the pixels as features (in 784 dimensions)
    """
    def __init__(self):
        super(BasicFeatureExtractorDigit, self).__init__()

    def fit(self, trainingData):
        pass
    
    def extract(self, data):
        return data
    
    def visualize(self, data):
        # reconstruction and visualize
        display.displayDigit(data, outfile='visualize/original_digits.png')
        
        
class PCAFeatureExtractorDigit(BaseFeatureExtractor):
    """
    Principle Component Analysis(PCA)
    """

    def __init__(self, dimension):
        """
        self.weights: weight to learn in PCA, in numpy format and shape=(dimension, 784)
        self.mean: mean of training data, in numpy format

        :param dimension: dimension to reduction
        """
        super(PCAFeatureExtractorDigit, self).__init__()
        self.dimension = dimension
        self.weights = None
        self.mean = None

    def fit(self, trainingData):
        """
        Train PCA given the training Data

        Some numpy functions that may be of use (we consider np as short of numpy)
        np.mean(a, axis): mean value of array elements over a given axis
        np.linalg.svd(X, full_matrices=False): perform SVD decomposition to X
        np.dot(A, B): dot product of two arrays, or matrix multiplication between A and B.

        :param trainingData: in numpy format
        :return:
        """
        self.mean = np.mean(trainingData, axis=0)
        data = trainingData - self.mean
        _, _, VT = np.linalg.svd(data, full_matrices=False)
        self.weights = VT[:self.dimension]

        return np.dot(trainingData, self.weights.T)

    # util.raiseNotDefined()

    def extract(self, data):
        """

        :param data: in numpy format
        :return: features, in numpy format, features.shape = (len(data), self.dimension)
        """
        return np.dot(data - self.mean, self.weights.T)

    # util.raiseNotDefined()

    def reconstruct(self, pcaData):
        """
        Perform reconstruction of data given PCA features

        :param pcaData: in numpy format, features.shape[1] = self.dimension
        :return: originalData, in numpy format, originalData.shape[1] = 784
        """
        assert pcaData.shape[1] == self.dimension
        return self.mean + np.dot(pcaData, self.weights)

    def visualize(self, data):
        """
        Visualize data with both PCA and reconstruction
        :param data: in numpy format
        :return:
        """
        # extract features
        pcaData = self.extract(data)
        # reconstruction and visualize
        reconstructImg = self.reconstruct(pcaData)
        display.displayDigit(np.clip(reconstructImg, 0, 1), outfile='visualize/pca_digits.png')


class KMeansClusterDigit(BaseFeatureExtractor):
    """
    K-means clustering
    """
    def __init__(self, num_cluster, num_iterations):
        """
        :param num_cluster: number of clusters
        :param num_iterations: number of iterations
        """
        super(KMeansClusterDigit, self).__init__()
        self.num_cluster = num_cluster
        self.num_iterations = num_iterations
        self.clusters = None
    
    def fit(self, trainingData):
        """
        Perfrom K-means clustering.

        Some numpy functions that may be of use (we consider np as short of numpy)
        np.sum(a, axis): sum of array elements over a given axis.
        np.dot(A, B): dot product of two arrays, or matrix multiplication between A and B.
        np.min(a, axis), np.argmin(a, axis): return the minimum value (or indices) of an array over a given axis.

        :param trainingData: Data for clustering. (in numpy format)
        :return The clusters. It must be stored in self.clusters (in numpy format)
        """
        # initialization: random assign each data point to certain cluster, i.e. cluster_no
        # DO NOT change the initialization, especially the cluster_no
        np.random.seed(7)
        n, dim = trainingData.shape[0], trainingData.shape[1]
        cluster_no = np.random.randint(self.num_cluster, size=(n))
        self.clusters = np.zeros((self.num_cluster, dim))
        
        
        # performing K-means clustering
        # YOU SHOULD use the cluster_no for computing the clusters
        "*** YOUR CODE HERE ***"
        centroids = cluster_no
        Cluster = self.clusters
        flag = 1 # cluster change flag
        while flag:
            flag = 0
            for i in range(n):
                Num = -1
                Dis = 10000.0
                for j in range(self.num_cluster):
                    dis = np.sqrt(sum((Cluster[j, :] - trainingData[i, :]) ** 2))
                    if Dis > dis:
                        Dis = dis
                        Num = j
                if centroids[i] != Num:
                    centroids[i] = Num
                    flag = 1
                    #Cluster[i,:] = Dis ** 2
            for k in range(self.num_cluster):
                pointIncluster = trainingData[np.nonzero(centroids == k)]
                if(np.nonzero(centroids == k)[0].size):
                    Cluster[k, :] = np.mean(trainingData[np.nonzero(centroids == k)], axis = 0)
                
        return Cluster
        util.raiseNotDefined()
        
    def visualize(self, data):
        XX = np.sum(data*data, axis=1, keepdims=True)
        Cls2 = np.sum(self.clusters*self.clusters, axis=1)
        occupy = np.zeros(len(self.clusters), dtype=np.int32)
        D = -2*np.dot(data, self.clusters.T) + Cls2 + XX
        ind = np.argmin(D, axis=1)
        # print (ind)
        kmdigit = self.clusters[ind]
        display.displayDigit(np.clip(kmdigit, 0, 1), outfile='visualize/kmeans_digits.png')


if __name__ == '__main__':
    from optparse import OptionParser
    parser = OptionParser(USAGE_STRING)
    parser.add_option('-f', '--feature', help=default('Unsupervised method'), choices=['basic', 'pca', 'kmeans'], default='kmeans')
    parser.add_option('-s', '--size', help=default('Dimension size (PCA) or cluster size (KMeans)'), default=10, type=int)
    parser.add_option('-i', '--iterations', help=default("Maximum iterations to run training"), default=50, type=int)
    parser.add_option('-v', '--visualize', help=default('Whether to visualize some results'), action='store_true')
    
    options, otherjunk = parser.parse_args()
    if len(otherjunk) != 0: raise Exception('Command line input not understood: ' + str(otherjunk))

    if options.feature == 'basic':
        featureExtractor = BasicFeatureExtractorDigit()
    elif options.feature == 'pca':
        featureExtractor = PCAFeatureExtractorDigit(options.size)
    else:
        featureExtractor = KMeansClusterDigit(options.size, options.iterations)
    
    numTraining = TRAIN_SET_SIZE
    numTest = TEST_SET_SIZE
    
    print ('Loading data ...')
    rawDigitData = np.load('data/digitdata/mnist.npz')
    rawTrainingData = rawDigitData['x_train'][:numTraining]
    rawTrainingLabels = rawDigitData['y_train'][:numTraining]
    rawValidationData = rawDigitData['x_valid'][:numTest]
    rawValidationLabels = rawDigitData['y_valid'][:numTest]
    rawTestData = rawDigitData['x_test'][:numTest]
    rawTestLabels = rawDigitData['y_test'][:numTest]

    print ("Training with", options.feature, '...')
    featureExtractor.fit(rawTrainingData)
    
    if options.visualize:
        visdata = rawTrainingData[[np.argwhere(rawTrainingLabels==i)[0,0] for i in range(10)]]
        featureExtractor.visualize(visdata)