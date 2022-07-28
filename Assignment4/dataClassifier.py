# dataClassifier.py
# -----------------
# Licensing Information:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to UC Berkeley, including a link to http://ai.berkeley.edu.
# 
# Attribution Information: The Pacman AI projects were developed at UC Berkeley.
# The core projects and autograders were primarily created by John DeNero
# (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and
# Pieter Abbeel (pabbeel@cs.berkeley.edu).


# This file contains feature extraction methods and harness
# code for data classification

import classifiers
import sys
import util
from featureExtractor import *
import numpy as np

TRAIN_SET_SIZE = 5000
TEST_SET_SIZE = 5000
DIGIT_DATUM_WIDTH=28
DIGIT_DATUM_HEIGHT=28


def analysis(classifier, guesses, testLabels, testData, rawTestData):
    """
    This function is called after learning.
    Include any code that you want here to help you analyze your results.

    Use the printImage(<list of pixels>) function to visualize features.

    An example of use has been given to you.

    - classifier is the trained classifier
    - guesses is the list of labels predicted by your classifier on the test set
    - testLabels is the list of true labels
    - testData is the list of training datapoints (as util.Counter of features)
    - rawTestData is the list of training datapoints (as samples.Datum)
    - printImage is a method to visualize the features
    (see its use in the odds ratio part in runClassifier method)

    This code won't be evaluated. It is for your own optional use
    (and you can modify the signature if you want).
    """

    # Put any code here...
    # Example of use:
    # for i in range(len(guesses)):
    #     prediction = guesses[i]
    #     truth = testLabels[i]
    #     if (prediction != truth):
    #         print "==================================="
    #         print "Mistake on example %d" % i
    #         print "Predicted %d; truth is %d" % (prediction, truth)
    #         print "Image: "
    #         break


## =====================
## You don't have to modify any code below.
## =====================


def default(str):
    return str + ' [Default: %default]'

USAGE_STRING = """
  USAGE:      python dataClassifier.py <options>
  EXAMPLES:   python dataClassifier.py -c knn -d digits -t 6000 -n 5
                  - would run the KNN classifier on 6000 training examples
                  using the PCAFeatureExtractorDigit class to get the features
                 """


def readCommand( argv ):
    "Processes the command used to run from the command line."
    global ClassifierText
    from optparse import OptionParser
    parser = OptionParser(USAGE_STRING)

    parser.add_option('-c', '--classifier', help=default('The type of classifier'), choices=['knn', 'perceptron', 'svm', 'lr', 'best', 'none'], default='none')
    parser.add_option('-d', '--data', help=default('Dataset to use'), choices=['digits'], default='digits')
    parser.add_option('-t', '--training', help=default('The size of the training set'), default=TRAIN_SET_SIZE, type="int")
    parser.add_option('-i', '--iterations', help=default("Maximum iterations to run training"), default=50, type="int")
    parser.add_option('-s', '--test', help=default("Amount of test data to use"), default=TEST_SET_SIZE, type="int")
    parser.add_option('-n', '--neighbors', help=default("Number of clusters(for KMeans) or nearest neighbors(for KNN)"), default=5, type="int")
    parser.add_option('-v', '--visualize', help=default('Whether to visualize some results'), action='store_true')

    options, otherjunk = parser.parse_args(argv)
    if len(otherjunk) != 0: raise Exception('Command line input not understood: ' + str(otherjunk))
    args = {}

    # Set up variables according to the command line input.
    print ("Doing classification")
    print ("--------------------")
    print ("data:\t\t" + options.data)
    if options.data == 'digits':
        print ("classifier:\t\t" + options.classifier)
    else:
        print ("classifier:\t\t" + options.data + " classifier")
    print ("training set size:\t" + str(options.training))
    
    featureExtractor = BasicFeatureExtractorDigit()
    
    legalLabels = range(10)

    if options.training <= 0:
        print ("Training set size should be a positive integer (you provided: %d)" % options.training)
        print (USAGE_STRING)
        sys.exit(2)

    if options.classifier == "perceptron":
        classifier = classifiers.PerceptronClassifier(legalLabels, options.iterations)
    elif options.classifier == "svm":
        classifier = classifiers.SVMClassifier(legalLabels)
    elif options.classifier == "knn":
        classifier = classifiers.KNNClassifier(legalLabels, options.neighbors)
    elif options.classifier == 'lr':
        classifier = classifiers.LinearRegressionClassifier(legalLabels)
    elif options.classifier == 'best':
        classifier = classifiers.BestClassifier(legalLabels)
    else:
        print ("Do not use any classifier:", options.classifier)
        classifier = None

    args['classifier'] = classifier
    args['featureExtractor'] = featureExtractor

    return args, options


def runClassifier(args, options):
    featureExtractor = args['featureExtractor']
    classifier = args['classifier']

    # Load data
    numTraining = options.training
    numTest = options.test
    
    rawDigitData = np.load('data/digitdata/mnist.npz')
    rawTrainingData = rawDigitData['x_train'][:numTraining]
    rawTrainingLabels = rawDigitData['y_train'][:numTraining]
    rawValidationData = rawDigitData['x_valid'][:numTest]
    rawValidationLabels = rawDigitData['y_valid'][:numTest]
    rawTestData = rawDigitData['x_test'][:numTest]
    rawTestLabels = rawDigitData['y_test'][:numTest]

    # Extract features
    print ("Extracting features...")
    featureExtractor.fit(rawTrainingData)
    trainingData = featureExtractor.extract(rawTrainingData)
    validationData = featureExtractor.extract(rawValidationData)
    testData = featureExtractor.extract(rawTestData)
    trainingLabels = np.array(rawTrainingLabels)
    validationLabels = np.array(rawValidationLabels)
    testLabels = np.array(rawTestLabels)
    
    if classifier is not None:
        # Conduct training and testing
        print ("Training...")
        classifier.train(trainingData, trainingLabels, validationData, validationLabels)
        print ("Validating...")
        guesses = classifier.classify(validationData)
        correct = [guesses[i] == validationLabels[i] for i in range(len(validationLabels))].count(True)
        print (str(correct), ("correct out of " + str(len(validationLabels)) + " (%.1f%%).") % \
                            (100.0 * correct / len(validationLabels)))
        print ("Testing...")
        guesses = classifier.classify(testData)
        correct = [guesses[i] == testLabels[i] for i in range(len(testLabels))].count(True)
        print (str(correct), ("correct out of " + str(len(testLabels)) + " (%.1f%%).") % \
                                (100.0 * correct / len(testLabels)))

        analysis(classifier, guesses, testLabels, testData, rawTestData)

    # do odds ratio computation if specified at command line
    if options.visualize and options.classifier=='perceptron':
        import display
        weights = classifier.visualize()
        display.displayDigit(np.clip(weights, 0, 1), outfile='visualize/weights.png')

if __name__ == '__main__':
    np.random.seed(0)
    # Read input
    args, options = readCommand( sys.argv[1:] )
    # Run classifier
    runClassifier(args, options)
