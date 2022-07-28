#!/usr/bin/python3

import sys, time, traceback
import os

import classifiers
from featureExtractor import *

## code to handle timeouts
import signal


# check whether sklearn is used
def checkSklearnPackage(grade, q):
    if 'sklearn' in sys.modules:
        grade[q] = 0
        raise Exception("Python packages sklearn is abandoned in this problem {}".format(q))


class TimeoutFunctionException(Exception):
    """Exception to raise on a timeout"""
    pass


class TimeoutFunction:

    def __init__(self, function, timeout):
        "timeout must be at least 1 second. WHY??"
        self.timeout = timeout
        self.function = function

    def handle_timeout(self, signum, frame):
        raise TimeoutFunctionException()

    def __call__(self, *args):
        if not 'SIGALRM' in dir(signal):
            return self.function(*args)
        old = signal.signal(signal.SIGALRM, self.handle_timeout)
        signal.alarm(self.timeout)
        # try:
        result = self.function(*args)
        # finally:
        # signal.signal(signal.SIGALRM, old)
        signal.alarm(0)
        return result


def getExceptionTraceBack():
    return ''.join(traceback.format_exception(*sys.exc_info())[-2:]).strip()     # .replace('\n',': ')


def display(str, minor = False):
    global allPassed
    if (not minor):
        allPassed = False
    str = str.replace("\n"," &&&\n&&& ")
    print ("&&& %s &&&" % str)


numTraining = 5000
numTest = 1000
DIGIT_DATUM_WIDTH = 28
DIGIT_DATUM_HEIGHT = 28


def readDigitData(trainingSize=5000, testSize=1000):
    print (trainingSize, testSize)
    rootdata = 'data/digitdata/'
    # loading digits data
    rawDigitData = np.load(rootdata+'mnist.npz')
    rawTrainingData = rawDigitData['x_train'][:trainingSize]
    trainingLabels = rawDigitData['y_train'][:trainingSize]
    rawValidationData = rawDigitData['x_valid'][:testSize]
    validationLabels = rawDigitData['y_valid'][:testSize]
    rawTestData = rawDigitData['x_test'][:testSize]
    testLabels = rawDigitData['y_test'][:testSize]
    return (rawTrainingData, trainingLabels, rawValidationData, validationLabels, rawTestData, testLabels)


def getAccuracy(data, classifier, featureExtractor=BasicFeatureExtractorDigit()):
    rawTrainingData, trainingLabels, rawValidationData, validationLabels, rawTestData, testLabels = data
    featureExtractor.fit(rawTrainingData)
    trainingData = featureExtractor.extract(rawTrainingData)
    validationData = featureExtractor.extract(rawValidationData)
    testData = featureExtractor.extract(rawTestData)
    classifier.train(trainingData, trainingLabels, validationData, validationLabels)
    guesses = classifier.classify(testData)
    correct = [guesses[i] == testLabels[i] for i in range(len(testLabels))].count(True)
    acc = 100.0 * correct / len(testLabels)
    print (str(correct), ("correct out of " + str(len(testLabels)) + " (%.1f%%).") % (acc))
    return acc

digitData = readDigitData(numTraining, numTest)


def q1(grades):
    print ("\n===== Grading Q1 ===== ")
    
    grades[1] = 0
    
    print ("1. checking python featureExtractor.py -f kmeans -n 10 -i 50")
    try:
        rawTrainingData = digitData[0]
        featureExtractor = KMeansClusterDigit(10, 50)
        tmp = featureExtractor.fit(rawTrainingData)
        if tmp is None:
            cluster = featureExtractor.clusters
        else:
            cluster = tmp
        cluster_ans = np.load('data/cluster_mnist_seed_7.npy')
        ddiff = -2*np.dot(cluster, cluster_ans.T) + np.sum(cluster*cluster, axis=1, keepdims=True) + np.sum(cluster_ans*cluster_ans, axis=1, keepdims=True).T
        diff = np.sum(np.min(ddiff, axis=1))
        print ('Total cluster distance:', diff)
        if diff < 30:
            grades[1] += 2
        if diff < 1:
            grades[1] += 2
        checkSklearnPackage(grades, 1)
    except:
        display("An exception was raised:\n%s" % getExceptionTraceBack())



def q2(grades):
    print ("\n===== Grading Q2 ===== ")

    grades[2] = 0

    print ("1. checking python dataClassifier.py -d digits -c knn -t 5000 -s 1000 -f basic -n 5")
    try:
        legalLabels = range(10)
        classifier = classifiers.KNNClassifier(legalLabels, 5)
        accDigits = getAccuracy(digitData, classifier) # our solution: 91.0%
        accThres = [60, 80, 88]
        score = [accDigits >= t for t in accThres].count(True)
        grades[2] = score
        checkSklearnPackage(grades, 2)
    except:
        display("An exception was raised:\n%s" % getExceptionTraceBack())


def q3(grades):
    print ("\n===== Grading Q3 ===== ")

    grades[3] = 0

    print ("2. checking python dataClassifier.py -d digits -c perceptron -t 5000 -s 1000 -f basic -i 50")
    try:
        legalLabels = range(10)
        classifier = classifiers.PerceptronClassifier(legalLabels, 50)
        accDigits = getAccuracy(digitData, classifier)  # our solution: 87.4%
        accThres = [50, 70, 80, 85]
        score = [accDigits >= tt for tt in accThres].count(True)
        grades[3] = score

        checkSklearnPackage(grades, 3)
    except:
        display("An exception was raised:\n%s" % getExceptionTraceBack())


def q4(grades):
    print ("\n===== Grading Q4 ===== ")

    grades[4] = 0

    print ("4. checking python dataClassifier.py -d digits -c svm -t 5000 -s 1000")
    try:
        legalLabels = range(10)
        classifier = classifiers.SVMClassifier(legalLabels)

        # good classification result (1 point)
        accDigits = getAccuracy(digitData, classifier) # our solution: 93%
        if accDigits >= 92:
            grades[4] += 1
            
        # right hyper-parameter (2 points)
        from sklearn import svm
        import hashlib
        q4_param_test = {'C': '743da23f34ee7adc1bc280808926e060c096910e',
                        'kernel': 'bded05acba64254e04ec70571c0962bb9469cf0e',
                        'gamma': 'c9d439fd9b3c41a54dae29e8d37bcdf513c59e41'}
        for attr in classifier.__dict__:
            obj = getattr(classifier, attr)
            if isinstance(obj, svm.SVC):
                param = obj.get_params()
                for k in q4_param_test:
                    p = param[k] if isinstance(param[k], str) else str(float(param[k]))
                    print (k, p, hashlib.sha1(p.encode('utf-8')).hexdigest())
                    if hashlib.sha1(p.encode('utf-8')).hexdigest() != q4_param_test[k]:
                        raise Exception(""" TEST FAILED: error initialize the SVM param <{}>.
                                        """.format(k))
                break
        
        grades[4] += 1
    except:
        display("An exception was raised:\n%s" % getExceptionTraceBack())


def q5(grades):
    print ("\n===== Grading Q5 ===== ")

    grades[5] = 0

    print ("5. checking python dataClassifier.py -d digits -c best -t 5000 -s 1000")
    try:
        legalLabels = range(10)
        classifier = classifiers.BestClassifier(legalLabels)
        accDigits = getAccuracy(digitData, classifier)
        accThres = [94, 94.5, 95, 95.5]
        score = [accDigits >= tt for tt in accThres].count(True)
        grades[5] = score*0.5

    except:
        display("An exception was raised:\n%s" % getExceptionTraceBack())


if __name__ == '__main__':
    np.random.seed(0)
    NUM_TESTS = 5
    USAGE_STRING = """
      USAGE:      python classificationAutograder.py <options>
      EXAMPLES:   python classificationAutograder.py -q 1,2 -o score
                      - would test question 1,2, and the output would be in file score
                  python classificationAutograder.py
                      - would test all questions and print the output on stdout
                     """

    from optparse import OptionParser

    parser = OptionParser(USAGE_STRING)

    parser.add_option('-o', '--output', help='Save the autograder results to file output', default='stdout')
    parser.add_option('-q', '--q', help='Test No. of question. If 0, then test all questions', default='0')

    options, otherjunk = parser.parse_args(sys.argv[1:])
    if len(otherjunk) != 0: raise Exception('Command line input not understood: ' + str(otherjunk))

    if options.output != 'stdout':
        sys.stdout = file(options.output, 'w')
        sys.stderr = sys.stdout
    print ('Autograder transcript for project 3\n', 'Starting on %d-%d at %d:%02d:%02d' % time.localtime()[1:6])
    grades = {}
    print ('\n+++++++++++++++++++++++++++++++++++++++')
    print ('+++ AUTOGRADING FOR THIS SUBMISSION +++')
    print ('+++++++++++++++++++++++++++++++++++++++')
    if options.q == '0':
        tests = [1, 2, 3, 4, 5]
    else:
        tests = [int(t) for t in options.q.strip().split(',')]
        for t in tests:
            if t <= 0 or t > NUM_TESTS:
                raise Exception('Test No. of question illegal.')
    for t in tests:
        TimeoutFunction(getattr(sys.modules[__name__], 'q'+str(t)), 1200)(grades)
    print ("\n")
    full_credits = {1:4, 2:3, 3:4, 4:2, 5:2}
    for t in tests:
      print ("GRADE FOR QUESTION %d: %.1f / %.1f" % (t, grades[t], full_credits[t]))
    print ('+++++++++++++++++++++++++++++++++++++++')
    print ('TOTAL: %.1f/%.1f' % (sum([grades[g] for g in tests]), sum([full_credits[g] for g in tests])))
    print ('+++++++++++++++++++++++++++++++++++++++')
    print ('Done with Autograder on %d-%d at %d:%02d:%02d' % time.localtime()[1:6])
    sys.stdout.close
