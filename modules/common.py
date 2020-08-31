import sklearn

# Strings:

processed_suffix = '.processed'

# Functions:

def printAccuracy(target, predictions, label):

    accuracy = sklearn.metrics.accuracy_score(target, predictions)
    print "Accuracy for " + label + " is " + str(accuracy) + "."
