import sklearn

# Strings:

processed_suffix = '.processed'

# Functions:

def getAccuracy(target, predictions):

    return sklearn.metrics.accuracy_score(target, predictions)
