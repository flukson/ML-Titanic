#!/usr/bin/env python2.7
# -*- coding: utf-8 -*-

import sys
import warnings

if not sys.warnoptions:
    warnings.simplefilter("ignore")
    warnings.filterwarnings("ignore", category=DeprecationWarning)

from modules import build_features, train, predict

# Data pats:
data_dir = "./data/"

# Data for training:
train_csv = data_dir + "train.csv"

# Data for validation:
val_csv = data_dir + "val.csv"

if __name__ == '__main__':

    from sklearn.utils.testing import all_estimators

    estimators = all_estimators(type_filter='classifier')

    classifiers = []
    for name, ClassifierClass in estimators:
        clf = ClassifierClass() # classifier is added with a default constructor, which may be not optimal
        classifiers.append(clf)

    results = {}

    for classifier in classifiers:

        try:
            classifier_name = classifier.__class__.__name__

            #print "Classifier: " + classifier_name

            # Age and Fare not categorized:

            # A1. Training:
            build_features.execute(train_csv)
            accuracy_train = train.execute(train_csv, classifier)

            # A2. Inference:
            build_features.execute(val_csv)
            accuracy_val = predict.execute(val_csv, classifier)

            # Age and Fare categorized:

            # B1. Training:
            build_features.execute(train_csv, categorize=True)
            accuracy_train_c = train.execute(train_csv, classifier, categorize=True)

            # B2. Inference:
            build_features.execute(val_csv, categorize=True)
            accuracy_val_c = predict.execute(val_csv, classifier, categorize=True)

            results[classifier_name] = [accuracy_train, accuracy_val, accuracy_train_c, accuracy_val_c]

        except:

            ""

    sorted_results = sorted(results.items(), reverse=True, key=lambda x: x[1][1])

    print "============================="
    print "Age and Fare not categorized:"
    print "============================="
    for sr in sorted_results:
        print sr[0], str(sr[1][1])
    print

    sorted_results = sorted(results.items(), reverse=True, key=lambda x: x[1][3])

    print "========================="
    print "Age and Fare categorized:"
    print "========================="
    for sr in sorted_results:
        print sr[0], str(sr[1][3])
    print
