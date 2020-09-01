#!/usr/bin/env python2.7
# -*- coding: utf-8 -*-

import argparse
from sklearn.utils.testing import all_estimators
import sys
import warnings

if not sys.warnoptions:
    warnings.simplefilter("ignore")
    warnings.filterwarnings("ignore", category=DeprecationWarning)

from modules import build_features, train, predict

# Data for training:
train_csv = "train.csv"

# Data for validation:
val_csv = "val.csv"

if __name__ == '__main__':


    parser = argparse.ArgumentParser(description='Analyze Titanic dataset',
                                   formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('-dd', '--data-dir',
                        dest='data_dir',
                        type=str,
                        default='./data/',
                        help='relative path do data subdirectory')
    args = parser.parse_args()
    data_dir = args.data_dir

    estimators = all_estimators(type_filter='classifier')

    classifiers = []
    for name, ClassifierClass in estimators:
        clf = ClassifierClass() # classifier is added with a default constructor, which may be not optimal
        classifiers.append(clf)

    results = {}

    for classifier in classifiers:

        try:
            classifier_name = classifier.__class__.__name__

            print "Start calculations for " + classifier_name + " classifier."

            # Age and Fare not categorized:

            # A1. Training:
            build_features.execute(data_dir, train_csv)
            accuracy_train = train.execute(data_dir, train_csv, classifier)

            # A2. Inference:
            build_features.execute(data_dir, val_csv)
            accuracy_val = predict.execute(data_dir, val_csv, classifier)

            # Age and Fare categorized:

            # B1. Training:
            build_features.execute(data_dir, train_csv, categorize=True)
            accuracy_train_c = train.execute(data_dir, train_csv, classifier, categorize=True)

            # B2. Inference:
            build_features.execute(data_dir, val_csv, categorize=True)
            accuracy_val_c = predict.execute(data_dir, val_csv, classifier, categorize=True)

            print "Stop calculations for " + classifier_name + " classifier."

            results[classifier_name] = [accuracy_train, accuracy_val, accuracy_train_c, accuracy_val_c]

        except:

            ""

    print

    sorted_results = sorted(results.items(), reverse=True, key=lambda x: x[1][1])

    print "================================"
    print "A. Age and Fare not categorized:"
    print "================================"
    for sr in sorted_results:
        print sr[0], str(sr[1][1])
    print

    sorted_results = sorted(results.items(), reverse=True, key=lambda x: x[1][3])

    print "============================"
    print "B. Age and Fare categorized:"
    print "============================"
    for sr in sorted_results:
        print sr[0], str(sr[1][3])
    print
