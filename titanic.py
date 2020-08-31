#!/usr/bin/env python2.7
# -*- coding: utf-8 -*-

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
        clf = ClassifierClass() # classifier is added with default constructor, this of course may be adjusted in further analysis
        classifiers.append(clf)

    results = {}

    for classifier in classifiers:

        try:
            classifier_name = classifier.__class__.__name__

            print "Classifier: " + classifier_name

            # 1. Training:
            build_features.execute(train_csv)
            train.execute(train_csv, classifier)

            # 2. Inference:
            build_features.execute(val_csv)
            predict.execute(val_csv, classifier)

            print

        except:

            ""
