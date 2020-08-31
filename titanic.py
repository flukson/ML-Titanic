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

    classifiers = []

    from sklearn.ensemble import RandomForestClassifier
    classifiers.append(RandomForestClassifier(n_estimators=10))

    from sklearn.svm import LinearSVC
    classifiers.append(LinearSVC())

    from sklearn.linear_model import Perceptron
    classifiers.append(Perceptron(max_iter=10))

    from sklearn.tree import DecisionTreeClassifier
    classifiers.append(DecisionTreeClassifier())

    results = {}

    for classifier in classifiers:

        classifier_name = classifier.__class__.__name__

        print "Classifier: " + classifier_name

        # 1. Training:
        build_features.execute(train_csv)
        train.execute(train_csv, classifier)

        # 2. Inference:
        build_features.execute(val_csv)
        predict.execute(val_csv, classifier)

        print
