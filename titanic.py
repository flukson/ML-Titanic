#!/usr/bin/env python2.7
# -*- coding: utf-8 -*-

from modules import build_features, train, predict

# Data pats:
data_dir = "./data/"

# Data for training:
train_csv = data_dir + "train.csv"

# Data for validation:
val_csv = data_dir + "val.csv"

model_path = data_dir + "model.pkl"

if __name__ == '__main__':

    # Create a classifier and select scoring methods:
    from sklearn.ensemble import RandomForestClassifier
    clf = RandomForestClassifier(n_estimators=10)

    # 1. Training:
    build_features.execute(train_csv)
    train.execute(train_csv, model_path, clf)

    # 2. Inference:
    build_features.execute(val_csv)
    predict.execute(val_csv, model_path)
