#!/usr/bin/env python2.7
# -*- coding: utf-8 -*-

from src import preprocess

# paths:
data_dir = "./data/"
# data for training:
train_csv = data_dir + "train.csv"
train_df_csv = data_dir + "train_df.csv"
# data for validation:
val_csv = data_dir + "val.csv"
val_df_csv = data_dir + "val_df.csv"

if __name__ == '__main__':

    # 1. preprocess
    preprocess.execute(train_csv, train_df_csv)

    # 2. build features
    # 3. train
    # 4. predict
