import numpy as np
import pandas as pd
import pickle as pkl # tool for object (de)serialization
import sklearn

from common import processed_suffix

def execute(data_file, model_path):

    """Performs model training
    Args:
        data_file (str): path to input raw csv data file
        model_path (str): path to file with model
    """

    # Split the data for training:
    df = pd.read_csv(data_file + processed_suffix, sep = ';')

    y = df["Survived"]

    tr_col = []
    for c in df.columns:
        if c == "Survived":
            pass
        else:
            tr_col.append(c)

    # Create a classifier and select scoring methods:
    from sklearn.ensemble import RandomForestClassifier
    clf = RandomForestClassifier(n_estimators=10)

    # Fit model and predict on validate data:
    clf.fit(df[tr_col], y)
    preds = clf.predict(df[tr_col])
    metric_name = "train_accuracy"
    metric_result = sklearn.metrics.accuracy_score(y, preds)

    # Save model to file:
    model_pickle = open(model_path, 'wb')
    pkl.dump(clf, model_pickle)
    model_pickle.close()

    # Return metrics and model:
    info = ""
    info = info + metric_name
    info = info + " for the model is "
    info = info + str(metric_result)
    print(info)
