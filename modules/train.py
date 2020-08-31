import numpy as np
import pandas as pd
import pickle as pkl # tool for object (de)serialization
import sklearn

from common import processed_suffix, printAccuracy

def execute(data_file, model_path):

    """Performs model training
    Args:
        data_file (str): path to input raw csv data file
        model_path (str): path to file with model
    """

    # Split the data for training:
    data = pd.read_csv(data_file + processed_suffix, sep = ';')

    target = data["Survived"]
    del(data["Survived"])

    tr_col = []
    for c in data.columns:
        if c == "Survived":
            pass
        else:
            tr_col.append(c)

    # Create a classifier and select scoring methods:
    from sklearn.ensemble import RandomForestClassifier
    clf = RandomForestClassifier(n_estimators=10)

    # Fit model and predict on training data:
    clf.fit(data[tr_col], target)
    predictions = clf.predict(data[tr_col])

    # Save model to file:
    model_pickle = open(model_path, 'wb')
    pkl.dump(clf, model_pickle)
    model_pickle.close()

    # Calculate and print accuracy:
    printAccuracy(target, predictions, "training data")
