import numpy as np
import pandas as pd
import pickle as pkl
import sklearn

def execute(input_file, model_path):

    # Split the data for training.
    df = pd.read_csv(input_file, sep = ';')

    y = df["Survived"]

    tr_col = []
    for c in df.columns:
        if c == "Survived":
            pass
        else:
            tr_col.append(c)

    # Create a classifier and select scoring methods.
    from sklearn.ensemble import RandomForestClassifier
    clf = RandomForestClassifier(n_estimators=10)

    # Fit full model and predict on both train and test.
    clf.fit(df[tr_col], y)
    preds = clf.predict(df[tr_col])
    metric_name = "train_accuracy"
    metric_result = sklearn.metrics.accuracy_score(y, preds)

    model_pickle = open(model_path, 'wb')
    pkl.dump(clf, model_pickle)
    model_pickle.close()

    # Return metrics and model.
    info = ""
    info = info + metric_name
    info = info + " for the model is "
    info = info + str(metric_result)
    print(info)
