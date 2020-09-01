import pandas as pd
import pickle as pkl

from common import processed_suffix, getAccuracy

def execute(data_dir, data_file, clf, categorize=False):

    """Performs prediction
    Args:
        data_dir (str): relative path to data subdirectory
        data_file (str): name of csv data file
        clf: classifier
        categorize: set to True if Age and Fare should be categorized
    """

    data = pd.read_csv(data_dir + data_file + processed_suffix + "_" + str(categorize), sep = ';')

    target = data["Survived"]
    del(data["Survived"])

    # Load model from file:
    model_unpickle = open(data_dir + clf.__class__.__name__ + ".pkl", 'rb')
    model = pkl.load(model_unpickle)

    # Predict data with loaded model:
    predictions = model.predict(data)

    # Calculate and print accuracy:
    accuracy = getAccuracy(target, predictions)
    return accuracy
