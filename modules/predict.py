import pandas as pd
import pickle as pkl

from common import processed_suffix

def execute(data_file, model_path):

    """Performs prediction
    Args:
        data_file (str): path to input raw csv data file
        model_path (str): path to file with model
    """

    df = pd.read_csv(data_file + processed_suffix, sep = ';')

    df.dropna(inplace = True)

    target = df["Survived"]
    del(df["Survived"])

    # Load model from file:
    model_unpickle = open(model_path, 'rb')
    model = pkl.load(model_unpickle)

    predictions = model.predict(df)

    # Reassign target (if it was present) and predictions:
    df["prediction"] = predictions
    df["target"] = target

    # Check and print accuracy of the model:
    ok = 0
    for i in df.iterrows():
        if (i[1]["target"] == i[1]["prediction"]):
            ok = ok + 1
    print("accuracy is", float(ok) / float(df.shape[0]))
