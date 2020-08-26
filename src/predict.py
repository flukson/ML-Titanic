import pandas as pd
import pickle as pkl

def execute(input_file, model_path):

    df = pd.read_csv(input_file, sep = ';')

    df.dropna(inplace = True)

    target = df["Survived"]
    del(df["Survived"])

    model_unpickle = open(model_path, 'rb')
    model = pkl.load(model_unpickle)
    #model.close()

    predictions = model.predict(df)
    # Reassign target (if it was present) and predictions.
    df["prediction"] = predictions
    df["target"] = target

    ok = 0
    for i in df.iterrows():
        if (i[1]["target"] == i[1]["prediction"]):
            ok = ok + 1

    print("accuracy is", float(ok) / float(df.shape[0]))
