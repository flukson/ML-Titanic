import numpy as np
import pandas as pd

def execute(input_file, output_file):

    data = pd.read_csv(input_file, sep = ";")

    del(data["Name"])
    del(data["Ticket"])
    del(data["Cabin"])

    data = data.dropna() # dropna removes records with missing values so it should be executed after removing columns from data

    data.to_csv(output_file, sep = ";", index = False)
