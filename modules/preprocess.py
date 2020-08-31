import numpy as np
import pandas as pd

def execute(data_file):

    """Performs preprocessing (removes not needed columns and records with missing data)
    Args:
        data_file (str): path to input raw csv data file
    """

    # Open data file:
    data = pd.read_csv(data_file, sep = ";")

    # Remove not needed columns:
    del(data["PassengerId"])
    del(data["Name"])
    del(data["Ticket"])
    del(data["Cabin"])

    # Remove broken records:
    data = data.dropna() # dropna removes records with missing values so it should be executed after removing columns from data

    # Return preprocessed data:
    return data
