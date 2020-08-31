import numpy as np
import pandas as pd

def execute(input_file, output_file):

    """Performs preprocessing (removes not needed columns and records with missing data)
    Args:
        input_file (str): input raw data file
        output_file (str): output file after preprocessing
    """

    # Open data file:
    data = pd.read_csv(input_file, sep = ";")

    # Remove not needed columns:
    del(data["Name"])
    del(data["Ticket"])
    del(data["Cabin"])

    # Remove broken records:
    data = data.dropna() # dropna removes records with missing values so it should be executed after removing columns from data

    # Save data to the new file:
    data.to_csv(output_file, sep = ";", index = False)
