import pandas as pd
import sklearn

import preprocess
from common import processed_suffix

def execute(data_file):

    """Builds features
    Args:
        data_file (str): path to input raw csv data file
    """

    # Read preprocessed data:
    data = preprocess.execute(data_file)

    # Replace sex strings with binary value:
    data["Sex"] = data["Sex"].replace("male", 0)
    data["Sex"] = data["Sex"].replace("female", 1)

    # Embarked: C = Cherbourg, Q = Queenstown, S = Southampton
    # Replace above labels with numbers from 1 to 3:
    embarked_dict = {}
    embarked_dict_values = 0
    for i in data.Embarked:
        if i in embarked_dict.keys():
            pass
        else:
            embarked_dict_values = embarked_dict_values + 1
            embarked_dict[i] = embarked_dict_values
    for i in embarked_dict.keys():
        data["Embarked"].replace(i, embarked_dict[i], inplace = True)

    # Add columns FamilySize and IsAlone:
    data["FamilySize"] = data["SibSp"] + data["Parch"] + 1
    data["IsAlone"] = 0
    data.loc[data["FamilySize"] == 1, "IsAlone"] = 1

    data.to_csv(data_file + processed_suffix, sep = ";", index = False)
