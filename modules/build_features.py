import pandas as pd
import sklearn

import preprocess
from common import processed_suffix

def execute(data_file, categorize=False):

    """Builds features
    Args:
        data_file (str): path to input raw csv data file
        categorize: set to True if Age and Fare should be categorized
    """

    # Read preprocessed data:
    data = preprocess.execute(data_file)

    # Replace sex strings with binary value:
    data["Sex"] = data["Sex"].replace("male", 0)
    data["Sex"] = data["Sex"].replace("female", 1)

    if categorize:

        print "Features Age and Fare categorized."

        # Convert age into categories:
        data["Age"] = data["Age"].astype(int)
        data.loc[ data["Age"] <= 19, "Age"] = 0
        data.loc[(data["Age"] > 19) & (data["Age"] <= 25), "Age"] = 1
        data.loc[(data["Age"] > 25) & (data["Age"] <= 32), "Age"] = 2
        data.loc[(data["Age"] > 32) & (data["Age"] <= 42), "Age"] = 3
        data.loc[(data["Age"] > 42), "Age"] = 4

        # Convert fare into categories:
        data.loc[ data["Fare"] <= 7.854, "Fare"] = 0
        data.loc[(data["Fare"] > 7.854) & (data["Fare"] <= 10.5), "Fare"] = 1
        data.loc[(data["Fare"] > 10.5) & (data["Fare"] <= 22.225), "Fare"] = 2
        data.loc[(data["Fare"] > 22.225) & (data["Fare"] <= 39.688), "Fare"] = 3
        data.loc[(data["Fare"] > 39.688), "Fare"] = 4

    else:

        print "Features Age and Fare not categorized."

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
