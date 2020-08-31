import pandas as pd
import sklearn

def execute(input_file, output_file):

    """Builds features
    Args:
        input_file (str): input file after preprocessing
        output_file (str): output file with additional features
    """

    # Read data:
    df = pd.read_csv(input_file, sep = ';')

    # Replace sex strings with binary value:
    df["Sex"] = df["Sex"].replace("male", 0)
    df["Sex"] = df["Sex"].replace("female", 1)

    # Embarked: C = Cherbourg, Q = Queenstown, S = Southampton
    # Replace above labels with numbers from 1 to 3:
    embarked_dict = {}
    embarked_dict_values = 0
    for i in df.Embarked:
        if i in embarked_dict.keys():
            pass
        else:
            embarked_dict_values = embarked_dict_values + 1
            embarked_dict[i] = embarked_dict_values
    for i in embarked_dict.keys():
        df["Embarked"].replace(i, embarked_dict[i], inplace = True)

    # Add column FamilySize and IsAlone:
    df["FamilySize"] = df["SibSp"] + df["Parch"] + 1
    df["IsAlone"] = 0
    df.loc[df["FamilySize"] == 1, "IsAlone"] = 1

    df.to_csv(output_file, sep = ";", index = False)
