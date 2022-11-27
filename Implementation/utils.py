import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder

# Note: Many .csv files in the data directory were downloaded from https://github.com/tailequy/fairness_dataset/tree/main/experiments/data
# Paper: https://arxiv.org/pdf/2110.00530.pdf



def load_lawSchool_dataset(use_gender_as_sensitive_attribute=True):
    # Load data
    df = pd.read_csv("data/law_school_clean.csv")
    
    # Extract label and sensitive attribute
    y = df["pass_bar"].to_numpy().flatten()
    if use_gender_as_sensitive_attribute is True:
        y_sensitive = df["male"].to_numpy().flatten()
    else:
        y_sensitive = df["race"]
        y_sensitive = (y_sensitive == "White").astype(int)

    del df["pass_bar"]

    # Remove other columns and create final data set
    del df["male"];del df["race"]

    X = df.to_numpy()

    return X, y, y_sensitive


# https://archive.ics.uci.edu/ml/datasets/default+of+credit+card+clients
def load_creditCardClients_dataset():
    # Load data
    df = pd.read_csv("data/credit-card-clients.csv")

    # Extract label and sensitive attribute (AGE could also be used as a sensitive attribute)
    y_sensitive = df["SEX"].to_numpy().flatten() - 1  # Transform it to {0, 1}
    y = df["default payment"].to_numpy().flatten()

    del df["SEX"];del df["default payment"]

    # Remove other "meaningless" columns and create final data set
    # [MARRIAGE, AGE]
    del df["MARRIAGE"];del df["AGE"]

    X = df.to_numpy()

    return X, y, y_sensitive


# https://archive.ics.uci.edu/ml/datasets/Communities+and+Crime
def load_communitiesAndCrime_dataset():
    # Load data
    df = pd.read_csv("data/communities_crime.csv")

    # Extract label and sensitive attribute
    y_sensitive = df["Black"].to_numpy().flatten()
    y = df["class"].to_numpy().flatten()

    del df["Black"];del df["class"]

    # Remove other "meaningless" columns and create final data set
    # [state, communityname, fold]
    del df["state"];del df["communityname"];del df["fold"]

    X = df.to_numpy()

    return X, y, y_sensitive

    # Return final dataset
    return X, y, y_sensitive
