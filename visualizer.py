import pandas as pd
from .config import DATA_PATH, FEATURE_COLUMNS, TARGET_COLUMN


def load_dataset():
    """Load the flower dataset from the CSV file."""
    data = pd.read_csv(DATA_PATH)
    return data


def split_features_and_target(data):
    """Separate input features X from the answer/label y."""
    X = data[FEATURE_COLUMNS]
    y = data[TARGET_COLUMN]
    return X, y
