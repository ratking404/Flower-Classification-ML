from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from .config import RANDOM_STATE, TEST_SIZE


def create_train_test_data(X, y):
    """Split the data into training data and testing data."""
    return train_test_split(
        X,
        y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y,
    )


def train_model(X_train, y_train):
    """Train a simple Decision Tree model."""
    model = DecisionTreeClassifier(max_depth=3, random_state=RANDOM_STATE)
    model.fit(X_train, y_train)
    return model
