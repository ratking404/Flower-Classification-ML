from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_PATH = BASE_DIR / "data" / "iris_flowers.csv"
MODEL_PATH = BASE_DIR / "models" / "decision_tree_iris_model.joblib"
CHARTS_DIR = BASE_DIR / "outputs" / "charts"
REPORTS_DIR = BASE_DIR / "outputs" / "reports"

FEATURE_COLUMNS = [
    "sepal_length_cm",
    "sepal_width_cm",
    "petal_length_cm",
    "petal_width_cm",
]

TARGET_COLUMN = "species"
RANDOM_STATE = 42
TEST_SIZE = 0.2
