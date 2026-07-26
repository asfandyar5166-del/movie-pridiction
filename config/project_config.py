from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
MODELS_DIR = PROJECT_ROOT / "models"
REPORTS_DIR = PROJECT_ROOT / "reports"
VIZ_DIR = PROJECT_ROOT / "visualizations"
TESTS_DIR = PROJECT_ROOT / "tests"

DATA_FILE = PROJECT_ROOT / "Dataset.csv"

RAW_DATA_FILE = DATA_DIR / "raw" / "Dataset.csv"
PROCESSED_DATA_FILE = DATA_DIR / "processed" / "netflix_processed.csv"

RANDOM_STATE = 42
TEST_SIZE = 0.2
VAL_SIZE = 0.1
CV_FOLDS = 5

RECOMMENDATION_K = 10
CLASSIFICATION_TARGET = "type"
RATING_TARGET = "rating"
CLUSTERING_N_CLUSTERS = 5
FORECAST_HORIZON = 12
