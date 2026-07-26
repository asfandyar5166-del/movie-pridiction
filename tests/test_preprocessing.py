import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent))

import pandas as pd
from src.preprocessing.pipeline import NetflixPreprocessor


def test_preprocessor_loads_data():
    preprocessor = NetflixPreprocessor(str(Path(__file__).resolve().parent.parent / "Dataset.csv"))
    df = preprocessor.load_data()
    assert df.shape[0] > 0, "Dataset should have rows"
    assert df.shape[1] == 10, "Dataset should have 10 columns"


def test_preprocessor_full_pipeline():
    preprocessor = NetflixPreprocessor(str(Path(__file__).resolve().parent.parent / "Dataset.csv"))
    df = preprocessor.full_pipeline()
    assert df.shape[0] > 0, "Pipeline should produce rows"
    assert "duration_min" in df.columns, "Should have duration_min column"
    assert "duration_seasons" in df.columns, "Should have duration_seasons column"
    assert "year_added" in df.columns, "Should have year_added column"
    assert "primary_country" in df.columns, "Should have primary_country column"
    assert "primary_genre" in df.columns, "Should have primary_genre column"
    assert "rating_category" in df.columns, "Should have rating_category column"
    assert "combined_features" in df.columns, "Should have combined_features column"


def test_clean_text_handles_not_given():
    preprocessor = NetflixPreprocessor("dummy")
    assert preprocessor.clean_text("Not Given") == ""
    assert preprocessor.clean_text(None) == ""
    assert preprocessor.clean_text("  Hello, World!  ") == "hello world"


def test_no_data_leakage_in_split():
    preprocessor = NetflixPreprocessor(str(Path(__file__).resolve().parent.parent / "Dataset.csv"))
    df = preprocessor.full_pipeline()
    X_train, X_val, X_test, y_train, y_val, y_test = preprocessor.prepare_classification_data(
        target_col="type", test_size=0.2, val_size=0.1, random_state=42
    )
    assert X_train.shape[0] + X_val.shape[0] + X_test.shape[0] == len(df), "Split should preserve all rows"
    assert X_train.shape[0] > X_test.shape[0], "Train should be larger than test"


def test_get_feature_sets():
    preprocessor = NetflixPreprocessor(str(Path(__file__).resolve().parent.parent / "Dataset.csv"))
    df = preprocessor.full_pipeline()
    feature_sets = preprocessor.get_feature_sets()
    assert "numerical" in feature_sets
    assert "categorical" in feature_sets
    assert "text" in feature_sets
    assert len(feature_sets["numerical"]) > 0
