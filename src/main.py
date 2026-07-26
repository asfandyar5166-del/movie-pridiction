#!/usr/bin/env python3
"""Netflix Content Analysis & ML Pipeline - Main Entry Point"""

import sys
import warnings
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))
from config.project_config import DATA_FILE, REPORTS_DIR
from src.preprocessing.pipeline import NetflixPreprocessor
from src.analysis.eda import ExploratoryDataAnalysis
from src.recommendation.content_based import ContentBasedRecommender, run_recommendation_pipeline
from src.classification.type_classifier import train_and_evaluate_models
from src.classification.rating_classifier import train_rating_classifiers
from src.clustering.content_clustering import run_clustering_pipeline
from src.forecasting.release_forecast import run_forecast_pipeline
from src.analysis.business_intelligence import BusinessIntelligence
from src.utils.helpers import ensure_dir, save_dataframe

warnings.filterwarnings("ignore")


def main():
    print("=" * 60)
    print("NETFLIX CONTENT ANALYSIS & ML PIPELINE")
    print("=" * 60)

    ensure_dir(REPORTS_DIR)

    print("\n[STEP 1] Loading and preprocessing data...")
    preprocessor = NetflixPreprocessor(str(DATA_FILE))
    df = preprocessor.full_pipeline()
    print(f"  Dataset shape after preprocessing: {df.shape}")

    print("\n[STEP 2] Exploratory Data Analysis...")
    eda = ExploratoryDataAnalysis(df)
    eda.run_all()

    print("\n[STEP 3] Task 1: Content-Based Recommendation...")
    recommender = run_recommendation_pipeline(df)

    print("\n[STEP 4] Task 2: Movie vs TV Show Classification...")
    X_train, X_val, X_test, y_train, y_val, y_test = preprocessor.prepare_classification_data(
        target_col="type", test_size=0.2, val_size=0.1
    )
    print(f"  Train: {X_train.shape}, Val: {X_val.shape}, Test: {X_test.shape}")
    type_model, type_results = train_and_evaluate_models(
        X_train, y_train, X_val, y_val, X_test, y_test
    )

    print("\n[STEP 5] Task 3: Audience Rating Prediction...")
    X_train_r, X_val_r, X_test_r, y_train_r, y_val_r, y_test_r = preprocessor.prepare_classification_data(
        target_col="rating", test_size=0.2, val_size=0.1
    )
    print(f"  Train: {X_train_r.shape}, Val: {X_val_r.shape}, Test: {X_test_r.shape}")
    rating_model, rating_results = train_rating_classifiers(
        X_train_r, y_train_r, X_val_r, y_val_r, X_test_r, y_test_r
    )

    print("\n[STEP 6] Task 4: Content Clustering...")
    clusterer = run_clustering_pipeline(df)

    print("\n[STEP 7] Task 5: Forecast Netflix Release Trends...")
    forecast_results, forecasts, ts_data = run_forecast_pipeline(df)

    print("\n[STEP 8] Task 6: Business Intelligence System...")
    bi = BusinessIntelligence(df)
    insights = bi.run_all()

    print("\n" + "=" * 60)
    print("PIPELINE COMPLETE")
    print("=" * 60)
    print(f"\nAll outputs saved to:")
    print(f"  Reports:     reports/")
    print(f"  Models:      models/")
    print(f"  Visualizations: visualizations/")
    print(f"  Data:        data/")


if __name__ == "__main__":
    main()
