# Netflix Content Analysis & ML Pipeline

A production-quality machine learning project implementing a complete analytics pipeline for Netflix content data. Built as a professional portfolio piece demonstrating best practices in ML, software engineering, and data science.

## Project Structure

```
movie-pridiction/
├── Dataset.csv                    # Raw Netflix titles dataset (8,790 entries)
├── requirements.txt               # Python dependencies
├── README.md                      # This file
├── config/
│   └── project_config.py          # Centralized configuration
├── src/
│   ├── main.py                    # Pipeline entry point
│   ├── preprocessing/
│   │   └── pipeline.py            # Data preprocessing & feature engineering
│   ├── recommendation/
│   │   └── content_based.py       # Task 1: Content-Based Recommendations
│   ├── classification/
│   │   ├── type_classifier.py     # Task 2: Movie vs TV Show Classification
│   │   └── rating_classifier.py   # Task 3: Audience Rating Prediction
│   ├── clustering/
│   │   └── content_clustering.py  # Task 4: Content Clustering
│   ├── forecasting/
│   │   └── release_forecast.py    # Task 5: Release Trend Forecasting
│   └── analysis/
│       ├── eda.py                 # Exploratory Data Analysis
│       └── business_intelligence.py  # Task 6: BI & Insights
├── models/                        # Saved trained models (pickle)
├── reports/                       # JSON evaluation reports
├── visualizations/                # Generated plots (PNG)
├── data/
│   ├── raw/                       # Raw data copy
│   └── processed/                 # Processed data output
├── tests/                         # Unit tests
└── notebooks/                     # Jupyter notebooks (optional)
```

## Requirements

```bash
pip install -r requirements.txt
```

Core dependencies: numpy, pandas, scikit-learn, xgboost, lightgbm, statsmodels, matplotlib, seaborn, plotly, prophet, nltk, umap-learn

## Quick Start

```bash
python3 src/main.py
```

This runs the entire pipeline end-to-end, producing all models, visualizations, reports, and insights.

## Dataset Overview

- **Source:** Netflix titles catalog
- **Size:** 8,790 entries (6,126 Movies, 2,664 TV Shows)
- **Columns:** 10 original features (show_id, type, title, director, country, date_added, release_year, rating, duration, listed_in)
- **Engineered features:** 24 additional columns after preprocessing
- **Date range:** 2008-2021 (content added), 1925-2021 (release year)

## Tasks Implemented

### Task 1: Content-Based Recommendation System
- TF-IDF and CountVectorizer text embeddings
- Cosine similarity scoring
- Combined feature engineering (title + director + country + genres)
- Precision@K evaluation with genre-based relevance
- Query any Netflix title for similar recommendations

### Task 2: Movie vs TV Show Classification
- 7 algorithms compared: Logistic Regression, Decision Tree, Random Forest, Gradient Boosting, SVM, XGBoost, LightGBM
- 5-fold stratified cross-validation
- Hyperparameter tuning via GridSearchCV
- Best model: Logistic Regression (100% test accuracy)

### Task 3: Audience Rating Prediction
- 14 rating classes predicted
- 6 algorithms compared with cross-validation
- Best model: Logistic Regression (F1=0.797 weighted)
- Includes classification report and confusion matrix

### Task 4: Content Clustering
- 4 algorithms: K-Means, Hierarchical, GMM, DBSCAN
- Optimal K determination via Elbow + Silhouette methods
- PCA dimensionality reduction for visualization
- Cluster interpretation with genre/country/type profiles

### Task 5: Forecast Netflix Release Trends
- Auto ARIMA, Exponential Smoothing, Prophet
- MAE, RMSE, MAPE evaluation
- 12-month forecast with confidence intervals
- Best model: Exponential Smoothing (MAE=30.99)

### Task 6: End-to-End Business Intelligence System
- Genre analysis, rating distribution, release trends
- Country production analysis, director analysis
- Content maturity trends, duration analysis
- Automated business recommendations
- Executive summary generation

## Key Results Summary

| Task | Best Model | Metric | Value |
|------|-----------|--------|-------|
| Recommendation | TF-IDF | Precision@10 | 1.00 |
| Type Classification | Logistic Regression | Test Accuracy | 100% |
| Rating Prediction | Logistic Regression | Test F1 (weighted) | 0.767 |
| Clustering | K-Means | Silhouette Score | varies by K |
| Forecasting | ExpSmoothing | MAE | 30.99 |
| BI System | - | Insights | 6 recommendations |

## Output Files

- **Models:** `models/*.pkl` (trained models + scalers + encoders)
- **Reports:** `reports/*.json` (evaluation results, insights)
- **Visualizations:** `visualizations/*.png` (20+ charts)

## Reproducibility

- Fixed random state (42) throughout
- All preprocessing steps documented in pipeline
- Version-controlled with Git
- Requirements file with pinned versions

## Limitations & Future Work

- Content-based recs rely on TF-IDF (no deep embeddings like Sentence-BERT)
- Rating prediction limited by 14-class imbalance (logistic regression best, LightGBM struggles)
- Forecasting limited by only 8 years of monthly data
- No collaborative filtering implemented (no user interaction data)
- Future: add deep learning embeddings, collaborative filtering, LSTM forecasting, interactive dashboard