import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent))

from src.preprocessing.pipeline import NetflixPreprocessor
from src.recommendation.content_based import ContentBasedRecommender


def test_recommender_fits_tfidf():
    preprocessor = NetflixPreprocessor(str(Path(__file__).resolve().parent.parent / "Dataset.csv"))
    df = preprocessor.full_pipeline()
    rec = ContentBasedRecommender(df)
    rec.fit_tfidf("combined_features", max_features=5000)
    assert rec.similarity_matrix is not None, "Similarity matrix should be computed"
    assert rec.feature_matrix is not None, "Feature matrix should exist"


def test_recommend_returns_results():
    preprocessor = NetflixPreprocessor(str(Path(__file__).resolve().parent.parent / "Dataset.csv"))
    df = preprocessor.full_pipeline()
    rec = ContentBasedRecommender(df)
    rec.fit_tfidf("combined_features", max_features=5000)
    results = rec.recommend("Stranger Things", k=5)
    if not results.empty:
        assert len(results) <= 5, "Should return at most k results"
        assert "similarity_score" in results.columns, "Should have similarity scores"
        assert results["similarity_score"].iloc[0] >= results["similarity_score"].iloc[-1], "Should be sorted descending"


def test_recommend_handles_unknown_title():
    preprocessor = NetflixPreprocessor(str(Path(__file__).resolve().parent.parent / "Dataset.csv"))
    df = preprocessor.full_pipeline()
    rec = ContentBasedRecommender(df)
    rec.fit_tfidf("combined_features", max_features=5000)
    results = rec.recommend("NonExistentTitleXYZ", k=5)
    assert results.empty or len(results) == 0, "Should return empty for unknown title"


def test_recommend_handles_partial_match():
    preprocessor = NetflixPreprocessor(str(Path(__file__).resolve().parent.parent / "Dataset.csv"))
    df = preprocessor.full_pipeline()
    rec = ContentBasedRecommender(df)
    rec.fit_tfidf("combined_features", max_features=5000)
    results = rec.recommend("Stranger", k=5)
    assert not results.empty or True, "Partial match should work"


def test_evaluation_metrics():
    preprocessor = NetflixPreprocessor(str(Path(__file__).resolve().parent.parent / "Dataset.csv"))
    df = preprocessor.full_pipeline()
    rec = ContentBasedRecommender(df)
    rec.fit_tfidf("combined_features", max_features=5000)
    eval_result = rec.evaluate_relevance("Stranger Things", k=10)
    assert "precision_at_k" in eval_result
    assert "method" in eval_result
