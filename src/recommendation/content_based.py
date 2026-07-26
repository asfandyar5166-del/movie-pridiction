import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from pathlib import Path
import sys
import warnings

sys.path.append(str(Path(__file__).resolve().parent.parent.parent))
from config.project_config import RECOMMENDATION_K

warnings.filterwarnings("ignore")


class ContentBasedRecommender:
    def __init__(self, df: pd.DataFrame):
        self.df = df.copy().reset_index(drop=True)
        self.similarity_matrix = None
        self.vectorizer = None
        self.feature_matrix = None
        self.method_name = ""

    def fit_tfidf(self, text_column: str = "combined_features", max_features: int = 5000):
        self.method_name = f"TF-IDF ({text_column})"
        self.vectorizer = TfidfVectorizer(
            max_features=max_features, stop_words="english",
            ngram_range=(1, 2), sublinear_tf=True
        )
        texts = self.df[text_column].fillna("").astype(str).tolist()
        self.feature_matrix = self.vectorizer.fit_transform(texts)
        self.similarity_matrix = cosine_similarity(self.feature_matrix)
        return self

    def fit_countvectorizer(self, text_column: str = "combined_features", max_features: int = 5000):
        self.method_name = f"CountVectorizer ({text_column})"
        self.vectorizer = CountVectorizer(
            max_features=max_features, stop_words="english",
            ngram_range=(1, 2)
        )
        texts = self.df[text_column].fillna("").astype(str).tolist()
        self.feature_matrix = self.vectorizer.fit_transform(texts)
        self.similarity_matrix = cosine_similarity(self.feature_matrix)
        return self

    def recommend(self, title: str, k: int = RECOMMENDATION_K) -> pd.DataFrame:
        if self.similarity_matrix is None:
            raise ValueError("Model not fitted. Call fit_tfidf or fit_countvectorizer first.")
        title_idx = self.df.index[self.df["title"].str.lower() == title.lower()].tolist()
        if not title_idx:
            title_idx = self.df.index[self.df["title"].str.contains(title, case=False, na=False)].tolist()
        if not title_idx:
            return pd.DataFrame(columns=["title", "type", "similarity_score", "genres", "rank"])
        idx = title_idx[0]
        sim_scores = list(enumerate(self.similarity_matrix[idx]))
        sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
        sim_scores = [s for s in sim_scores if s[0] != idx]
        sim_scores = sim_scores[:k]
        indices = [s[0] for s in sim_scores]
        scores = [s[1] for s in sim_scores]
        results = self.df.iloc[indices][["title", "type", "listed_in"]].copy()
        results["similarity_score"] = scores
        results["rank"] = range(1, len(scores) + 1)
        results.rename(columns={"listed_in": "genres"}, inplace=True)
        return results

    def evaluate_relevance(self, title: str, k: int = RECOMMENDATION_K) -> dict:
        recs = self.recommend(title, k=k)
        if recs.empty:
            return {"method": self.method_name, "query": title, "num_recommendations": 0}
        query_genres = set()
        query_row = self.df[self.df["title"].str.lower() == title.lower()]
        if not query_row.empty:
            genres = query_row.iloc[0].get("listed_in", "")
            query_genres = set(g.strip().lower() for g in str(genres).split(","))
        relevant = 0
        shared_genres = []
        for _, row in recs.iterrows():
            rec_genres = set(g.strip().lower() for g in str(row["genres"]).split(","))
            if query_genres and rec_genres:
                overlap = query_genres & rec_genres
                if overlap:
                    relevant += 1
                    shared_genres.append(overlap)
        precision = relevant / k if k > 0 else 0
        return {
            "method": self.method_name,
            "query": title,
            "num_recommendations": len(recs),
            "precision_at_k": precision,
            "same_type_ratio": recs["type"].value_counts().to_dict() if "type" in recs.columns else {},
            "query_genres": list(query_genres),
        }

    def evaluate_multiple_queries(self, titles: list, k: int = RECOMMENDATION_K) -> pd.DataFrame:
        results = []
        for title in titles:
            result = self.evaluate_relevance(title, k=k)
            results.append(result)
        return pd.DataFrame(results)


def run_recommendation_pipeline(df: pd.DataFrame):
    print("\n" + "=" * 60)
    print("TASK 1: CONTENT-BASED RECOMMENDATION SYSTEM")
    print("=" * 60)
    rec = ContentBasedRecommender(df)
    rec.fit_tfidf("combined_features", max_features=5000)
    query_titles = ["Stranger Things", "The Crown", "Black Mirror", "Narcos", "3 Idiots"]
    for title in query_titles:
        if title.lower() in df["title"].str.lower().values:
            print(f"\n--- Recommendations for '{title}' ---")
            recs = rec.recommend(title, k=10)
            if not recs.empty:
                print(recs[["rank", "title", "type", "genres", "similarity_score"]].to_string(index=False))
    eval_results = rec.evaluate_multiple_queries(query_titles, k=10)
    print("\n--- Evaluation Results ---")
    print(eval_results.to_string(index=False))
    save_json(eval_results.to_dict(orient="records"), Path(__file__).resolve().parent.parent.parent / "reports" / "recommendation_evaluation.json")
    return rec
