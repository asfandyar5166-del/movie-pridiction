import numpy as np
import pandas as pd
from sklearn.cluster import KMeans, AgglomerativeClustering, DBSCAN
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
from pathlib import Path
import warnings
import sys
import matplotlib.pyplot as plt

sys.path.append(str(Path(__file__).resolve().parent.parent.parent))
from config.project_config import RANDOM_STATE, MODELS_DIR, VIZ_DIR, REPORTS_DIR
from src.utils.metrics import clustering_metrics
from src.utils.helpers import save_model, save_json

warnings.filterwarnings("ignore")


class ContentClustering:
    def __init__(self, df: pd.DataFrame):
        self.df = df.copy()
        self.scaler = StandardScaler()
        self.pca = None
        self.X_scaled = None
        self.X_pca = None
        self.models = {}
        self.results = {}

    def prepare_features(self):
        numerical_cols = [
            "release_year", "duration_min", "duration_seasons",
            "num_countries", "num_genres", "num_directors",
            "year_added", "month_added", "days_since_added"
        ]
        available = [c for c in numerical_cols if c in self.df.columns]
        X = self.df[available].fillna(0).values
        self.X_scaled = self.scaler.fit_transform(X)
        self.feature_names = available
        return self

    def reduce_dimensions(self, n_components: int = 10):
        self.pca = PCA(n_components=min(n_components, self.X_scaled.shape[1]), random_state=RANDOM_STATE)
        self.X_pca = self.pca.fit_transform(self.X_scaled)
        print(f"  PCA explained variance: {self.pca.explained_variance_ratio_.sum():.3f}")
        return self

    def find_optimal_k(self, max_k: int = 15):
        inertia = []
        silhouette_scores = []
        K_range = range(2, max_k + 1)
        for k in K_range:
            km = KMeans(n_clusters=k, random_state=RANDOM_STATE, n_init=10)
            labels = km.fit_predict(self.X_scaled)
            inertia.append(km.inertia_)
            sil = silhouette_score(self.X_scaled, labels) if len(set(labels)) > 1 else 0
            silhouette_scores.append(sil)
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        axes[0].plot(K_range, inertia, "bo-")
        axes[0].set_title("Elbow Method")
        axes[0].set_xlabel("K")
        axes[0].set_ylabel("Inertia")
        axes[1].plot(K_range, silhouette_scores, "ro-")
        axes[1].set_title("Silhouette Score")
        axes[1].set_xlabel("K")
        axes[1].set_ylabel("Score")
        plt.tight_layout()
        plt.savefig(VIZ_DIR / "optimal_k.png", dpi=150, bbox_inches="tight")
        plt.close()
        best_k = K_range[np.argmax(silhouette_scores)]
        print(f"  Optimal K (silhouette): {best_k}")
        return best_k, silhouette_scores

    def run_kmeans(self, n_clusters: int = 5):
        km = KMeans(n_clusters=n_clusters, random_state=RANDOM_STATE, n_init=10)
        labels = km.fit_predict(self.X_scaled)
        self.models["K-Means"] = km
        self.results["K-Means"] = {
            "labels": labels,
            "metrics": clustering_metrics(self.X_scaled, labels),
            "model": km,
        }
        return labels

    def run_hierarchical(self, n_clusters: int = 5):
        hc = AgglomerativeClustering(n_clusters=n_clusters)
        labels = hc.fit_predict(self.X_scaled)
        self.models["Hierarchical"] = hc
        self.results["Hierarchical"] = {
            "labels": labels,
            "metrics": clustering_metrics(self.X_scaled, labels),
            "model": hc,
        }
        return labels

    def run_dbscan(self, eps: float = 0.5, min_samples: int = 5):
        db = DBSCAN(eps=eps, min_samples=min_samples)
        labels = db.fit_predict(self.X_scaled)
        n_noise = list(labels).count(-1)
        self.models["DBSCAN"] = db
        self.results["DBSCAN"] = {
            "labels": labels,
            "metrics": clustering_metrics(self.X_scaled, labels),
            "model": db,
            "noise_points": n_noise,
        }
        print(f"  DBSCAN noise points: {n_noise}/{len(labels)}")
        return labels

    def run_gmm(self, n_components: int = 5):
        gmm = GaussianMixture(n_components=n_components, random_state=RANDOM_STATE)
        labels = gmm.fit_predict(self.X_scaled)
        self.models["GMM"] = gmm
        self.results["GMM"] = {
            "labels": labels,
            "metrics": clustering_metrics(self.X_scaled, labels),
            "model": gmm,
        }
        return labels

    def visualize_clusters(self, labels, title: str = "Clusters"):
        if self.X_pca is None:
            self.reduce_dimensions()
        fig, axes = plt.subplots(1, 2, figsize=(16, 6))
        scatter1 = axes[0].scatter(self.X_pca[:, 0], self.X_pca[:, 1], c=labels, cmap="viridis", alpha=0.6)
        axes[0].set_title(f"{title} (PCA 1-2)")
        axes[0].set_xlabel("PC1")
        axes[0].set_ylabel("PC2")
        plt.colorbar(scatter1, ax=axes[0])
        if self.X_pca.shape[1] >= 3:
            scatter2 = axes[1].scatter(self.X_pca[:, 0], self.X_pca[:, 2], c=labels, cmap="viridis", alpha=0.6)
            axes[1].set_title(f"{title} (PCA 1-3)")
            axes[1].set_xlabel("PC1")
            axes[1].set_ylabel("PC3")
            plt.colorbar(scatter2, ax=axes[1])
        plt.tight_layout()
        safe_title = title.replace(" ", "_").lower()
        plt.savefig(VIZ_DIR / f"clusters_{safe_title}.png", dpi=150, bbox_inches="tight")
        plt.close()

    def interpret_clusters(self, labels, n_clusters: int):
        df = self.df.copy()
        df["cluster"] = labels
        interpretations = {}
        for c in range(n_clusters):
            cluster_data = df[df["cluster"] == c]
            profile = {
                "size": len(cluster_data),
                "pct": round(len(cluster_data) / len(df) * 100, 1),
                "top_genres": cluster_data["primary_genre"].value_counts().head(3).to_dict() if "primary_genre" in cluster_data.columns else {},
                "top_countries": cluster_data["primary_country"].value_counts().head(3).to_dict() if "primary_country" in cluster_data.columns else {},
                "avg_release_year": round(cluster_data["release_year"].mean(), 1) if "release_year" in cluster_data.columns else None,
                "type_mix": cluster_data["type"].value_counts().to_dict() if "type" in cluster_data.columns else {},
                "avg_duration_min": round(cluster_data["duration_min"].mean(), 1) if "duration_min" in cluster_data.columns else None,
            }
            interpretations[f"Cluster_{c}"] = profile
        return interpretations

    def run_comparison(self, n_clusters: int = 5):
        print(f"\n--- Clustering with {n_clusters} clusters ---")
        self.prepare_features()
        self.run_kmeans(n_clusters)
        self.run_hierarchical(n_clusters)
        self.run_dbscan(eps=0.8, min_samples=5)
        self.run_gmm(n_clusters)
        comparison = []
        for name, result in self.results.items():
            entry = {"algorithm": name}
            entry.update(result["metrics"])
            entry["noise_points"] = result.get("noise_points", 0)
            comparison.append(entry)
        comp_df = pd.DataFrame(comparison)
        print(f"\n--- Clustering Comparison ---")
        print(comp_df.to_string(index=False))
        save_json(comp_df.to_dict(orient="records"), REPORTS_DIR / "clustering_comparison.json")
        return comp_df


def run_clustering_pipeline(df: pd.DataFrame):
    print("\n" + "=" * 60)
    print("TASK 4: CONTENT CLUSTERING")
    print("=" * 60)
    clusterer = ContentClustering(df)
    clusterer.prepare_features()
    optimal_k, sil_scores = clusterer.find_optimal_k(max_k=12)
    n_clusters = max(optimal_k, 4)
    clusterer.run_kmeans(n_clusters)
    clusterer.run_hierarchical(n_clusters)
    clusterer.run_gmm(n_clusters)
    if clusterer.X_pca is None:
        clusterer.reduce_dimensions()
    clusterer.visualize_clusters(clusterer.results["K-Means"]["labels"], "K-Means Clusters")
    clusterer.visualize_clusters(clusterer.results["Hierarchical"]["labels"], "Hierarchical Clusters")
    clusterer.visualize_clusters(clusterer.results["GMM"]["labels"], "GMM Clusters")
    interpretations = clusterer.interpret_clusters(
        clusterer.results["K-Means"]["labels"], n_clusters
    )
    print("\n--- Cluster Interpretations ---")
    for cluster, profile in interpretations.items():
        print(f"\n{cluster}: {profile['size']} items ({profile['pct']}%)")
        print(f"  Top genres: {profile['top_genres']}")
        print(f"  Type mix: {profile['type_mix']}")
        print(f"  Avg release year: {profile['avg_release_year']}")
    save_json(interpretations, REPORTS_DIR / "cluster_interpretations.json")
    return clusterer
