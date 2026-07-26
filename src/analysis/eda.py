import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from datetime import datetime
import warnings
import sys

sys.path.append(str(Path(__file__).resolve().parent.parent.parent))
from config.project_config import VIZ_DIR, REPORTS_DIR


sns.set_style("whitegrid")
plt.rcParams["figure.figsize"] = (12, 6)
warnings.filterwarnings("ignore")


class ExploratoryDataAnalysis:
    def __init__(self, df: pd.DataFrame):
        self.df = df
        self.viz_dir = Path(VIZ_DIR)
        self.viz_dir.mkdir(parents=True, exist_ok=True)

    def overview(self):
        print("=" * 60)
        print("DATASET OVERVIEW")
        print("=" * 60)
        print(f"Shape: {self.df.shape}")
        print(f"\nColumns:\n{list(self.df.columns)}")
        print(f"\nData Types:\n{self.df.dtypes}")
        print(f"\nMissing Values:\n{self.df.isnull().sum()[self.df.isnull().sum() > 0]}")
        hashable_cols = [c for c in self.df.columns if self.df[c].dtype != 'object' or self.df[c].apply(lambda x: not isinstance(x, (list, dict))).all()]
        try:
            dup_count = self.df[hashable_cols].duplicated().sum() if hashable_cols else 0
            print(f"\nDuplicate Rows (on hashable cols): {dup_count}")
        except Exception:
            print(f"\nDuplicate Rows: Could not compute (unhashable columns present)")

    def distribution_plots(self):
        fig, axes = plt.subplots(2, 3, figsize=(18, 10))
        if "type" in self.df.columns:
            self.df["type"].value_counts().plot(kind="bar", ax=axes[0, 0], color=["skyblue", "salmon"])
            axes[0, 0].set_title("Content Type Distribution")
            axes[0, 0].set_ylabel("Count")
        if "rating" in self.df.columns:
            rating_order = self.df["rating"].value_counts().index
            self.df["rating"].value_counts()[rating_order].plot(kind="bar", ax=axes[0, 1], color="lightgreen")
            axes[0, 1].set_title("Rating Distribution")
            axes[0, 1].tick_params(axis="x", rotation=45)
        if "release_year" in self.df.columns:
            self.df["release_year"].plot(kind="hist", bins=50, ax=axes[0, 2], color="purple", edgecolor="black")
            axes[0, 2].set_title("Release Year Distribution")
        if "primary_country" in self.df.columns:
            top_countries = self.df["primary_country"].value_counts().head(10)
            top_countries.plot(kind="bar", ax=axes[1, 0], color="orange")
            axes[1, 0].set_title("Top 10 Producing Countries")
            axes[1, 0].tick_params(axis="x", rotation=45)
        if "primary_genre" in self.df.columns:
            top_genres = self.df["primary_genre"].value_counts().head(10)
            top_genres.plot(kind="bar", ax=axes[1, 1], color="teal")
            axes[1, 1].set_title("Top 10 Genres")
            axes[1, 1].tick_params(axis="x", rotation=45)
        if "year_added" in self.df.columns:
            self.df["year_added"].value_counts().sort_index().plot(kind="line", marker="o", ax=axes[1, 2], color="red")
            axes[1, 2].set_title("Content Added Over Time")
            axes[1, 2].set_ylabel("Count")
        plt.tight_layout()
        plt.savefig(self.viz_dir / "distribution_plots.png", dpi=150, bbox_inches="tight")
        plt.close()

    def rating_by_type(self):
        if "type" not in self.df.columns or "rating" not in self.df.columns:
            return
        fig, ax = plt.subplots(figsize=(14, 6))
        ctab = pd.crosstab(self.df["rating"], self.df["type"])
        ctab.plot(kind="bar", ax=ax, color=["skyblue", "salmon"])
        ax.set_title("Rating Distribution by Content Type")
        ax.set_ylabel("Count")
        ax.legend(title="Type")
        plt.tight_layout()
        plt.savefig(self.viz_dir / "rating_by_type.png", dpi=150, bbox_inches="tight")
        plt.close()

    def genre_trends(self):
        if "primary_genre" not in self.df.columns or "release_year" not in self.df.columns:
            return
        top_genres = self.df["primary_genre"].value_counts().head(5).index
        genre_year = self.df[self.df["primary_genre"].isin(top_genres)]
        genre_year_pivot = genre_year.pivot_table(
            index="release_year", columns="primary_genre", aggfunc="size", fill_value=0
        )
        fig, ax = plt.subplots(figsize=(14, 6))
        genre_year_pivot.plot(ax=ax, linewidth=2)
        ax.set_title("Top 5 Genre Trends Over Years")
        ax.set_ylabel("Count")
        ax.legend(title="Genre")
        plt.tight_layout()
        plt.savefig(self.viz_dir / "genre_trends.png", dpi=150, bbox_inches="tight")
        plt.close()

    def duration_analysis(self):
        if "duration_min" not in self.df.columns and "duration_seasons" not in self.df.columns:
            return
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        if "duration_min" in self.df.columns:
            movie_durs = self.df[self.df["type"] == "Movie"]["duration_min"].dropna()
            movie_durs.plot(kind="hist", bins=50, ax=axes[0], color="skyblue", edgecolor="black")
            axes[0].set_title("Movie Duration Distribution (minutes)")
            axes[0].set_xlabel("Duration (min)")
        if "duration_seasons" in self.df.columns:
            tv_seasons = self.df[self.df["type"] == "TV Show"]["duration_seasons"].dropna()
            tv_seasons.value_counts().sort_index().plot(kind="bar", ax=axes[1], color="salmon")
            axes[1].set_title("TV Show Seasons Distribution")
            axes[1].set_xlabel("Seasons")
        plt.tight_layout()
        plt.savefig(self.viz_dir / "duration_analysis.png", dpi=150, bbox_inches="tight")
        plt.close()

    def correlation_heatmap(self):
        num_cols = self.df.select_dtypes(include=[np.number]).columns
        if len(num_cols) < 2:
            return
        fig, ax = plt.subplots(figsize=(12, 10))
        corr = self.df[num_cols].corr()
        mask = np.triu(np.ones_like(corr, dtype=bool))
        sns.heatmap(corr, mask=mask, annot=True, fmt=".2f", cmap="RdBu_r",
                    center=0, square=True, ax=ax)
        ax.set_title("Feature Correlation Matrix")
        plt.tight_layout()
        plt.savefig(self.viz_dir / "correlation_heatmap.png", dpi=150, bbox_inches="tight")
        plt.close()

    def country_production_heatmap(self):
        if "primary_country" not in self.df.columns or "type" not in self.df.columns:
            return
        top_countries = self.df["primary_country"].value_counts().head(15).index
        country_type = self.df[self.df["primary_country"].isin(top_countries)]
        ctab = pd.crosstab(country_type["primary_country"], country_type["type"])
        fig, ax = plt.subplots(figsize=(12, 8))
        sns.heatmap(ctab, annot=True, fmt="d", cmap="YlOrRd", ax=ax)
        ax.set_title("Content Production by Country and Type")
        plt.tight_layout()
        plt.savefig(self.viz_dir / "country_production_heatmap.png", dpi=150, bbox_inches="tight")
        plt.close()

    def run_all(self):
        self.overview()
        self.distribution_plots()
        self.rating_by_type()
        self.genre_trends()
        self.duration_analysis()
        self.correlation_heatmap()
        self.country_production_heatmap()
        print(f"\nAll visualizations saved to {self.viz_dir}")
