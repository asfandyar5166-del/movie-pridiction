import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import warnings
import sys

sys.path.append(str(Path(__file__).resolve().parent.parent.parent))
from config.project_config import VIZ_DIR, REPORTS_DIR
from src.utils.helpers import save_json

warnings.filterwarnings("ignore")


class BusinessIntelligence:
    def __init__(self, df: pd.DataFrame):
        self.df = df.copy()
        self.insights = {}

    def genre_analysis(self):
        if "genre_list" not in self.df.columns:
            return
        all_genres = []
        for g_list in self.df["genre_list"]:
            all_genres.extend(g_list)
        genre_counts = pd.Series(all_genres).value_counts()
        self.insights["top_genres"] = genre_counts.head(15).to_dict()
        fig, ax = plt.subplots(figsize=(12, 6))
        genre_counts.head(15).plot(kind="bar", ax=ax, color="teal")
        ax.set_title("Top 15 Genres on Netflix")
        ax.set_ylabel("Count")
        plt.xticks(rotation=45, ha="right")
        plt.tight_layout()
        plt.savefig(VIZ_DIR / "top_genres.png", dpi=150, bbox_inches="tight")
        plt.close()

    def rating_distribution_analysis(self):
        if "rating" not in self.df.columns:
            return
        rating_counts = self.df["rating"].value_counts()
        self.insights["rating_distribution"] = rating_counts.to_dict()
        fig, ax = plt.subplots(figsize=(12, 5))
        rating_counts.plot(kind="bar", ax=ax, color="coral")
        ax.set_title("Content Rating Distribution")
        ax.set_ylabel("Count")
        plt.xticks(rotation=45, ha="right")
        for i, v in enumerate(rating_counts.values):
            ax.text(i, v + 20, str(v), ha="center", fontsize=8)
        plt.tight_layout()
        plt.savefig(VIZ_DIR / "rating_distribution.png", dpi=150, bbox_inches="tight")
        plt.close()

    def release_trend_analysis(self):
        if "release_year" not in self.df.columns:
            return
        year_counts = self.df["release_year"].value_counts().sort_index()
        self.insights["release_trends"] = {
            "min_year": int(year_counts.index.min()),
            "max_year": int(year_counts.index.max()),
            "peak_year": int(year_counts.idxmax()),
            "peak_count": int(year_counts.max()),
            "avg_per_year": round(year_counts.mean(), 1),
        }
        fig, ax = plt.subplots(figsize=(14, 5))
        year_counts.plot(kind="line", marker="o", ax=ax, color="navy", linewidth=2)
        ax.set_title("Content Releases Over Time")
        ax.set_ylabel("Number of Titles")
        ax.set_xlabel("Release Year")
        ax.fill_between(year_counts.index, year_counts.values, alpha=0.2)
        plt.tight_layout()
        plt.savefig(VIZ_DIR / "release_trends.png", dpi=150, bbox_inches="tight")
        plt.close()

    def country_production_analysis(self):
        if "primary_country" not in self.df.columns:
            return
        country_counts = self.df["primary_country"].value_counts()
        self.insights["top_countries"] = country_counts.head(15).to_dict()
        fig, ax = plt.subplots(figsize=(12, 7))
        country_counts.head(15).plot(kind="barh", ax=ax, color="steelblue")
        ax.set_title("Top 15 Producing Countries")
        ax.set_xlabel("Number of Titles")
        plt.tight_layout()
        plt.savefig(VIZ_DIR / "top_countries.png", dpi=150, bbox_inches="tight")
        plt.close()

    def director_analysis(self):
        if "director" not in self.df.columns:
            return
        directors = self.df[self.df["director"] != "Not Given"]["director"]
        all_directors = []
        for d in directors:
            all_directors.extend([x.strip() for x in str(d).split(",")])
        dir_counts = pd.Series(all_directors).value_counts()
        self.insights["top_directors"] = dir_counts.head(15).to_dict()
        fig, ax = plt.subplots(figsize=(12, 7))
        dir_counts.head(15).plot(kind="barh", ax=ax, color="darkgreen")
        ax.set_title("Top 15 Directors by Number of Titles")
        ax.set_xlabel("Number of Titles")
        plt.tight_layout()
        plt.savefig(VIZ_DIR / "top_directors.png", dpi=150, bbox_inches="tight")
        plt.close()

    def type_trends(self):
        if "type" not in self.df.columns or "release_year" not in self.df.columns:
            return
        type_year = self.df.groupby(["release_year", "type"]).size().unstack(fill_value=0)
        self.insights["movie_vs_tv_ratio"] = {
            "movie_pct": round(self.df["type"].value_counts(normalize=True).get("Movie", 0) * 100, 1),
            "tv_pct": round(self.df["type"].value_counts(normalize=True).get("TV Show", 0) * 100, 1),
        }
        fig, ax = plt.subplots(figsize=(14, 5))
        type_year.plot(ax=ax, linewidth=2)
        ax.set_title("Movies vs TV Shows Over Time")
        ax.set_ylabel("Count")
        plt.tight_layout()
        plt.savefig(VIZ_DIR / "type_trends.png", dpi=150, bbox_inches="tight")
        plt.close()

    def content_maturity_trend(self):
        if "rating_category" not in self.df.columns or "release_year" not in self.df.columns:
            return
        cat_year = self.df.groupby(["release_year", "rating_category"]).size().unstack(fill_value=0)
        self.insights["maturity_trends"] = cat_year.mean().to_dict()
        fig, ax = plt.subplots(figsize=(14, 5))
        cat_year.plot(ax=ax, linewidth=2)
        ax.set_title("Content Maturity Trends Over Time")
        ax.set_ylabel("Count")
        plt.tight_layout()
        plt.savefig(VIZ_DIR / "maturity_trends.png", dpi=150, bbox_inches="tight")
        plt.close()

    def duration_vs_rating(self):
        if "duration_min" not in self.df.columns or "rating" not in self.df.columns:
            return
        fig, ax = plt.subplots(figsize=(14, 6))
        movie_df = self.df[self.df["type"] == "Movie"].dropna(subset=["duration_min"])
        top_ratings = movie_df["rating"].value_counts().head(8).index
        data = movie_df[movie_df["rating"].isin(top_ratings)]
        sns.boxplot(data=data, x="rating", y="duration_min", ax=ax, palette="Set2")
        ax.set_title("Movie Duration by Rating Category")
        ax.set_ylabel("Duration (minutes)")
        plt.tight_layout()
        plt.savefig(VIZ_DIR / "duration_vs_rating.png", dpi=150, bbox_inches="tight")
        plt.close()

    def generate_business_recommendations(self):
        recommendations = []

        if "primary_genre" in self.df.columns:
            top_genre = self.df["primary_genre"].value_counts().index[0]
            recommendations.append(f"Focus on {top_genre} content - it has the highest volume on the platform.")

        if "type" in self.df.columns:
            movie_pct = self.df["type"].value_counts(normalize=True).get("Movie", 0) * 100
            tv_pct = self.df["type"].value_counts(normalize=True).get("TV Show", 0) * 100
            if movie_pct > tv_pct:
                recommendations.append(f"Movies dominate ({movie_pct:.0f}% of catalog). Consider increasing TV Show investments to balance content library.")
            else:
                recommendations.append(f"TV Shows dominate ({tv_pct:.0f}% of catalog). Consider increasing Movie investments.")

        if "rating_category" in self.df.columns:
            top_maturity = self.df["rating_category"].value_counts().index[0]
            recommendations.append(f"'{top_maturity}' content is most prevalent. Ensure content acquisition targets this demographic.")

        if "primary_country" in self.df.columns:
            non_us = self.df[self.df["primary_country"] != "United States"]["primary_country"].value_counts()
            if len(non_us) > 0:
                top_non_us = non_us.index[0]
                recommendations.append(f"International expansion opportunity: {top_non_us} is a strong content producer outside the US.")

        if "release_year" in self.df.columns:
            recent = self.df[self.df["release_year"] >= self.df["release_year"].quantile(0.9)]
            recent_tv = (recent["type"] == "TV Show").mean() * 100
            recommendations.append(f"Recent trend: {recent_tv:.0f}% of new content is TV Shows, indicating a shift toward TV series production.")

        if "num_directors" in self.df.columns:
            multi_dir = (self.df["num_directors"] > 1).mean() * 100
            recommendations.append(f"{multi_dir:.0f}% of titles have multiple directors. Collaborative productions are common.")

        self.insights["business_recommendations"] = recommendations
        return recommendations

    def generate_executive_summary(self):
        summary = {
            "total_titles": len(self.df),
            "unique_genres": self.df["genre_list"].explode().nunique() if "genre_list" in self.df.columns else None,
            "unique_countries": self.df["primary_country"].nunique() if "primary_country" in self.df.columns else None,
            "total_directors": self.df[self.df["director"] != "Not Given"]["director"].nunique() if "director" in self.df.columns else None,
            "content_mix": self.df["type"].value_counts(normalize=True).to_dict() if "type" in self.df.columns else {},
            "most_common_rating": self.df["rating"].mode().iloc[0] if "rating" in self.df.columns else None,
            "most_common_genre": self.df["primary_genre"].mode().iloc[0] if "primary_genre" in self.df.columns else None,
            "top_producing_country": self.df["primary_country"].value_counts().index[0] if "primary_country" in self.df.columns else None,
            "avg_movie_duration": round(self.df[self.df["type"] == "Movie"]["duration_min"].mean(), 1) if "duration_min" in self.df.columns else None,
            "avg_tv_seasons": round(self.df[self.df["type"] == "TV Show"]["duration_seasons"].mean(), 1) if "duration_seasons" in self.df.columns else None,
            "year_range": f"{int(self.df['release_year'].min())}-{int(self.df['release_year'].max())}" if "release_year" in self.df.columns else None,
        }
        self.insights["executive_summary"] = summary
        return summary

    def run_all(self):
        print("\n" + "=" * 60)
        print("TASK 6: BUSINESS INTELLIGENCE SYSTEM")
        print("=" * 60)
        print("\n--- Generating Business Insights ---")
        self.genre_analysis()
        self.rating_distribution_analysis()
        self.release_trend_analysis()
        self.country_production_analysis()
        self.director_analysis()
        self.type_trends()
        self.content_maturity_trend()
        self.duration_vs_rating()
        self.generate_executive_summary()
        recommendations = self.generate_business_recommendations()
        print("\n--- Executive Summary ---")
        for k, v in self.insights.get("executive_summary", {}).items():
            print(f"  {k}: {v}")
        print("\n--- Business Recommendations ---")
        for i, rec in enumerate(recommendations, 1):
            print(f"  {i}. {rec}")
        save_json(self.insights, REPORTS_DIR / "business_insights.json")
        return self.insights
