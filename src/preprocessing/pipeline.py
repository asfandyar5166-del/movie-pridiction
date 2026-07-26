import pandas as pd
import numpy as np
import re
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler, MinMaxScaler
from typing import Tuple, Optional, List, Dict


class NetflixPreprocessor:
    def __init__(self, data_path: str):
        self.data_path = Path(data_path)
        self.df = None
        self.label_encoders = {}

    def load_data(self) -> pd.DataFrame:
        self.df = pd.read_csv(self.data_path)
        return self.df

    def clean_text(self, text: str) -> str:
        if pd.isna(text) or text == "Not Given":
            return ""
        text = str(text).lower().strip()
        text = re.sub(r"[^a-zA-Z0-9\s]", " ", text)
        text = re.sub(r"\s+", " ", text)
        return text.strip()

    def parse_duration(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        duration_min = []
        duration_seasons = []
        for val in df["duration"]:
            val = str(val).strip().lower()
            if "season" in val:
                num = re.search(r"(\d+)", val)
                seasons = int(num.group(1)) if num else 0
                duration_min.append(np.nan)
                duration_seasons.append(seasons)
            else:
                num = re.search(r"(\d+)", val)
                mins = int(num.group(1)) if num else 0
                duration_min.append(mins)
                duration_seasons.append(0)
        df["duration_min"] = duration_min
        df["duration_seasons"] = duration_seasons
        return df

    def parse_date_added(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        df["date_added"] = pd.to_datetime(df["date_added"], errors="coerce")
        df["year_added"] = df["date_added"].dt.year
        df["month_added"] = df["date_added"].dt.month
        df["day_added"] = df["date_added"].dt.day
        df["days_since_added"] = (pd.Timestamp.now() - df["date_added"]).dt.days
        return df

    def extract_countries(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        df["country_list"] = df["country"].apply(
            lambda x: [c.strip() for c in str(x).split(",")] if str(x) != "Not Given" else ["Unknown"]
        )
        df["primary_country"] = df["country_list"].apply(lambda x: x[0] if x else "Unknown")
        df["num_countries"] = df["country_list"].apply(len)
        return df

    def extract_genres(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        df["genre_list"] = df["listed_in"].apply(
            lambda x: [g.strip() for g in str(x).split(",")]
        )
        df["primary_genre"] = df["genre_list"].apply(lambda x: x[0] if x else "Unknown")
        df["num_genres"] = df["genre_list"].apply(len)
        return df

    def extract_directors(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        df["director_list"] = df["director"].apply(
            lambda x: [d.strip() for d in str(x).split(",")] if str(x) != "Not Given" else []
        )
        df["num_directors"] = df["director_list"].apply(len)
        df["has_director"] = (df["director"] != "Not Given").astype(int)
        return df

    def encode_ratings(self, df: pd.DataFrame) -> pd.DataFrame:
        rating_order = {
            "TV-Y": 0, "TV-Y7": 1, "TV-Y7-FV": 1, "G": 2, "PG": 3,
            "TV-G": 2, "TV-PG": 3, "PG-13": 4, "TV-14": 5,
            "R": 6, "NC-17": 7, "TV-MA": 7, "NR": 8, "UR": 8
        }
        df = df.copy()
        df["rating_ordered"] = df["rating"].map(rating_order).fillna(8)
        df["rating_category"] = df["rating"].apply(self._categorize_rating)
        return df

    @staticmethod
    def _categorize_rating(rating: str) -> str:
        kids = {"TV-Y", "TV-Y7", "TV-Y7-FV", "G", "TV-G"}
        teens = {"PG", "TV-PG", "PG-13", "TV-14"}
        adults = {"R", "NC-17", "TV-MA"}
        if rating in kids:
            return "Kids/Family"
        elif rating in teens:
            return "Teens/Young Adults"
        elif rating in adults:
            return "Adults"
        else:
            return "Other"

    def create_text_features(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        df["clean_title"] = df["title"].apply(self.clean_text)
        df["clean_director"] = df["director"].apply(self.clean_text)
        df["clean_country"] = df["country"].apply(self.clean_text)
        df["clean_genres"] = df["listed_in"].apply(self.clean_text)
        df["clean_description"] = df.get("description", pd.Series([""] * len(df))).apply(self.clean_text)
        df["combined_features"] = (
            df["clean_title"] + " " +
            df["clean_director"] + " " +
            df["clean_country"] + " " +
            df["clean_genres"]
        )
        df["combined_features_light"] = (
            df["clean_genres"] + " " +
            df["clean_title"]
        )
        return df

    def remove_duplicates(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        df = df.drop_duplicates(subset=["title", "type", "release_year"], keep="first")
        return df

    def full_pipeline(self) -> pd.DataFrame:
        df = self.load_data()
        df = self.remove_duplicates(df)
        df = self.parse_duration(df)
        df = self.parse_date_added(df)
        df = self.extract_countries(df)
        df = self.extract_genres(df)
        df = self.extract_directors(df)
        df = self.encode_ratings(df)
        df = self.create_text_features(df)
        self.df = df
        return df

    def get_feature_sets(self) -> Dict[str, List[str]]:
        return {
            "numerical": [
                "release_year", "duration_min", "duration_seasons",
                "num_countries", "num_genres", "num_directors",
                "year_added", "month_added", "days_since_added"
            ],
            "categorical": [
                "type", "primary_country", "primary_genre",
                "rating_category"
            ],
            "text": ["combined_features", "combined_features_light"],
            "id_features": ["show_id", "title"],
        }

    def prepare_classification_data(
        self, target_col: str, test_size: float = 0.2, val_size: float = 0.1,
        random_state: int = 42
    ) -> Tuple:
        df = self.df.copy()
        feature_sets = self.get_feature_sets()
        num_cols = [c for c in feature_sets["numerical"] if c in df.columns]
        df_num = df[num_cols].fillna(0)
        cat_cols = [c for c in feature_sets["categorical"] if c in df.columns and c != target_col]
        df_cat = pd.get_dummies(df[cat_cols], drop_first=True)
        X = pd.concat([df_num.reset_index(drop=True), df_cat.reset_index(drop=True)], axis=1)
        y = df[target_col].values
        X_temp, X_test, y_temp, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state, stratify=y
        )
        val_ratio = val_size / (1 - test_size)
        X_train, X_val, y_train, y_val = train_test_split(
            X_temp, y_temp, test_size=val_ratio, random_state=random_state, stratify=y_temp
        )
        return X_train, X_val, X_test, y_train, y_val, y_test
