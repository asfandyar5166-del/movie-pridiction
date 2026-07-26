import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import cross_val_score, StratifiedKFold, GridSearchCV
from sklearn.metrics import classification_report, confusion_matrix
from pathlib import Path
import warnings
import sys
import time

sys.path.append(str(Path(__file__).resolve().parent.parent.parent))
from config.project_config import RANDOM_STATE, CV_FOLDS, MODELS_DIR, REPORTS_DIR
from src.utils.metrics import classification_metrics
from src.utils.helpers import save_model, save_json

warnings.filterwarnings("ignore")

XGB_AVAILABLE = False
try:
    from xgboost import XGBClassifier
    XGB_AVAILABLE = True
except ImportError:
    pass

LGBM_AVAILABLE = False
try:
    from lightgbm import LGBMClassifier
    LGBM_AVAILABLE = True
except ImportError:
    pass


def get_rating_models():
    models = {
        "Logistic Regression": LogisticRegression(
            max_iter=2000, random_state=RANDOM_STATE, n_jobs=-1
        ),
        "Decision Tree": DecisionTreeClassifier(random_state=RANDOM_STATE),
        "Random Forest": RandomForestClassifier(n_estimators=200, random_state=RANDOM_STATE, n_jobs=-1),
        "Gradient Boosting": GradientBoostingClassifier(n_estimators=200, random_state=RANDOM_STATE),
    }
    if XGB_AVAILABLE:
        models["XGBoost"] = XGBClassifier(
            n_estimators=200, random_state=RANDOM_STATE, n_jobs=-1,
            use_label_encoder=False, eval_metric="mlogloss"
        )
    if LGBM_AVAILABLE:
        models["LightGBM"] = LGBMClassifier(
            n_estimators=200, random_state=RANDOM_STATE, n_jobs=-1, verbose=-1
        )
    return models


def train_rating_classifiers(X_train, y_train, X_val, y_val, X_test, y_test):
    print("\n" + "=" * 60)
    print("TASK 3: AUDIENCE RATING PREDICTION")
    print("=" * 60)

    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_val_s = scaler.transform(X_val)
    X_test_s = scaler.transform(X_test)

    le = LabelEncoder()
    y_train_enc = le.fit_transform(y_train)
    y_val_enc = le.transform(y_val)
    y_test_enc = le.transform(y_test)

    rating_class_counts = pd.Series(y_train_enc).value_counts().to_dict()
    print(f"\nRating classes in training: {len(rating_class_counts)}")

    models = get_rating_models()
    results = []
    best_model = None
    best_f1 = 0
    best_name = ""

    skf = StratifiedKFold(n_splits=min(CV_FOLDS, 3), shuffle=True, random_state=RANDOM_STATE)

    for name, model in models.items():
        print(f"\n--- Training {name} ---")
        start = time.time()
        try:
            cv_scores = cross_val_score(
                model, X_train_s, y_train_enc,
                cv=skf, scoring="f1_weighted"
            )
            model.fit(X_train_s, y_train_enc)
            y_pred = model.predict(X_val_s)
            y_proba = model.predict_proba(X_val_s) if hasattr(model, "predict_proba") else None
            val_metrics = classification_metrics(y_val_enc, y_pred, y_proba)
            train_time = time.time() - start
            y_test_pred = model.predict(X_test_s)
            test_metrics = classification_metrics(y_test_enc, y_test_pred, None)

            result = {
                "model": name,
                "cv_f1_mean": cv_scores.mean(),
                "cv_f1_std": cv_scores.std(),
                "val_accuracy": val_metrics["accuracy"],
                "val_f1_weighted": val_metrics["f1_weighted"],
                "test_accuracy": test_metrics["accuracy"],
                "test_f1_weighted": test_metrics["f1_weighted"],
                "train_time_seconds": round(train_time, 3),
            }
            results.append(result)
            print(f"  CV F1: {cv_scores.mean():.4f} +/- {cv_scores.std():.4f}")
            print(f"  Val Acc: {val_metrics['accuracy']:.4f}, F1: {val_metrics['f1_weighted']:.4f}")

            if val_metrics["f1_weighted"] > best_f1:
                best_f1 = val_metrics["f1_weighted"]
                best_model = model
                best_name = name
                best_y_pred = y_pred
        except Exception as e:
            print(f"  Error training {name}: {e}")
            continue

    print(f"\nBEST: {best_name} (F1={best_f1:.4f})")

    if best_model:
        param_grid = {
            "Random Forest": {"n_estimators": [100, 200], "max_depth": [10, 20, None]},
            "Gradient Boosting": {"n_estimators": [100, 200], "learning_rate": [0.05, 0.1]},
            "XGBoost": {"n_estimators": [100, 200], "learning_rate": [0.05, 0.1], "max_depth": [4, 6]},
        }
        if best_name in param_grid:
            print(f"\n--- Tuning {best_name} ---")
            grid = GridSearchCV(
                best_model.__class__(random_state=RANDOM_STATE),
                param_grid[best_name], cv=3, scoring="f1_weighted", n_jobs=-1
            )
            grid.fit(X_train_s, y_train_enc)
            print(f"  Best params: {grid.best_params_}")
            print(f"  Best CV F1: {grid.best_score_:.4f}")
            best_model = grid.best_estimator_
            y_test_pred = best_model.predict(X_test_s)
            print(f"\n--- Classification Report ---")
            print(classification_report(y_test_enc, y_test_pred, target_names=le.classes_))

    results_df = pd.DataFrame(results).sort_values("test_f1_weighted", ascending=False)
    print(f"\n--- Model Comparison ---")
    print(results_df.to_string(index=False))

    save_json(results_df.to_dict(orient="records"), REPORTS_DIR / "rating_classification_results.json")
    save_model(scaler, MODELS_DIR / "rating_scaler.pkl")
    save_model(best_model, MODELS_DIR / "rating_best_model.pkl")
    save_model(le, MODELS_DIR / "rating_label_encoder.pkl")

    return best_model, results_df
