import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import cross_val_score, StratifiedKFold
from pathlib import Path
import warnings
import sys
import json
import time

sys.path.append(str(Path(__file__).resolve().parent.parent.parent))
from config.project_config import RANDOM_STATE, CV_FOLDS, MODELS_DIR, REPORTS_DIR
from src.utils.metrics import classification_metrics
from src.utils.helpers import save_model, save_json

warnings.filterwarnings("ignore")

XGB_AVAILABLE = False
LGBM_AVAILABLE = False
CATB_AVAILABLE = False

try:
    from xgboost import XGBClassifier
    XGB_AVAILABLE = True
except ImportError:
    pass

try:
    from lightgbm import LGBMClassifier
    LGBM_AVAILABLE = True
except ImportError:
    pass

try:
    from catboost import CatBoostClassifier
    CATB_AVAILABLE = True
except ImportError:
    pass


def get_models():
    models = {
        "Logistic Regression": LogisticRegression(
            max_iter=2000, random_state=RANDOM_STATE, n_jobs=-1
        ),
        "Decision Tree": DecisionTreeClassifier(
            random_state=RANDOM_STATE
        ),
        "Random Forest": RandomForestClassifier(
            n_estimators=200, random_state=RANDOM_STATE, n_jobs=-1
        ),
        "Gradient Boosting": GradientBoostingClassifier(
            n_estimators=200, random_state=RANDOM_STATE
        ),
        "SVM": SVC(
            probability=True, random_state=RANDOM_STATE
        ),
    }

    if XGB_AVAILABLE:
        models["XGBoost"] = XGBClassifier(
            n_estimators=200, random_state=RANDOM_STATE, n_jobs=-1,
            use_label_encoder=False, eval_metric="logloss"
        )
    if LGBM_AVAILABLE:
        models["LightGBM"] = LGBMClassifier(
            n_estimators=200, random_state=RANDOM_STATE, n_jobs=-1, verbose=-1
        )
    if CATB_AVAILABLE:
        models["CatBoost"] = CatBoostClassifier(
            iterations=200, random_state=RANDOM_STATE, verbose=0
        )
    return models


return best_model, all_results_df

    print("\n" + "=" * 60)
    print("TASK 2: MOVIE vs TV SHOW CLASSIFICATION")
    print("=" * 60)

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    X_test_scaled = scaler.transform(X_test)

    if isinstance(y_train[0], str):
        le = LabelEncoder()
        y_train_enc = le.fit_transform(y_train)
        y_val_enc = le.transform(y_val)
        y_test_enc = le.transform(y_test)
    else:
        le = None
        y_train_enc = y_train
        y_val_enc = y_val
        y_test_enc = y_test

    models = get_models()
    results = []
    best_model = None
    best_f1 = 0
    best_name = ""
    best_y_pred = None
    best_y_proba = None

    skf = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_STATE)

    for name, model in models.items():
        print(f"\n--- Training {name} ---")
        start = time.time()

        cv_scores = cross_val_score(
            model, X_train_scaled, y_train_enc,
            cv=skf, scoring="f1_weighted"
        )

        model.fit(X_train_scaled, y_train_enc)
        y_pred = model.predict(X_val_scaled)
        y_proba = model.predict_proba(X_val_scaled) if hasattr(model, "predict_proba") else None

        val_metrics = classification_metrics(y_val_enc, y_pred, y_proba)
        train_time = time.time() - start

        y_test_pred = model.predict(X_test_scaled)
        y_test_proba = model.predict_proba(X_test_scaled) if hasattr(model, "predict_proba") else None
        test_metrics = classification_metrics(y_test_enc, y_test_pred, y_test_proba)

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

        print(f"  CV F1 (weighted): {cv_scores.mean():.4f} +/- {cv_scores.std():.4f}")
        print(f"  Val Accuracy: {val_metrics['accuracy']:.4f}, Val F1: {val_metrics['f1_weighted']:.4f}")
        print(f"  Test Accuracy: {test_metrics['accuracy']:.4f}, Test F1: {test_metrics['f1_weighted']:.4f}")

        if val_metrics["f1_weighted"] > best_f1:
            best_f1 = val_metrics["f1_weighted"]
            best_model = model
            best_name = name
            best_y_pred = y_pred
            best_y_proba = y_proba

    print(f"\n{'=' * 60}")
    print(f"BEST MODEL: {best_name} (Val F1: {best_f1:.4f})")
    print(f"{'=' * 60}")

    all_results_df = pd.DataFrame(results).sort_values("test_f1_weighted", ascending=False)
    print(f"\n--- Model Comparison ---")
    print(all_results_df.to_string(index=False))

    save_json(all_results_df.to_dict(orient="records"), REPORTS_DIR / "type_classification_results.json")
    save_model(scaler, MODELS_DIR / "type_scaler.pkl")
    save_model(best_model, MODELS_DIR / "type_best_model.pkl")
    if le:
        save_model(le, MODELS_DIR / "type_label_encoder.pkl")

return best_model, all_results_df


def train_and_evaluate_models(X_train, y_train, X_val, y_val, X_test, y_test):
