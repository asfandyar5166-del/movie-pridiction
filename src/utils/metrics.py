from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report, roc_auc_score,
    mean_absolute_error, mean_squared_error, r2_score, silhouette_score,
    davies_bouldin_score, calinski_harabasz_score,
    precision_recall_fscore_support
)
import numpy as np
import pandas as pd


def classification_metrics(y_true, y_pred, y_proba=None):
    result = {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision_macro": precision_score(y_true, y_pred, average="macro", zero_division=0),
        "recall_macro": recall_score(y_true, y_pred, average="macro", zero_division=0),
        "f1_macro": f1_score(y_true, y_pred, average="macro", zero_division=0),
        "precision_weighted": precision_score(y_true, y_pred, average="weighted", zero_division=0),
        "recall_weighted": recall_score(y_true, y_pred, average="weighted", zero_division=0),
        "f1_weighted": f1_score(y_true, y_pred, average="weighted", zero_division=0),
    }
    if y_proba is not None and len(np.unique(y_true)) == 2:
        try:
            result["roc_auc"] = roc_auc_score(y_true, y_proba[:, 1])
        except Exception:
            result["roc_auc"] = None
    return result


def regression_metrics(y_true, y_pred):
    return {
        "mae": mean_absolute_error(y_true, y_pred),
        "mse": mean_squared_error(y_true, y_pred),
        "rmse": np.sqrt(mean_squared_error(y_true, y_pred)),
        "r2": r2_score(y_true, y_pred),
    }


def clustering_metrics(X, labels):
    if len(set(labels)) < 2:
        return {"silhouette_score": None, "davies_bouldin_score": None, "calinski_harabasz_score": None}
    return {
        "silhouette_score": silhouette_score(X, labels),
        "davies_bouldin_score": davies_bouldin_score(X, labels),
        "calinski_harabasz_score": calinski_harabasz_score(X, labels),
    }


def recommendation_precision_at_k(relevant_items, recommended_items, k):
    if not recommended_items:
        return 0.0
    recommended_at_k = recommended_items[:k]
    relevant_set = set(relevant_items)
    hits = sum(1 for item in recommended_at_k if item in relevant_set)
    return hits / min(k, len(recommended_at_k))


def recommendation_recall_at_k(relevant_items, recommended_items, k):
    if not relevant_items:
        return 0.0
    recommended_at_k = recommended_items[:k]
    relevant_set = set(relevant_items)
    hits = sum(1 for item in recommended_at_k if item in relevant_set)
    return hits / len(relevant_set)
