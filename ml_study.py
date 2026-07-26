"""
A transparent walk-forward machine-learning study over the full indicator
universe (~35-40 signals from indicators.py / strategy_engine.py).

This is deliberately NOT trying to produce one polished confidence number.
It trains a 3-model ensemble (Random Forest + Gradient Boosting +
Logistic Regression, soft-voted) across multiple walk-forward folds --
each fold trains only on data BEFORE the test window, like live trading
would -- and reports per-fold accuracy/precision/recall/AUC plus which
features were consistently important across folds (stability, not just
one fold's importance ranking, which can be noise).

Honest framing baked in: a coin flip on balanced classes scores ~50%
accuracy. Anything consistently in the mid-to-high 50s, holding up across
folds, is a real (small) edge in this domain -- there is no realistic
version of this study that should ever report 90%+ accuracy on daily
market direction. If it did, that would indicate a bug (e.g. label
leakage), not a breakthrough.
"""
from __future__ import annotations
import numpy as np
import pandas as pd
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, roc_auc_score

import config


def run_walk_forward_study(feat_df: pd.DataFrame, feature_columns: list[str], n_folds: int = config.ML_STUDY_FOLDS) -> dict:
    data = feat_df.dropna(subset=feature_columns + ["label_up"]).copy()
    if len(data) < config.ML_MIN_ROWS_FOR_STUDY:
        return {
            "ran": False,
            "reason": f"Only {len(data)} clean rows available; need at least {config.ML_MIN_ROWS_FOR_STUDY} "
                      "for a walk-forward study to mean anything.",
            "fold_results": [], "feature_stability": pd.DataFrame(), "aggregate": {},
        }

    X = data[feature_columns].values
    y = data["label_up"].values
    baseline_accuracy = max(y.mean(), 1 - y.mean())  # majority-class baseline -- the bar to beat

    tscv = TimeSeriesSplit(n_splits=n_folds)
    fold_results = []
    importances_per_fold = []

    for fold_i, (train_idx, test_idx) in enumerate(tscv.split(X)):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]
        if len(np.unique(y_train)) < 2 or len(X_test) < 5:
            continue

        scaler = StandardScaler().fit(X_train)
        X_train_s, X_test_s = scaler.transform(X_train), scaler.transform(X_test)

        rf = RandomForestClassifier(n_estimators=200, max_depth=5, random_state=config.RANDOM_STATE, n_jobs=-1)
        gb = GradientBoostingClassifier(n_estimators=100, max_depth=3, random_state=config.RANDOM_STATE)
        lr = LogisticRegression(max_iter=1000)
        ensemble = VotingClassifier(estimators=[("rf", rf), ("gb", gb), ("lr", lr)], voting="soft")
        ensemble.fit(X_train_s, y_train)

        preds = ensemble.predict(X_test_s)
        probs = ensemble.predict_proba(X_test_s)[:, 1]

        acc = accuracy_score(y_test, preds)
        prec = precision_score(y_test, preds, zero_division=0)
        rec = recall_score(y_test, preds, zero_division=0)
        auc = roc_auc_score(y_test, probs) if len(np.unique(y_test)) > 1 else float("nan")

        fold_results.append({
            "fold": fold_i + 1, "n_train": len(train_idx), "n_test": len(test_idx),
            "accuracy": round(acc, 3), "precision": round(prec, 3), "recall": round(rec, 3),
            "auc": round(auc, 3) if not np.isnan(auc) else None,
        })
        importances_per_fold.append(pd.Series(ensemble.named_estimators_["rf"].feature_importances_, index=feature_columns))

    if not fold_results:
        return {
            "ran": False, "reason": "Folds were too small or single-class to train on.",
            "fold_results": [], "feature_stability": pd.DataFrame(), "aggregate": {},
        }

    accs = [f["accuracy"] for f in fold_results]
    aucs = [f["auc"] for f in fold_results if f["auc"] is not None]

    imp_df = pd.DataFrame(importances_per_fold)
    stability = pd.DataFrame({
        "mean_importance": imp_df.mean(),
        "std_importance": imp_df.std(),
        "folds_in_top_10": (imp_df.rank(axis=1, ascending=False) <= 10).sum(),
    }).sort_values("mean_importance", ascending=False)

    aggregate = {
        "n_folds_run": len(fold_results),
        "n_features_tested": len(feature_columns),
        "baseline_accuracy": round(baseline_accuracy, 3),
        "mean_accuracy": round(float(np.mean(accs)), 3),
        "std_accuracy": round(float(np.std(accs)), 3),
        "mean_auc": round(float(np.mean(aucs)), 3) if aucs else None,
        "edge_over_baseline": round(float(np.mean(accs)) - baseline_accuracy, 3),
        "consistent_edge": bool(np.mean(accs) > baseline_accuracy and min(accs) > baseline_accuracy - 0.03),
    }

    return {"ran": True, "fold_results": fold_results, "feature_stability": stability, "aggregate": aggregate}
