"""
Stage 5: Evaluation.

evaluate_model() computes threshold-free metrics (PR-AUC, ROC-AUC) plus
threshold-dependent metrics at a chosen operating point, since the business
decision (block / flag / allow) happens at a threshold, not on the raw score.

compare_to_champion() is the champion/challenger check: same test set, same
metrics, a plain delta -- the thing you'd actually show before promoting a
retrained model to production.
"""

import json

from sklearn.metrics import (
    average_precision_score, roc_auc_score, precision_score,
    recall_score, f1_score, confusion_matrix,
)


def evaluate_model(model, X_test, y_test, threshold: float = 0.5) -> dict:
    y_scores = model.predict_proba(X_test)[:, 1]
    y_pred = (y_scores >= threshold).astype(int)

    tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()

    return {
        "threshold": threshold,
        "pr_auc": round(average_precision_score(y_test, y_scores), 4),
        "roc_auc": round(roc_auc_score(y_test, y_scores), 4),
        "precision": round(precision_score(y_test, y_pred, zero_division=0), 4),
        "recall": round(recall_score(y_test, y_pred, zero_division=0), 4),
        "f1": round(f1_score(y_test, y_pred, zero_division=0), 4),
        "true_positives": int(tp),
        "false_positives": int(fp),
        "false_negatives": int(fn),
        "true_negatives": int(tn),
    }


def compare_to_champion(challenger_metrics: dict, champion_metrics: dict | None) -> dict:
    """Returns deltas of challenger vs champion. If no champion exists yet
    (first run), the challenger is trivially promoted."""
    if champion_metrics is None:
        return {"promote": True, "reason": "no existing champion", "deltas": None}

    keys = ["pr_auc", "roc_auc", "precision", "recall", "f1"]
    deltas = {k: round(challenger_metrics[k] - champion_metrics[k], 4) for k in keys}

    # Simple promotion rule: challenger must not regress PR-AUC (primary metric)
    # and must not regress recall by more than a small tolerance.
    promote = deltas["pr_auc"] >= 0 and deltas["recall"] >= -0.02

    return {"promote": promote, "reason": "pr_auc/recall check", "deltas": deltas}


def save_metrics(metrics: dict, path: str):
    with open(path, "w") as f:
        json.dump(metrics, f, indent=2)


def load_metrics(path: str) -> dict | None:
    try:
        with open(path) as f:
            return json.load(f)
    except FileNotFoundError:
        return None
