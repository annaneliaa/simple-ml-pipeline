"""
Stage 3: Model.

build_pipeline() wires the preprocessor from features.py to a classifier
into a single sklearn Pipeline, so tuning/evaluation can treat it as one
object. XGBoost is the default; scale_pos_weight compensates for class
imbalance (ratio of negatives to positives) instead of resampling.
"""

from sklearn.pipeline import Pipeline
from xgboost import XGBClassifier

from features import build_preprocessor


def build_pipeline(scale_pos_weight: float = 1.0, **xgb_params) -> Pipeline:
    default_params = dict(
        n_estimators=300,
        max_depth=4,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        eval_metric="aucpr",
        scale_pos_weight=scale_pos_weight,
        random_state=42,
        n_jobs=-1,
    )
    default_params.update(xgb_params)

    return Pipeline([
        ("preprocess", build_preprocessor()),
        ("clf", XGBClassifier(**default_params)),
    ])


def compute_scale_pos_weight(y) -> float:
    n_pos = (y == 1).sum()
    n_neg = (y == 0).sum()
    return n_neg / max(n_pos, 1)
