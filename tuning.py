"""
Stage 4: Parameter tuning.

RandomizedSearchCV over the XGBoost params inside the pipeline, scored on
average precision (= PR-AUC), which is the right optimization target for a
rare positive class -- ROC-AUC is easy to look good on and easy to be
misled by here.

Swap in Optuna later if you want smarter (Bayesian) search; the param
distributions below are deliberately simple to keep this stage runnable
without extra infra.
"""

from scipy.stats import randint, uniform
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold

from model import build_pipeline, compute_scale_pos_weight

PARAM_DISTRIBUTIONS = {
    "clf__n_estimators": randint(150, 500),
    "clf__max_depth": randint(3, 7),
    "clf__learning_rate": uniform(0.01, 0.19),
    "clf__subsample": uniform(0.6, 0.4),
    "clf__colsample_bytree": uniform(0.6, 0.4),
    "clf__min_child_weight": randint(1, 8),
}


def tune_model(X_train, y_train, n_iter: int = 20, cv_folds: int = 3, random_state: int = 42):
    spw = compute_scale_pos_weight(y_train)
    pipeline = build_pipeline(scale_pos_weight=spw)

    cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=random_state)
    search = RandomizedSearchCV(
        estimator=pipeline,
        param_distributions=PARAM_DISTRIBUTIONS,
        n_iter=n_iter,
        scoring="average_precision",
        cv=cv,
        random_state=random_state,
        n_jobs=-1,
        verbose=1,
    )
    search.fit(X_train, y_train)
    return search.best_estimator_, search.best_params_, search.best_score_
