"""
End-to-end run: data -> preprocessing/features -> model -> tuning -> evaluation.

Usage:
    python pipeline.py

Each run trains a "challenger" model and compares it against whatever
"champion" model/metrics are currently saved in artifacts/. If the
challenger passes the promotion check, it becomes the new champion.
"""

import os
import joblib
from sklearn.model_selection import train_test_split

import config
from data import generate_synthetic_transactions
from features import engineer_features, split_X_y
from tuning import tune_model
from evaluate import evaluate_model, compare_to_champion, save_metrics, load_metrics


def main():
    os.makedirs(config.MODEL_DIR, exist_ok=True)

    # 1. Data
    print("[1/5] Loading data...")
    df = generate_synthetic_transactions()

    # 2. Preprocessing + features
    print("[2/5] Engineering features...")
    df = engineer_features(df)
    X, y = split_X_y(df)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=config.TEST_SIZE, stratify=y, random_state=config.RANDOM_SEED
    )

    # 3 & 4. Model + tuning
    print("[3-4/5] Training and tuning model (this does the CV search)...")
    best_model, best_params, best_cv_score = tune_model(X_train, y_train, n_iter=15)
    print(f"  best CV PR-AUC: {best_cv_score:.4f}")
    print(f"  best params: {best_params}")

    # 5. Evaluation
    print("[5/5] Evaluating on held-out test set...")
    challenger_metrics = evaluate_model(best_model, X_test, y_test)
    champion_metrics = load_metrics(config.CHAMPION_METRICS_PATH)
    comparison = compare_to_champion(challenger_metrics, champion_metrics)

    print("\n--- Challenger metrics ---")
    for k, v in challenger_metrics.items():
        print(f"  {k}: {v}")

    print("\n--- Champion/challenger comparison ---")
    print(f"  promote: {comparison['promote']}  ({comparison['reason']})")
    if comparison["deltas"]:
        print(f"  deltas: {comparison['deltas']}")

    # Always save the challenger artifacts
    joblib.dump(best_model, config.CHALLENGER_MODEL_PATH)
    save_metrics(challenger_metrics, config.CHALLENGER_METRICS_PATH)

    if comparison["promote"]:
        joblib.dump(best_model, config.CHAMPION_MODEL_PATH)
        save_metrics(challenger_metrics, config.CHAMPION_METRICS_PATH)
        print(f"\n-> Challenger promoted to champion. Saved to {config.CHAMPION_MODEL_PATH}")
    else:
        print(f"\n-> Challenger NOT promoted. Champion unchanged.")

    return best_model, challenger_metrics, comparison


if __name__ == "__main__":
    main()
