# Fraud detection ML pipeline (MVP scaffold)

Bare end-to-end pipeline: **data -> preprocessing/features -> model -> tuning -> evaluation**.
Run it with:

```bash
pip install xgboost scikit-learn pandas numpy joblib scipy
python pipeline.py
```

## Files (one per stage)

- `config.py` — shared constants (seed, paths, target column)
- `data.py` — **Stage 1**. `generate_synthetic_transactions()` produces a synthetic,
  realistically-imbalanced (~0.8%) transaction dataset. Swap point clearly marked
  in the docstring for plugging in a real data source (CSV, SQL, warehouse table).
- `features.py` — **Stage 2**. `engineer_features()` adds derived signals
  (amount-to-account-age ratio, night-transaction flag). `build_preprocessor()`
  returns a sklearn `ColumnTransformer` (scaling + one-hot encoding).
- `model.py` — **Stage 3**. `build_pipeline()` wires preprocessing + XGBoost into
  one sklearn `Pipeline`. Uses `scale_pos_weight` to handle class imbalance.
- `tuning.py` — **Stage 4**. `tune_model()` runs `RandomizedSearchCV` optimizing
  PR-AUC (average precision) — the right target for a rare positive class.
- `evaluate.py` — **Stage 5**. `evaluate_model()` computes PR-AUC, ROC-AUC,
  precision/recall/F1, and a confusion matrix at a given threshold.
  `compare_to_champion()` implements the champion/challenger check: compares a
  newly trained model against the currently saved one and returns a promote/
  no-promote decision with metric deltas.
- `pipeline.py` — orchestrates all five stages end to end. Each run saves a
  "challenger" model, compares it to the saved "champion", and promotes it if
  it passes the check. Artifacts land in `artifacts/`.

## Where to extend next

- Swap `data.py` for a real loader.
- Add more engineered features in `features.py` (true velocity windows over
  a rolling window, counterparty graph features, device/session signals).
- Add an anomaly-detection branch (e.g. `IsolationForest`) alongside the
  supervised model and blend the two scores.
- Add a cost-based business metric (est. $ saved vs. investigator hours) on
  top of `evaluate_model()`'s output — this is the layer that turns metrics
  into a stakeholder-facing report.
- Swap `RandomizedSearchCV` for Optuna if you want smarter search.
