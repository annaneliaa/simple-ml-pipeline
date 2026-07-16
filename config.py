"""Shared configuration for the fraud detection pipeline."""

RANDOM_SEED = 42
TEST_SIZE = 0.2

# Synthetic data generation
N_SAMPLES = 50_000
FRAUD_RATE = 0.008  # ~0.8% positive class, realistic order of magnitude for fraud

# Where trained artifacts get written
MODEL_DIR = "artifacts"
CHAMPION_MODEL_PATH = f"{MODEL_DIR}/champion_model.joblib"
CHAMPION_METRICS_PATH = f"{MODEL_DIR}/champion_metrics.json"
CHALLENGER_MODEL_PATH = f"{MODEL_DIR}/challenger_model.joblib"
CHALLENGER_METRICS_PATH = f"{MODEL_DIR}/challenger_metrics.json"

TARGET_COL = "is_fraud"
