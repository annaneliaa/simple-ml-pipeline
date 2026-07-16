"""
Stage 1: Data ingestion.

generate_synthetic_transactions() is a stand-in for a real feed. It produces
transaction-level rows with a realistic imbalance and a handful of raw fields
that plausibly correlate with fraud (large amount, odd hour, new account,
foreign merchant, high recent velocity) plus pure noise fields, so a model
actually has to learn signal rather than memorize an obvious rule.

SWAP POINT: replace this function's body with a real loader, e.g.
    return pd.read_csv(path)
    return pd.read_sql(query, connection)
as long as it returns a DataFrame containing the columns referenced in
features.py plus a TARGET_COL column of 0/1 labels.
"""

import numpy as np
import pandas as pd

from config import N_SAMPLES, FRAUD_RATE, RANDOM_SEED, TARGET_COL


def generate_synthetic_transactions(n_samples: int = N_SAMPLES, fraud_rate: float = FRAUD_RATE,
                                     seed: int = RANDOM_SEED) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    n_fraud = int(n_samples * fraud_rate)
    n_legit = n_samples - n_fraud

    def make_block(n, fraud: bool):
        if fraud:
            amount = rng.lognormal(mean=5.2, sigma=1.1, size=n)
            hour = rng.choice(range(24), size=n, p=_night_weighted_hours())
            account_age_days = rng.exponential(scale=60, size=n).clip(0, 4000)
            n_txn_last_hour = rng.poisson(lam=4.0, size=n)
            is_foreign_merchant = rng.binomial(1, 0.45, size=n)
            distance_from_home_km = rng.exponential(scale=800, size=n)
        else:
            amount = rng.lognormal(mean=3.6, sigma=0.9, size=n)
            hour = rng.integers(0, 24, size=n)
            account_age_days = rng.exponential(scale=500, size=n).clip(0, 4000)
            n_txn_last_hour = rng.poisson(lam=0.6, size=n)
            is_foreign_merchant = rng.binomial(1, 0.05, size=n)
            distance_from_home_km = rng.exponential(scale=15, size=n)

        return pd.DataFrame({
            "amount": amount,
            "hour_of_day": hour,
            "account_age_days": account_age_days,
            "n_txn_last_hour": n_txn_last_hour,
            "is_foreign_merchant": is_foreign_merchant,
            "distance_from_home_km": distance_from_home_km,
            # noise fields the model must learn to ignore
            "merchant_category_code": rng.integers(0, 50, size=n),
            "customer_tenure_band": rng.integers(0, 5, size=n),
            TARGET_COL: int(fraud),
        })

    df = pd.concat([make_block(n_legit, False), make_block(n_fraud, True)], ignore_index=True)
    df = df.sample(frac=1.0, random_state=seed).reset_index(drop=True)
    df.insert(0, "transaction_id", [f"txn_{i:07d}" for i in range(len(df))])
    return df


def _night_weighted_hours():
    # fraud skews toward late-night/early-morning hours
    weights = np.array([3 if h in list(range(0, 6)) + [23] else 1 for h in range(24)], dtype=float)
    return weights / weights.sum()


if __name__ == "__main__":
    df = generate_synthetic_transactions()
    print(df.shape)
    print(df[TARGET_COL].value_counts(normalize=True))
    print(df.head())
