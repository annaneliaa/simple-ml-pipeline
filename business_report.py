"""
Business translation layer, sitting on top of Stage 5 (evaluate.py).

The core idea: precision/recall are not decisions. The threshold is. This
module sweeps the classification threshold and converts each point into
euro terms, so a non-technical stakeholder can see what "moving the
threshold" actually costs and saves -- not just how a metric moves.

Assumptions (COST_PER_INVESTIGATION, INVESTIGATOR_MINUTES_PER_CASE) are
placeholders clearly called out as such in the generated report -- swap
them for real bunq unit-economics numbers when available.

Usage:
    python business_report.py
"""

import joblib
import numpy as np
import pandas as pd

import config

COST_PER_INVESTIGATION = 12.0   # EUR: investigator time per flagged case (~15 min @ ~€48/hr loaded cost)
DEFAULT_THRESHOLD = 0.5


def compute_cost_curve(model, X_test, y_test, thresholds=None) -> pd.DataFrame:
    if thresholds is None:
        thresholds = np.round(np.arange(0.05, 0.96, 0.05), 2)

    y_scores = model.predict_proba(X_test)[:, 1]
    amounts = X_test["amount"].to_numpy()
    y_true = y_test.to_numpy()

    rows = []
    for t in thresholds:
        y_pred = (y_scores >= t).astype(int)
        tp = (y_pred == 1) & (y_true == 1)
        fp = (y_pred == 1) & (y_true == 0)
        fn = (y_pred == 0) & (y_true == 1)

        fraud_caught_eur = amounts[tp].sum()
        fraud_missed_eur = amounts[fn].sum()
        n_flagged = int(tp.sum() + fp.sum())
        investigation_cost_eur = n_flagged * COST_PER_INVESTIGATION
        net_value_eur = fraud_caught_eur - investigation_cost_eur

        precision = tp.sum() / max(tp.sum() + fp.sum(), 1)
        recall = tp.sum() / max(tp.sum() + fn.sum(), 1)

        rows.append({
            "threshold": t,
            "precision": round(precision, 4),
            "recall": round(recall, 4),
            "n_flagged": n_flagged,
            "true_positives": int(tp.sum()),
            "false_positives": int(fp.sum()),
            "false_negatives": int(fn.sum()),
            "fraud_caught_eur": round(fraud_caught_eur, 2),
            "fraud_missed_eur": round(fraud_missed_eur, 2),
            "investigation_cost_eur": round(investigation_cost_eur, 2),
            "net_value_eur": round(net_value_eur, 2),
        })

    return pd.DataFrame(rows)


def find_optimal_threshold(cost_curve: pd.DataFrame) -> pd.Series:
    return cost_curve.loc[cost_curve["net_value_eur"].idxmax()]


def generate_report(cost_curve: pd.DataFrame, default_threshold: float = DEFAULT_THRESHOLD) -> str:
    default_row = cost_curve.iloc[(cost_curve["threshold"] - default_threshold).abs().idxmin()]
    optimal_row = find_optimal_threshold(cost_curve)

    total_fraud_eur = default_row["fraud_caught_eur"] + default_row["fraud_missed_eur"]
    uplift_eur = optimal_row["net_value_eur"] - default_row["net_value_eur"]
    investigator_hours_at_default = default_row["n_flagged"] * 15 / 60
    investigator_hours_at_optimal = optimal_row["n_flagged"] * 15 / 60

    lines = []
    lines.append("# Fraud model — business impact report\n")
    lines.append(
        "_Note: investigation cost (€12/case, ~15 min) is a placeholder assumption — "
        "replace with real bunq unit economics before using these numbers for decisions._\n"
    )
    lines.append("## Headline numbers\n")
    lines.append(f"- Total fraud value in test period: **€{total_fraud_eur:,.0f}**")
    lines.append(
        f"- At the default threshold ({default_threshold}): catches **€{default_row['fraud_caught_eur']:,.0f}** "
        f"of fraud, misses **€{default_row['fraud_missed_eur']:,.0f}**, costs **€{default_row['investigation_cost_eur']:,.0f}** "
        f"in investigator time (~{investigator_hours_at_default:.1f} hours), net value **€{default_row['net_value_eur']:,.0f}**"
    )
    lines.append(
        f"- At the profit-optimal threshold ({optimal_row['threshold']}): catches **€{optimal_row['fraud_caught_eur']:,.0f}**, "
        f"costs **€{optimal_row['investigation_cost_eur']:,.0f}** (~{investigator_hours_at_optimal:.1f} hours), "
        f"net value **€{optimal_row['net_value_eur']:,.0f}**"
    )
    lines.append(f"- Simply moving the threshold from {default_threshold} to {optimal_row['threshold']} is worth an estimated **€{uplift_eur:,.0f}** more, with no retraining required.\n")

    lines.append("## The trade-off, in plain terms\n")
    lines.append(
        "Lowering the threshold catches more fraud but flags more legitimate transactions too — "
        "each of those is a real customer (Eva) hitting a declined card or a review delay. "
        "Raising the threshold protects the customer experience but lets more fraud through. "
        "There is no single 'correct' threshold — it's a business choice about which cost the "
        "organisation is more willing to absorb, and this report exists to make that choice visible "
        "rather than buried in a model's default settings.\n"
    )

    lines.append("## Full threshold sweep\n")
    lines.append(cost_curve.to_markdown(index=False))

    return "\n".join(lines)


def main():
    model = joblib.load(config.CHAMPION_MODEL_PATH)
    split = joblib.load(f"{config.MODEL_DIR}/test_split.joblib")
    X_test, y_test = split["X_test"], split["y_test"]

    cost_curve = compute_cost_curve(model, X_test, y_test)
    cost_curve.to_csv(f"{config.MODEL_DIR}/cost_curve.csv", index=False)

    report = generate_report(cost_curve)
    with open(f"{config.MODEL_DIR}/business_report.md", "w") as f:
        f.write(report)

    print(report)


if __name__ == "__main__":
    main()
