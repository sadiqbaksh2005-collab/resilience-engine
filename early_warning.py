"""
Bank view: who to call this month.

Uses the early warning model exactly as validated (logistic regression, no
standing order snapshot feature, trained on t <= 2023-09, threshold frozen on
validation at 50% row recall), so the test period numbers describe this list.
Scores every account at its latest month (end of 2025).

Outputs: outputs/ew_model.pkl   (pipeline, features, threshold, test metrics)
         outputs/bank_view.csv  (one row per account, ranked)
"""
from pathlib import Path
import pickle
import numpy as np
import pandas as pd

from features import FEATURES, MIN_TENURE
from baselines import make_lr, threshold_for_recall, event_metrics

OUT = Path(__file__).parent / "outputs"
FEATS = [f for f in FEATURES if f != "so_burden"]

NICE = {
    "min_balance": "low minimum balance", "avg_daily_balance": "low average balance",
    "month_end_balance": "low month end balance", "money_in": "income level", "money_out": "spending level",
    "net_flow": "money out exceeds money in", "out_in_ratio": "spending high relative to income",
    "cash_wd_share": "spending mostly cash", "fixed_burden": "high fixed costs", "pension_share": "pension income",
    "transfer_in_share": "income by transfer", "net_flow_r3": "3 month cash outflow", "net_flow_r6": "6 month cash outflow",
    "bal_std_r3": "volatile balance (3 mo)", "bal_std_r6": "volatile balance (6 mo)", "min_bal_r3": "low balance in last 3 months",
    "min_bal_r6": "low balance in last 6 months", "bal_chg_3m": "balance falling", "money_in_cv_r6": "irregular income",
    "months_cover": "few months of spending covered", "low_bal_count_r6": "repeated low balance months",
    "z_min_balance": "balance below own normal", "z_net_flow": "cash flow below own normal",
    "z_money_out": "spending above own normal", "tenure": "new account", "has_active_loan": "has a running loan",
    "loan_payment_ratio": "loan payment heavy vs income", "has_card": "has a card",
    "freq_weekly": "weekly statements", "freq_each_tx": "statement after each transaction",
}


def fit_validated_model():
    p = pd.read_pickle(OUT / "panel.pkl")
    tr, va, te = (p[p.split == s].copy() for s in ["train", "val", "test"])
    m = make_lr().fit(tr[FEATS], tr["y"])
    va["s"], te["s"] = m.predict_proba(va[FEATS])[:, 1], m.predict_proba(te[FEATS])[:, 1]
    thr = threshold_for_recall(va["y"].to_numpy(), va["s"].to_numpy())
    flag = te["s"] >= thr
    ev = event_metrics(te, "s", thr)
    metrics = dict(threshold=float(thr), test_alert_rate=float(flag[te.y == 0].mean()),
                   test_precision=float(te.loc[flag, "y"].mean()), test_event_recall=ev["event_recall"],
                   test_median_lead=ev["median_lead"], test_base_rate=float(te.y.mean()))
    return m, thr, metrics


def contributions(m, X):
    """Per row contribution of each feature to the log odds (coef x standardized value)."""
    Z = X
    for name, step in m.steps[:-1]:
        Z = step.transform(Z)
    return Z * m.named_steps["lr"].coef_[0]


def main():
    m, thr, metrics = fit_validated_model()
    full = pd.read_pickle(OUT / "monthly_full.pkl").sort_values(["account_id", "t"])
    full["ever_neg"] = (full["min_balance"] < 0).groupby(full["account_id"]).cummax()
    last = full.groupby("account_id").tail(1).copy()
    last = last[last["tenure"] >= MIN_TENURE]
    last["risk"] = m.predict_proba(last[FEATS])[:, 1]
    last["flag"] = last["risk"] >= thr

    C = contributions(m, last[FEATS])
    top = np.argsort(-C, axis=1)[:, :2]
    last["why"] = [", ".join(NICE.get(FEATS[j], FEATS[j]) for j in row if C[i, j] > 0) or "no single driver"
                   for i, row in enumerate(top)]

    first_neg = full.loc[full["min_balance"] < 0].groupby("account_id")["month"].min()
    last["past_overdraft"] = last["account_id"].map(first_neg)
    pop = pd.read_csv(OUT / "population_accounts.csv")[
        ["account_id", "runway_base", "runway_ei55", "runway_ei_realistic", "buffer_months", "age_band", "loan"]]
    bv = (last[["account_id", "month", "risk", "flag", "why", "past_overdraft"]]
          .merge(pop, on="account_id", how="left")
          .sort_values("risk", ascending=False).reset_index(drop=True))
    bv["rank"] = np.arange(1, len(bv) + 1)
    bv.to_csv(OUT / "bank_view.csv", index=False)
    last[["account_id", "month"] + FEATS].to_pickle(OUT / "latest_features.pkl")   # small file the app reads
    with open(OUT / "ew_model.pkl", "wb") as f:
        pickle.dump(dict(model=m, features=FEATS, threshold=thr, metrics=metrics, nice=NICE), f)

    fresh = bv[bv.past_overdraft.isna()]
    print("validated performance at this threshold (test 2024-10..2025-09):",
          {k: round(v, 4) for k, v in metrics.items()})
    print(f"scored {len(bv):,} accounts at their latest month; flagged {bv.flag.sum()} "
          f"({fresh.flag.sum()} with no past overdraft)")
    blind = fresh[fresh.flag & (fresh.runway_base >= 12)]
    print(f"flagged, never overdrawn, yet judges' runway >= 12 months: {len(blind)} "
          f"({len(blind) / max(fresh.flag.sum(), 1):.0%} of fresh flags)  <- blind spot in action")
    print("\ntop 12 (no past overdraft):")
    print(fresh.head(12)[["rank", "account_id", "risk", "why", "runway_base", "runway_ei55", "age_band", "loan"]]
          .round(3).to_string(index=False))


if __name__ == "__main__":
    main()
