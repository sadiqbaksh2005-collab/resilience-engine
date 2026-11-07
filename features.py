"""
Build the account x month feature panel, label, risk set and time split.

Timing convention: a row (account i, month t) is a prediction made at the END
of month t. Features use months <= t only. Label looks at months t+1..t+3.

Output: outputs/panel.pkl
"""
from pathlib import Path
import numpy as np
import pandas as pd

from engine import DATA_DIR  # one configurable data location
OUT_DIR = Path(__file__).parent / "outputs"
HORIZON = 3                      # months ahead in the label
MIN_TENURE = 7                   # 6 month burn in: first months of a new account are
                                 # deposit only "quiet" months (99% of month 1, 57% of month 3)
TRAIN_END = pd.Period("2023-09", "M")   # last train t (labels end 2023-12)
VAL_END = pd.Period("2024-09", "M")     # last val t (labels end 2024-12)
LAST_T = pd.Period("2025-09", "M")      # last t with a full 3 month label window

PENALTY = "penalty interest (negative balance)"

# Features the models may use. Demographics are deliberately NOT here:
# they are reserved for the fairness audit.
FEATURES = [
    # levels (end of month t)
    "min_balance", "avg_daily_balance", "month_end_balance",
    # flows
    "money_in", "money_out", "net_flow", "out_in_ratio",
    "cash_wd_share", "fixed_burden", "pension_share", "transfer_in_share",
    # rolling dynamics (past only, window ends at t)
    "net_flow_r3", "net_flow_r6", "bal_std_r3", "bal_std_r6",
    "min_bal_r3", "min_bal_r6", "bal_chg_3m", "money_in_cv_r6",
    "months_cover", "low_bal_count_r6",
    # relative to the account's own past (expanding, strictly before t)
    "z_min_balance", "z_net_flow", "z_money_out",
    # structural
    "tenure", "so_burden", "has_active_loan", "loan_payment_ratio", "has_card",
    "freq_weekly", "freq_each_tx",
]


def load():
    r = lambda n, **k: pd.read_csv(DATA_DIR / f"{n}.csv", low_memory=False, **k)
    return {n: r(n) for n in ["monthly_summary", "transactions", "accounts",
                              "standing_orders", "loans", "dispositions", "cards"]}


def build_panel(d):
    ms = d["monthly_summary"].copy()
    ms["t"] = pd.PeriodIndex(ms["month"], freq="M")
    ms = ms.sort_values(["account_id", "t"]).reset_index(drop=True)

    # ---- remove penalty interest from bank_fees (it is caused by the event) ----
    tx = d["transactions"]
    pen = (tx[tx["purpose"].eq(PENALTY)]
           .assign(month=lambda x: x["date"].str[:7])
           .groupby(["account_id", "month"])["amount"].sum().rename("penalty"))
    ms = ms.merge(pen, on=["account_id", "month"], how="left")
    ms["penalty"] = ms["penalty"].fillna(0.0)
    ms["statement_fees"] = ms["bank_fees"] - ms["penalty"]
    ms = ms.drop(columns=["bank_fees"])

    g = ms.groupby("account_id", sort=False)
    eps = 1.0

    # ---- flow features ----
    ms["net_flow"] = ms["money_in"] - ms["money_out"]
    ms["out_in_ratio"] = ms["money_out"] / (ms["money_in"] + eps)
    ms["cash_wd_share"] = ms["cash_withdrawals_genuine"] / (ms["money_out"] + eps)
    ms["fixed_burden"] = (ms["household_bills"] + ms["insurance"] + ms["loan_payments"]) / (ms["money_in"] + eps)
    ms["pension_share"] = ms["pension_in"] / (ms["money_in"] + eps)
    ms["transfer_in_share"] = ms["transfers_in"] / (ms["money_in"] + eps)

    # ---- rolling features (window ends at t, so only past + current) ----
    def roll(col, w, fn, mp=None):
        return g[col].transform(lambda s: getattr(s.rolling(w, min_periods=mp or w), fn)())

    ms["net_flow_r3"] = roll("net_flow", 3, "mean")
    ms["net_flow_r6"] = roll("net_flow", 6, "mean", 3)
    ms["bal_std_r3"] = roll("month_end_balance", 3, "std")
    ms["bal_std_r6"] = roll("month_end_balance", 6, "std", 3)
    ms["min_bal_r3"] = roll("min_balance", 3, "min")
    ms["min_bal_r6"] = roll("min_balance", 6, "min", 3)
    ms["bal_chg_3m"] = ms["month_end_balance"] - g["month_end_balance"].shift(3)
    ms["money_in_cv_r6"] = roll("money_in", 6, "std", 3) / (roll("money_in", 6, "mean", 3) + eps)
    ms["months_cover"] = ms["avg_daily_balance"] / (roll("money_out", 3, "mean", 1) + eps)

    # low balance count: months in last 6 where min_balance < 10% of mean money_in so far
    inc_ref = g["money_in"].transform(lambda s: s.expanding().mean())
    ms["_low"] = (ms["min_balance"] < 0.1 * inc_ref).astype(float)
    ms["low_bal_count_r6"] = roll("_low", 6, "sum", 1)

    # ---- z scores vs own past (expanding stats strictly before t) ----
    for col in ["min_balance", "net_flow", "money_out"]:
        mu = g[col].transform(lambda s: s.expanding(min_periods=3).mean().shift(1))
        sd = g[col].transform(lambda s: s.expanding(min_periods=3).std().shift(1))
        ms[f"z_{col}"] = ((ms[col] - mu) / (sd + eps)).clip(-10, 10)

    # ---- structural ----
    ms["tenure"] = g.cumcount() + 1

    # standing orders have no dates; treated as a static monthly commitment.
    # ASSUMPTION (flag in pitch): orders are in force for the whole history.
    so_amt = d["standing_orders"].groupby("account_id")["amount"].sum()
    ms["so_total"] = ms["account_id"].map(so_amt).fillna(0.0)
    ms["so_burden"] = ms["so_total"] / (roll("money_in", 6, "mean", 1) + eps)

    # loans: only start date, amount, payment. Status/defaulted are OUTCOMES: never use.
    ln = d["loans"][["account_id", "loan_start", "duration_months", "monthly_payment"]].copy()
    ln["start"] = pd.PeriodIndex(ln["loan_start"].str[:7], freq="M")
    ms = ms.merge(ln[["account_id", "start", "duration_months", "monthly_payment"]], on="account_id", how="left")
    months_since = np.where(ms["start"].notna(),
                            (ms["t"].dt.year - ms["start"].dt.year) * 12 + (ms["t"].dt.month - ms["start"].dt.month),
                            -1)
    active = (months_since >= 0) & (months_since < ms["duration_months"].fillna(0))
    ms["has_active_loan"] = active.astype(int)
    ms["loan_payment_ratio"] = np.where(active, ms["monthly_payment"] / (roll("money_in", 6, "mean", 1) + eps), 0.0)

    # cards: issued date known, so only count it from issue month onward
    cd = d["cards"].merge(d["dispositions"][["disposition_id", "account_id"]], on="disposition_id")
    cd["issued_t"] = pd.PeriodIndex(cd["issued"].str[:7], freq="M")
    first_card = cd.groupby("account_id")["issued_t"].min()
    fc = ms["account_id"].map(first_card)
    ms["has_card"] = (fc.notna() & (ms["t"] >= fc.fillna(ms["t"]))).astype(int)

    freq = ms["account_id"].map(d["accounts"].set_index("account_id")["frequency"])
    ms["freq_weekly"] = freq.eq("weekly statement").astype(int)
    ms["freq_each_tx"] = freq.eq("statement after each transaction").astype(int)
    return ms


def add_label_and_split(ms):
    g = ms.groupby("account_id", sort=False)
    ms["neg"] = (ms["min_balance"] < 0).astype(int)
    fut = pd.concat([g["neg"].shift(-h) for h in range(1, HORIZON + 1)], axis=1)
    ms["y"] = fut.max(axis=1, skipna=False)           # NaN unless all 3 future months exist
    ms["already_neg"] = g["neg"].cummax()              # event at or before t

    # first event month per account (for event level metrics)
    first_neg = ms.loc[ms["neg"] == 1].groupby("account_id")["t"].min()
    ms["event_t"] = ms["account_id"].map(first_neg)

    risk = (ms["already_neg"] == 0) & ms["y"].notna() & (ms["tenure"] >= MIN_TENURE) & (ms["t"] <= LAST_T)
    p = ms.loc[risk].copy()
    p["y"] = p["y"].astype(int)
    p["split"] = np.select([p["t"] <= TRAIN_END, p["t"] <= VAL_END], ["train", "val"], "test")
    return p


def sanity(p):
    # leakage guards
    assert (p["penalty"] == 0).mean() > 0.99, "penalty interest present in risk set rows"
    assert p.groupby("account_id")["t"].apply(lambda s: s.is_monotonic_increasing).all()
    print("risk set rows by split:")
    print(p.groupby("split").agg(rows=("y", "size"), pos=("y", "sum"), rate=("y", "mean"),
                                 accounts=("account_id", "nunique")).loc[["train", "val", "test"]])
    print(f"penalty>0 rows in risk set: {(p['penalty'] > 0).sum()}")
    print(f"feature NaN share (top):\n{p[FEATURES].isna().mean().sort_values(ascending=False).head(6)}")


if __name__ == "__main__":
    OUT_DIR.mkdir(exist_ok=True)
    d = load()
    ms = build_panel(d)
    panel = add_label_and_split(ms)
    sanity(panel)
    keep = ["account_id", "t", "split", "y", "event_t"] + FEATURES
    panel[keep].to_pickle(OUT_DIR / "panel.pkl")
    ms.to_pickle(OUT_DIR / "monthly_full.pkl")   # full history incl. post event months (for HMM / charts)
    print(f"saved {len(panel):,} rows -> {OUT_DIR / 'panel.pkl'}")
