"""
Robustness of the headline "runway misses most real overdrafts".
Varies: horizon (3, 6, 12 months), buffer definition (judges' 6 month average
vs actual month end balance), and the 'looked safe' cutoff (>= 6, >= 12, 120+).
Everything as of t (history <= t), same risk set rules and test period start.
Output: outputs/robustness.csv
"""
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

import engine as E
from features import MIN_TENURE, VAL_END
from validate_runway import asof_numbers, runway_vec

OUT = Path(__file__).parent / "outputs"
LAST = pd.Period("2025-12", "M")


def main():
    cache = OUT / "asof.pkl"
    if cache.exists():
        a = pd.read_pickle(cache)
    else:
        a = asof_numbers()
        a.to_pickle(cache)
    ms = pd.read_csv(E.DATA_DIR / "monthly_summary.csv", usecols=["account_id", "month", "min_balance", "month_end_balance"])
    ms["t"] = pd.PeriodIndex(ms["month"], freq="M")
    d = a.merge(ms[["account_id", "t", "min_balance", "month_end_balance"]], on=["account_id", "t"])
    d = d.sort_values(["account_id", "t"]).reset_index(drop=True)
    g = d.groupby("account_id")
    d["tenure"] = g.cumcount() + 1
    d["neg"] = (d["min_balance"] < 0).astype(int)
    d["already_neg"] = g["neg"].cummax()
    d["rw_judge"] = runway_vec(d["B"], d["S"], d["I"])
    d["rw_current"] = runway_vec(d["month_end_balance"], d["S"], d["I"])
    d["rw_judge_ei55"] = runway_vec(d["B"], d["S"], 0.55 * d["I"])

    rows = []
    for h in (3, 6, 12):
        fut = pd.concat([g["neg"].shift(-k) for k in range(1, h + 1)], axis=1)
        y = fut.max(axis=1, skipna=False)
        r = d.assign(y=y)
        r = r[(r.already_neg == 0) & r.y.notna() & (r.tenure >= MIN_TENURE) &
              (r.t > VAL_END) & (r.t <= LAST - h)]
        r["y"] = r["y"].astype(int)
        # earliest warning row per event = h months before the first overdraft
        first = r[r.y == 1].sort_values("t").groupby("account_id").head(1)
        for col in ["rw_judge", "rw_current", "rw_judge_ei55"]:
            rows.append(dict(horizon=h, runway=col, rows=len(r), events=len(first),
                             auc=roc_auc_score(r.y, -r[col]),
                             safe_ge6=(first[col] >= 6).mean(), safe_ge12=(first[col] >= 12).mean(),
                             safe_120=(first[col] >= 120).mean()))
    res = pd.DataFrame(rows)
    res.to_csv(OUT / "robustness.csv", index=False)
    pd.set_option("display.width", 200)
    print(res.round(3).to_string(index=False))


if __name__ == "__main__":
    main()
