"""
Does the judges' runway predict real trouble?

Leakage note: computing runway with the judges' definitions on the FULL history
and correlating it with "ever overdrew" is circular (B uses the last 6 months,
which come after most events). So we rebuild every number AS OF month t,
using months <= t only, and predict the first overdraft in t+1..t+3 on the
same risk set rows, split and burn in as the model.

Also: runway at loan origination vs later loan default.

Outputs: outputs/runway_validation.csv, outputs/runway_buckets.csv
"""
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, average_precision_score

import engine as E
from features import FEATURES
from baselines import make_lr, bootstrap

OUT = Path(__file__).parent / "outputs"
CAP = E.RUNWAY_CAP


def runway_vec(B, S, I, one_off=0.0):
    left, gap = B - one_off, S - I
    r = np.where(gap > 0, left / np.where(gap > 0, gap, 1.0), CAP)
    r = np.where(left <= 0, 0.0, np.minimum(r, CAP))
    return np.round(r, 1)


def asof_numbers():
    """I, S, H, B and pension/interest as of each month, using months <= t only."""
    ms = pd.read_csv(E.DATA_DIR / "monthly_summary.csv").sort_values(["account_id", "month"])
    g = ms.groupby("account_id")
    exp_med = lambda c: g[c].transform(lambda s: s.expanding().median())
    out = ms[["account_id", "month"]].copy()
    out["I"], out["S"], out["H"] = exp_med("money_in"), exp_med("money_out"), exp_med("household_bills")
    out["prot"] = exp_med("pension_in") + exp_med("interest_in")
    out["B"] = g["avg_daily_balance"].transform(lambda s: s.rolling(6, min_periods=1).mean())
    out["t"] = pd.PeriodIndex(out["month"], freq="M")
    return out


def scenario_runways(d):
    ei = E.EI_RATE * d["I"]
    ei_real = np.minimum(E.EI_RATE * (d["I"] - d["prot"].clip(upper=d["I"])), E.EI_MAX_MONTHLY) + d["prot"].clip(upper=d["I"])
    return pd.DataFrame({
        "runway_base": runway_vec(d["B"], d["S"], d["I"]),
        "runway_ei55": runway_vec(d["B"], d["S"], ei),
        "runway_ei55_bills15": runway_vec(d["B"], d["S"] + 0.15 * d["H"], ei),
        "runway_ei_realistic": runway_vec(d["B"], d["S"], ei_real),      # cap + pension kept
        "runway_furnace": runway_vec(d["B"], d["S"], d["I"], one_off=d["S"]),
    }, index=d.index)


def bucket(r):
    return pd.cut(r, [-0.01, 1, 3, 6, 12, 119.9, CAP], labels=["0 to 1", "1 to 3", "3 to 6", "6 to 12", "12 to 120", "120+"])


def main():
    asof = asof_numbers()
    asof = pd.concat([asof, scenario_runways(asof)], axis=1)
    RW = [c for c in asof if c.startswith("runway_")]

    p = pd.read_pickle(OUT / "panel.pkl").merge(asof[["account_id", "t"] + RW], on=["account_id", "t"])
    tr, te = p[p.split == "train"], p[p.split == "test"]

    # ---- models: runway alone vs ours (and ours without the snapshot feature) ----
    feats_noso = [f for f in FEATURES if f != "so_burden"]
    lr_full = make_lr().fit(tr[FEATURES], tr["y"])
    lr_noso = make_lr().fit(tr[feats_noso], tr["y"])
    lr_rw = make_lr().fit(tr[feats_noso + RW], tr["y"])
    te = te.assign(model_full=lr_full.predict_proba(te[FEATURES])[:, 1],
                   model_no_snapshot=lr_noso.predict_proba(te[feats_noso])[:, 1],
                   model_no_snapshot_plus_runway=lr_rw.predict_proba(te[feats_noso + RW])[:, 1])
    for c in RW:
        te[f"neg_{c}"] = -te[c]          # shorter runway = more risk

    rows = []
    for col in [f"neg_{c}" for c in RW] + ["model_no_snapshot", "model_no_snapshot_plus_runway", "model_full"]:
        ci = bootstrap(te, col, thr=np.inf, n=200)
        rows.append(dict(score=col.replace("neg_", ""), auc=roc_auc_score(te.y, te[col]),
                         auc_lo=ci.auc.iloc[0], auc_hi=ci.auc.iloc[1],
                         ap=average_precision_score(te.y, te[col])))
    res = pd.DataFrame(rows)
    print(f"TEST: first overdraft within 3 months  (rows={len(te):,}, events={te.loc[te.y == 1, 'account_id'].nunique()}, "
          f"prevalence={te.y.mean():.4f})")
    print(res.round(3).to_string(index=False))

    # ---- calibration table: event rate by runway bucket (all splits, it is descriptive) ----
    tabs = []
    for c in ["runway_base", "runway_ei55", "runway_ei_realistic"]:
        t = p.groupby(bucket(p[c]), observed=False)["y"].agg(rows="size", events="sum", rate="mean")
        t["scenario"] = c
        tabs.append(t.reset_index().rename(columns={c: "bucket"}))
    tab = pd.concat(tabs)
    print("\n3 month overdraft rate by runway bucket (as of t):")
    print(tab.pivot(index="bucket", columns="scenario", values="rate").mul(100).round(2).astype(str).add("%").to_string())
    print(tab.pivot(index="bucket", columns="scenario", values="rows").to_string())

    # ---- blind spot: overdraft events that runway considered safe ----
    first_rows = p[p.y == 1].sort_values("t").groupby("account_id").head(1)   # earliest warning row per event
    for c in ["runway_base", "runway_ei55"]:
        safe = (first_rows[c] >= 12).mean()
        print(f"  events whose {c} was >= 12 months three months before they overdrew: {safe:.1%}")

    # ---- loans: runway at origination vs default ----
    ln = pd.read_csv(E.DATA_DIR / "loans.csv")
    ln["t"] = pd.PeriodIndex(ln["loan_start"].str[:7], freq="M") - 1        # month before the loan
    hist = asof.groupby("account_id")["t"].transform(lambda s: np.arange(1, len(s) + 1))
    L = ln.merge(asof.assign(hist=hist), on=["account_id", "t"], how="inner")
    L = L[L["hist"] >= 6]
    S_new = L["S"] + L["monthly_payment"]                                    # add the new payment
    L["runway_with_loan"] = runway_vec(L["B"], S_new, L["I"])
    L["runway_with_loan_ei55"] = runway_vec(L["B"], S_new, E.EI_RATE * L["I"])
    print(f"\nLOANS with >= 6 months history before origination: {len(L)}  defaults: {L.defaulted.sum()}")
    for c in ["runway_base", "runway_ei55", "runway_with_loan", "runway_with_loan_ei55"]:
        print(f"  AUC of -{c:24s} for default: {roc_auc_score(L.defaulted, -L[c]):.3f}")
    print(L.groupby(bucket(L["runway_with_loan_ei55"]), observed=False)["defaulted"].agg(["size", "sum", "mean"]).round(3))

    res.to_csv(OUT / "runway_validation.csv", index=False)
    tab.to_csv(OUT / "runway_buckets.csv", index=False)


if __name__ == "__main__":
    main()
