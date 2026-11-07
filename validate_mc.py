"""
Out of sample check of the Monte Carlo: as of each test month t (history <= t only),
P(overdraft within 3 months) vs the real first overdraft, on the same rows as the model.
Nothing in the simulation is fitted, so no training is involved.
Output: outputs/mc_validation.csv, outputs/mc_scores.pkl
"""
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, average_precision_score, brier_score_loss

import engine as E
import montecarlo as MC
from features import FEATURES
from baselines import make_lr, bootstrap

OUT = Path(__file__).parent / "outputs"
N, T = 300, 3


def main():
    ms = pd.read_csv(E.DATA_DIR / "monthly_summary.csv").sort_values(["account_id", "month"])
    ms["t"] = pd.PeriodIndex(ms["month"], freq="M")
    panel = pd.read_pickle(OUT / "panel.pkl")
    rows = panel[panel.split.isin(["val", "test"])][["account_id", "t", "split", "y", "event_t"]]
    want = rows.groupby("account_id")["t"].apply(set).to_dict()
    print(MC.fit_feedback(ms).round(3).T.to_string())
    fb = MC.load_feedback()

    rec = []
    for a, g in ms.groupby("account_id", sort=False):
        if a not in want:
            continue
        g = g.reset_index(drop=True)
        inc, out = g["money_in"].to_numpy(float), g["money_out"].to_numpy(float)
        dip = (g["month_end_balance"] - g["min_balance"]).clip(lower=0).to_numpy()
        adb, meb = g["avg_daily_balance"].to_numpy(), g["month_end_balance"].to_numpy()
        for j in np.flatnonzero(g["t"].isin(want[a]).to_numpy()):
            lo = max(MC.BURN_IN, j - MC.POOL_MONTHS + 1)
            pi, po, pd_ = inc[lo:j + 1], out[lo:j + 1], dip[lo:j + 1]
            B = adb[max(0, j - 5):j + 1].mean()
            I = np.median(inc[:j + 1])
            seed = (a * 1000 + j) % (2**32)
            _, low_b = MC.simulate(pi, po, pd_, B, T, N=N, seed=seed)
            _, low_c = MC.simulate(pi, po, pd_, meb[j], T, N=N, seed=seed)          # actual current balance
            _, low_e = MC.simulate(pi, po, pd_, B, T, N=N, seed=seed, income_path=np.full(T, 0.55 * I))
            S = np.median(out[:j + 1])
            _, low_fb = MC.simulate(pi, po, pd_, B, T, N=N, seed=seed, feedback=fb, S=S)
            _, low_fc = MC.simulate(pi, po, pd_, meb[j], T, N=N, seed=seed, feedback=fb, S=S)
            rec.append((a, g["t"].iat[j],
                        1 - MC.survival_curve(low_b)[-1], 1 - MC.survival_curve(low_c)[-1],
                        1 - MC.survival_curve(low_e)[-1],
                        1 - MC.survival_curve(low_fb)[-1], 1 - MC.survival_curve(low_fc)[-1]))
    mc = pd.DataFrame(rec, columns=["account_id", "t", "mc_od_B", "mc_od_current", "mc_od_ei55",
                                    "mc_fb_B", "mc_fb_current"])
    d = rows.merge(mc, on=["account_id", "t"])
    d.to_pickle(OUT / "mc_scores.pkl")

    # model with MC as an extra feature (fit on train needs train rows too: use val as fit set? no.)
    te = d[d.split == "test"].copy()
    res = []
    for col in ["mc_od_B", "mc_od_current", "mc_od_ei55", "mc_fb_B", "mc_fb_current"]:
        ci = bootstrap(te, col, thr=np.inf, n=200)
        res.append(dict(score=col, auc=roc_auc_score(te.y, te[col]), auc_lo=ci.auc.iloc[0], auc_hi=ci.auc.iloc[1],
                        ap=average_precision_score(te.y, te[col]), brier=brier_score_loss(te.y, te[col])))
    res = pd.DataFrame(res)
    print(f"TEST rows={len(te):,} events={te.loc[te.y == 1, 'account_id'].nunique()} prevalence={te.y.mean():.4f}")
    print(res.round(4).to_string(index=False))

    # calibration: predicted P(overdraft within 3m) vs observed
    bins = [-0.001, 0.0, 0.01, 0.05, 0.2, 0.5, 1.0]
    for col in ["mc_od_current", "mc_fb_current"]:
        cal = te.groupby(pd.cut(te[col], bins), observed=False).agg(rows=("y", "size"), mean_pred=(col, "mean"),
                                                                   observed=("y", "mean"), events=("y", "sum"))
        print(f"\ncalibration {col}:\n{cal.round(4).to_string()}")

    # blind spot comparison with runway: earliest warning row of each event
    first = te[te.y == 1].sort_values("t").groupby("account_id").head(1)
    for col in ["mc_od_B", "mc_od_current", "mc_fb_B", "mc_fb_current"]:
        print(f"  {col}: events the MC called near safe (P(od) < 1%) three months out: {(first[col] < 0.01).mean():.1%}"
              f"   non event rows with P(od) >= 1%: {(te.loc[te.y == 0, col] >= 0.01).mean():.1%}")
    res.to_csv(OUT / "mc_validation.csv", index=False)


if __name__ == "__main__":
    main()
