"""
Population stress test and fairness audit.

Part 1 (descriptive, end of 2025 position, judges' definitions):
  every account through every twist, broken down by segment.
Part 2 (fairness of the early warning model, test period):
  event recall, alert rate and precision by segment at the frozen threshold.
  Demographics are NEVER model inputs; they are joined only here.

Outputs: outputs/population_accounts.csv, outputs/population_segments.csv,
         outputs/fairness.csv
"""
from pathlib import Path
import numpy as np
import pandas as pd

import engine as E
from features import FEATURES
from baselines import make_lr, threshold_for_recall

OUT = Path(__file__).parent / "outputs"
AS_OF = pd.Timestamp("2025-12-31")
EI = dict(income_mult=0.55, ei_months=6)
TWISTS = {
    "ei55": E.Shock(**EI),
    "ei55_bills15": E.Shock(**EI, bills_pct=0.15),
    "furnace": E.Shock(one_off_months_S=1.0),
    "ei_realistic": E.Shock(**EI, ei_cap=True, keep_pension=True),
}


def segments():
    D = E.DATA_DIR
    acc = pd.read_csv(D / "accounts.csv")
    dp = pd.read_csv(D / "dispositions.csv").query("type == 'owner'")
    cl = pd.read_csv(D / "clients.csv")
    di = pd.read_csv(D / "districts.csv")
    cd = pd.read_csv(D / "cards.csv").merge(pd.read_csv(D / "dispositions.csv")[["disposition_id", "account_id"]])
    ln = pd.read_csv(D / "loans.csv")

    s = acc.merge(dp[["account_id", "client_id"]], on="account_id").merge(cl, on="client_id", suffixes=("_branch", "_home"))
    s = s.merge(di[["district_id", "region", "unemployment_rate_2023", "average_salary"]],
                left_on="district_id_home", right_on="district_id", how="left")
    age = (AS_OF - pd.to_datetime(s["birth_date"])).dt.days / 365.25
    s["age_band"] = pd.cut(age, [0, 25, 35, 50, 65, 120], labels=["under 25", "25 to 34", "35 to 49", "50 to 64", "65+"], right=False)
    s["unemp_quartile"] = pd.qcut(s["unemployment_rate_2023"], 4, labels=["Q1 lowest", "Q2", "Q3", "Q4 highest"])
    best_card = cd.assign(r=cd.card_type.map({"junior": 1, "classic": 2, "gold": 3})).sort_values("r").groupby("account_id")["card_type"].last()
    s["card"] = s["account_id"].map(best_card).fillna("none")
    s["loan"] = s["account_id"].map(ln.set_index("account_id")["status"].map(
        {"A": "repaid", "B": "defaulted", "C": "running OK", "D": "running in debt"})).fillna("no loan")
    return s[["account_id", "gender", "age_band", "region", "unemp_quartile", "card", "loan"]]


def account_table():
    rows = []
    for a in E.account_ids():
        p = E.profile(a)
        r = dict(account_id=a, I=p["I"], S=p["S"], B=p["B"], H=p["H"], P=p["P"],
                 runway_base=p["runway"][0], buffer_months=p["B"] / p["S"] if p["S"] else np.inf,
                 ever_overdrawn=float((p["history"]["min_balance"] < 0).any()),
                 ei_cap_binds=float(0.55 * p["I"] > E.EI_MAX_MONTHLY))
        for k, s in TWISTS.items():
            r[f"runway_{k}"] = E.apply_shock(p, s)["runway"]
        r["p_survive6_ei55"] = E.survival(p, TWISTS["ei55"], T=6, N=2000)["p_survive_6"]
        rows.append(r)
    return pd.DataFrame(rows)


def summarize(df, by):
    g = df.groupby(by, observed=True)
    out = g.agg(accounts=("account_id", "size"),
                median_income=("I", "median"),
                median_buffer_months=("buffer_months", "median"),
                short_ei55=("runway_ei55", lambda r: (r < 6).mean()),
                short_ei55_bills15=("runway_ei55_bills15", lambda r: (r < 6).mean()),
                short_ei_realistic=("runway_ei_realistic", lambda r: (r < 6).mean()),
                short_furnace=("runway_furnace", lambda r: (r < 6).mean()),
                mc_p_survive6_ei55=("p_survive6_ei55", "mean"),
                ei_cap_binds=("ei_cap_binds", "mean"),
                ever_overdrawn=("ever_overdrawn", "mean"))
    out.insert(0, "segment", by)
    return out.reset_index().rename(columns={by: "group"})


def fairness(seg):
    p = pd.read_pickle(OUT / "panel.pkl")
    feats = [f for f in FEATURES if f != "so_burden"]
    tr, va, te = (p[p.split == s].copy() for s in ["train", "val", "test"])
    lr = make_lr().fit(tr[feats], tr["y"])
    va["s"], te["s"] = lr.predict_proba(va[feats])[:, 1], lr.predict_proba(te[feats])[:, 1]
    thr = threshold_for_recall(va["y"].to_numpy(), va["s"].to_numpy())
    te = te.merge(seg, on="account_id")
    te["flag"] = te["s"] >= thr
    rows = []
    for by in ["gender", "age_band", "region", "unemp_quartile"]:
        for grp, g in te.groupby(by, observed=True):
            pos = g[g.y == 1]
            ev_caught = pos[pos.flag].account_id.nunique()
            ev = pos.account_id.nunique()
            rows.append(dict(segment=by, group=grp, rows=len(g), events=ev,
                             event_recall=ev_caught / ev if ev else np.nan,
                             row_recall=pos.flag.mean() if len(pos) else np.nan,
                             alert_rate=g.loc[g.y == 0, "flag"].mean(),
                             precision=g.loc[g.flag, "y"].mean() if g.flag.any() else np.nan))
    f = pd.DataFrame(rows)
    f["small_sample"] = f["events"] < 10
    return f, thr


def main():
    pd.set_option("display.width", 220)
    seg = segments()
    acc = account_table().merge(seg, on="account_id")
    acc.to_csv(OUT / "population_accounts.csv", index=False)

    acc["all"] = "all customers"
    tabs = pd.concat([summarize(acc, by) for by in ["all", "gender", "age_band", "region", "unemp_quartile", "card", "loan"]])
    tabs.to_csv(OUT / "population_segments.csv", index=False)
    show = tabs.copy()
    for c in [c for c in show if c.startswith("short_") or c in ("mc_p_survive6_ei55", "ei_cap_binds", "ever_overdrawn")]:
        show[c] = (show[c] * 100).round(1)
    show["median_income"] = show["median_income"].round(0)
    show["median_buffer_months"] = show["median_buffer_months"].round(2)
    print("POPULATION (percent of accounts with runway < 6 months, per twist; MC = mean P(survive 6 months on EI))")
    print(show.to_string(index=False))

    f, thr = fairness(seg)
    f.to_csv(OUT / "fairness.csv", index=False)
    print(f"\nFAIRNESS of early warning model (test, frozen val threshold {thr:.4f}, no demographic inputs)")
    print(f.round(3).to_string(index=False))


if __name__ == "__main__":
    main()
