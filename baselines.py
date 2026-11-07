"""
Baselines: simple balance rule, logistic regression, gradient boosting.

Protocol
  fit on train, tune and freeze the alert threshold on val, report test ONCE.
  Metrics: AUC, average precision, precision at 50% recall,
           event recall and lead time at the frozen threshold,
           alert rate (share of non event rows flagged),
           95% CIs from an account level bootstrap on test.

Output: outputs/scores.pkl (row scores for every model), outputs/baseline_results.csv
"""
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, precision_recall_curve, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, StandardScaler

from features import FEATURES, HORIZON

OUT = Path(__file__).parent / "outputs"
TARGET_RECALL = 0.50
N_BOOT = 300
RNG = np.random.default_rng(42)


# ---------------- metrics ----------------
def precision_at_recall(y, s, r=TARGET_RECALL):
    prec, rec, _ = precision_recall_curve(y, s)
    return prec[rec >= r].max()


def threshold_for_recall(y, s, r=TARGET_RECALL):
    """Highest threshold whose row recall is at least r (set on VAL, then frozen)."""
    order = np.sort(s[y == 1])[::-1]
    k = int(np.ceil(r * len(order))) - 1
    return order[k]


LEAD_LOOKBACK = 6


def months_between(a, b):
    return (a.dt.year - b.dt.year) * 12 + (a.dt.month - b.dt.month)


def event_metrics(df, score_col, thr):
    """Event = an account's first negative month E.
    Caught: flagged in at least one label window row, t in [E-3, E-1].
    Lead (caught events only): E minus the earliest flag in [E-6, E-1].
    Lead is not capped by the 3 month label horizon; flags at E-6..E-4 are
    rows with y=0, so they also count as false alerts in alert_rate."""
    ev = df[df["event_t"].notna()].copy()
    ev["k"] = months_between(ev["event_t"], ev["t"])          # months before event
    ev = ev[(ev["k"] >= 1) & (ev["k"] <= LEAD_LOOKBACK)]
    n_events = ev.loc[ev["k"] <= HORIZON, "account_id"].nunique()
    if n_events == 0:
        return dict(events=0, event_recall=np.nan, median_lead=np.nan, mean_lead=np.nan)
    fl = ev[ev[score_col] >= thr]
    caught = fl.loc[fl["k"] <= HORIZON, "account_id"].unique()
    lead = fl[fl["account_id"].isin(caught)].groupby("account_id")["k"].max()
    return dict(events=n_events, event_recall=len(caught) / n_events,
                median_lead=float(lead.median()) if len(lead) else np.nan,
                mean_lead=float(lead.mean()) if len(lead) else np.nan)


def evaluate(df, score_col, thr):
    y, s = df["y"].to_numpy(), df[score_col].to_numpy()
    out = dict(auc=roc_auc_score(y, s), ap=average_precision_score(y, s),
               prec_at_r50=precision_at_recall(y, s),
               row_recall=(s[y == 1] >= thr).mean(),
               alert_rate=(s[y == 0] >= thr).mean())
    out.update(event_metrics(df, score_col, thr))
    return out


def bootstrap(df, score_col, thr, n=N_BOOT):
    """Resample ACCOUNTS (not rows): rows of one account are correlated."""
    groups = {a: g for a, g in df.groupby("account_id")}
    ids = np.array(list(groups))
    rows = []
    for _ in range(n):
        b = pd.concat([groups[a] for a in RNG.choice(ids, len(ids), replace=True)])
        if b["y"].nunique() < 2:
            continue
        y, s = b["y"].to_numpy(), b[score_col].to_numpy()
        rows.append(dict(auc=roc_auc_score(y, s), ap=average_precision_score(y, s),
                         event_recall=event_metrics(b, score_col, thr)["event_recall"]))
    return pd.DataFrame(rows).quantile([0.025, 0.975])


# ---------------- models ----------------
def signed_log(X):
    return np.sign(X) * np.log1p(np.abs(X))


def make_lr():
    return Pipeline([
        ("impute", SimpleImputer(strategy="median")),
        ("slog", FunctionTransformer(signed_log)),
        ("scale", StandardScaler()),
        ("lr", LogisticRegression(C=0.1, max_iter=5000)),
    ])


HGB_GRID = [
    dict(learning_rate=0.05, max_iter=300, max_leaf_nodes=15, min_samples_leaf=200, l2_regularization=1.0),
    dict(learning_rate=0.03, max_iter=500, max_leaf_nodes=7, min_samples_leaf=300, l2_regularization=1.0),
    dict(learning_rate=0.05, max_iter=200, max_leaf_nodes=31, min_samples_leaf=100, l2_regularization=0.5),
]


def main():
    p = pd.read_pickle(OUT / "panel.pkl")
    tr, va, te = (p[p.split == s].copy() for s in ["train", "val", "test"])
    Xtr, ytr = tr[FEATURES], tr["y"]

    scores = p[["account_id", "t", "split", "y", "event_t"]].copy()

    # 1) rule baseline: fewer months of spending covered by balance = riskier
    scores["rule"] = -p["months_cover"].to_numpy()

    # 2) logistic regression
    lr = make_lr().fit(Xtr, ytr)
    scores["logreg"] = lr.predict_proba(p[FEATURES])[:, 1]

    # 3) gradient boosting, config chosen by VAL average precision
    best, best_ap = None, -1
    for cfg in HGB_GRID:
        m = HistGradientBoostingClassifier(random_state=0, early_stopping=False, **cfg).fit(Xtr, ytr)
        ap = average_precision_score(va["y"], m.predict_proba(va[FEATURES])[:, 1])
        print(f"  HGB {cfg} val AP={ap:.4f}")
        if ap > best_ap:
            best, best_ap = m, ap
    scores["gbm"] = best.predict_proba(p[FEATURES])[:, 1]

    # ---- evaluation ----
    results = []
    for model in ["rule", "logreg", "gbm"]:
        v = scores[scores.split == "val"]
        thr = threshold_for_recall(v["y"].to_numpy(), v[model].to_numpy())
        for split in ["train", "val", "test"]:
            r = evaluate(scores[scores.split == split], model, thr)
            results.append(dict(model=model, split=split, threshold=thr, **r))
    res = pd.DataFrame(results)

    pd.set_option("display.width", 200)
    print("\nPrevalence (AP of a random score):",
          scores.groupby("split")["y"].mean().round(4).to_dict())
    cols = ["model", "split", "auc", "ap", "prec_at_r50", "row_recall", "alert_rate",
            "events", "event_recall", "median_lead", "mean_lead"]
    print(res[cols].round(3).to_string(index=False))

    # ---- bootstrap CIs on test ----
    te_s = scores[scores.split == "test"]
    print(f"\nTest 95% CIs (account bootstrap, {N_BOOT} draws):")
    for model in ["rule", "logreg", "gbm"]:
        thr = res.query("model == @model and split == 'val'")["threshold"].iloc[0]
        ci = bootstrap(te_s, model, thr)
        print(f"  {model:7s} AUC [{ci.auc.iloc[0]:.3f}, {ci.auc.iloc[1]:.3f}]  "
              f"AP [{ci.ap.iloc[0]:.3f}, {ci.ap.iloc[1]:.3f}]  "
              f"event recall [{ci.event_recall.iloc[0]:.2f}, {ci.event_recall.iloc[1]:.2f}]")

    # ---- interpretability ----
    coef = pd.Series(lr.named_steps["lr"].coef_[0], index=FEATURES).sort_values(key=np.abs, ascending=False)
    print("\nLogReg top standardized coefficients (+ = more risk):")
    print(coef.head(10).round(3).to_string())

    from sklearn.inspection import permutation_importance
    pi = permutation_importance(best, va[FEATURES], va["y"], scoring="average_precision",
                                n_repeats=5, random_state=0, n_jobs=-1)
    imp = pd.Series(pi.importances_mean, index=FEATURES).sort_values(ascending=False)
    print("\nGBM permutation importance on VAL (drop in AP):")
    print(imp.head(10).round(4).to_string())

    scores.to_pickle(OUT / "scores.pkl")
    res.to_csv(OUT / "baseline_results.csv", index=False)


if __name__ == "__main__":
    main()
