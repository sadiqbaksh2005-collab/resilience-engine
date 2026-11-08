"""
State migration matrix: Stable / Stretched / Distressed, like a credit rating migration matrix.

States (as of each month, history up to that month only):
    buffer_months = B_t / S_t,  B_t = mean avg_daily_balance of the last 6 months (judges' B, as of t),
                                S_t = median money_out so far (judges' S, as of t)
    Distressed : overdrawn this month (min_balance < 0) or buffer_months < 1
    Stretched  : 1 <= buffer_months < 3
    Stable     : buffer_months >= 3
Months 1..6 of an account are burn in (no state), as in the early warning model.

Estimated on train months (t <= 2023-09), checked on the held out test year (2024-10..2025-09):
  1. stability: train matrix vs test matrix (account bootstrap CIs)
  2. Markov check: P^k from train vs the actual k month moves in test (k = 3, 6, 12)
  3. memory check: does last month's state change next month's odds, given this month's?
  4. prediction: chance of a first overdraft in the next 3 months by state, AUC vs judges' runway
  5. shock matrix: moves in months when income fell below 60% of the customer's usual
Outputs: outputs/migration.json, outputs/states_latest.csv
"""
from __future__ import annotations
from pathlib import Path
import json
import numpy as np
import pandas as pd

OUT = Path(__file__).parent / "outputs"
NAMES = ["Stable", "Stretched", "Distressed"]
MIN_TENURE = 7
TRAIN_END = pd.Period("2023-09", "M")
VAL_END = pd.Period("2024-09", "M")
LAST = pd.Period("2025-12", "M")
SHOCK_INCOME = 0.6
N_BOOT = 200


# ---------------- states ----------------
def add_states(h: pd.DataFrame) -> pd.DataFrame:
    """h: one account's monthly rows sorted by month with money_in, money_out, avg_daily_balance,
    min_balance. Adds S_t, B_t, buffer_months, state (0/1/2, NaN during burn in)."""
    h = h.copy()
    h["S_t"] = h["money_out"].expanding().median()
    h["B_t"] = h["avg_daily_balance"].rolling(6, min_periods=1).mean()
    h["buffer_months"] = np.where(h["S_t"] > 0, h["B_t"] / h["S_t"].where(h["S_t"] > 0), np.inf)
    s = np.where((h["min_balance"] < 0) | (h["buffer_months"] < 1), 2, np.where(h["buffer_months"] < 3, 1, 0))
    h["state"] = np.where(np.arange(len(h)) + 1 >= MIN_TENURE, s, np.nan)
    return h


def all_states() -> pd.DataFrame:
    import engine as E
    m = pd.read_csv(E.DATA_DIR / "monthly_summary.csv",
                    usecols=["account_id", "month", "money_in", "money_out", "avg_daily_balance", "min_balance"])
    m = m.sort_values(["account_id", "month"]).reset_index(drop=True)
    g = m.groupby("account_id")
    m["S_t"] = g["money_out"].transform(lambda x: x.expanding().median())
    m["I_t"] = g["money_in"].transform(lambda x: x.expanding().median())
    m["B_t"] = g["avg_daily_balance"].transform(lambda x: x.rolling(6, min_periods=1).mean())
    m["buffer_months"] = np.where(m["S_t"] > 0, m["B_t"] / m["S_t"].where(m["S_t"] > 0), np.inf)
    m["tenure"] = g.cumcount() + 1
    s = np.where((m["min_balance"] < 0) | (m["buffer_months"] < 1), 2, np.where(m["buffer_months"] < 3, 1, 0))
    m["state"] = np.where(m["tenure"] >= MIN_TENURE, s, np.nan)
    m["t"] = pd.PeriodIndex(m["month"], freq="M")
    g = m.groupby("account_id")
    m["next_state"] = g["state"].shift(-1)
    m["prev_state"] = g["state"].shift(1)
    m["next_income"] = g["money_in"].shift(-1)
    consecutive = (g["t"].shift(-1) - m["t"]).map(lambda d: getattr(d, "n", 0)) == 1
    m["has_next"] = consecutive & m["state"].notna() & m["next_state"].notna()
    m["neg"] = (m["min_balance"] < 0).astype(int)
    m["ever_neg"] = g["neg"].cummax()
    return m


# ---------------- matrices ----------------
def counts(df: pd.DataFrame, a="state", b="next_state") -> np.ndarray:
    c = np.zeros((3, 3))
    np.add.at(c, (df[a].to_numpy(int), df[b].to_numpy(int)), 1)
    return c


def normalize(c: np.ndarray) -> np.ndarray:
    return c / np.maximum(c.sum(1, keepdims=True), 1)


def bootstrap_ci(df: pd.DataFrame, seed=0):
    rng = np.random.default_rng(seed)
    per_acc = {a: counts(g) for a, g in df.groupby("account_id")}
    C = np.stack(list(per_acc.values()))
    draws = [normalize(C[rng.integers(0, len(C), len(C))].sum(0)) for _ in range(N_BOOT)]
    return np.percentile(draws, 2.5, axis=0), np.percentile(draws, 97.5, axis=0)


def k_step_empirical(m: pd.DataFrame, rows: pd.Series, k: int) -> tuple[np.ndarray, int]:
    """Actual state k months later (same account, exactly k months on) for the given rows."""
    fut = m[["account_id", "t", "state"]].rename(columns={"state": "s_k"}).copy()
    fut["t"] = fut["t"] - k
    d = m.loc[rows, ["account_id", "t", "state"]].merge(fut, on=["account_id", "t"]).dropna()
    return normalize(counts(d, "state", "s_k")), len(d)


def main():
    from sklearn.metrics import roc_auc_score
    from validate_runway import runway_vec

    m = all_states()
    tr = m[m.has_next & (m.t <= TRAIN_END)]
    te = m[m.has_next & (m.t > VAL_END) & (m.t < LAST)]
    P, Pte = normalize(counts(tr)), normalize(counts(te))
    lo, hi = bootstrap_ci(tr)
    lo_te, hi_te = bootstrap_ci(te, seed=1)
    res = dict(states=NAMES, n_train=int(len(tr)), n_test=int(len(te)),
               P=P.tolist(), P_lo=lo.tolist(), P_hi=hi.tolist(),
               P_test=Pte.tolist(), P_test_lo=lo_te.tolist(), P_test_hi=hi_te.tolist(),
               state_share_train=np.bincount(tr.state.astype(int), minlength=3).tolist())

    # 2. Markov check: P^k (train) vs actual k step moves on the test year
    test_rows = m.state.notna() & (m.t > VAL_END)      # start in the test year; landing month must exist
    mk = {}
    for k in (3, 6, 12):
        emp, n = k_step_empirical(m, test_rows, k)
        pred = np.linalg.matrix_power(P, k)
        mk[k] = dict(pred=pred.tolist(), actual=emp.tolist(), n=n, max_abs_diff=float(np.abs(pred - emp).max()))
    res["markov_check"] = mk

    # 3. memory check: P(next | current = Stretched, previous = each state), train
    mem = tr[tr.prev_state.notna() & (tr.state == 1)]
    res["memory_stretched"] = {NAMES[int(ps)]: dict(n=int(len(g)), to_distressed=float((g.next_state == 2).mean()),
                                                     to_stable=float((g.next_state == 0).mean()))
                               for ps, g in mem.groupby("prev_state")}

    # 4. prediction of a first overdraft in the next 3 months (never overdrawn, test year)
    g = m.groupby("account_id")
    fut = pd.concat([g["neg"].shift(-k) for k in (1, 2, 3)], axis=1)
    m["y3"] = fut.max(axis=1, skipna=False)
    a = pd.read_pickle(OUT / "asof.pkl")
    a["rw"] = runway_vec(a["B"], a["S"], a["I"])
    r = m[(m.ever_neg == 0) & m.state.notna() & m.y3.notna() & (m.t > VAL_END) & (m.t <= LAST - 3)]
    r = r.merge(a[["account_id", "t", "rw"]], on=["account_id", "t"], how="left")
    P3 = np.linalg.matrix_power(P, 3)
    r["p_distressed_3"] = P3[r.state.astype(int), 2]
    res["prediction"] = dict(
        rows=int(len(r)), events=int(r.y3.sum()),
        auc_state=float(roc_auc_score(r.y3, r.state)), auc_runway=float(roc_auc_score(r.y3, -r.rw)),
        overdraft_rate_by_state={NAMES[int(s)]: dict(rows=int(len(gg)), rate=float(gg.y3.mean()))
                                 for s, gg in r.groupby("state")},
        share_of_events_by_state={NAMES[int(s)]: float(gg.y3.sum() / r.y3.sum()) for s, gg in r.groupby("state")})

    # 5. shock matrix: transitions into a month where income fell below 60% of usual (all periods, burn in excluded)
    allm = m[m.has_next]
    shock = allm[(allm.I_t > 0) & (allm.next_income < SHOCK_INCOME * allm.I_t)]
    calm = allm[(allm.I_t > 0) & (allm.next_income >= SHOCK_INCOME * allm.I_t)]
    Ps, Pc = normalize(counts(shock)), normalize(counts(calm))
    slo, shi = bootstrap_ci(shock, seed=2)
    res["shock"] = dict(rule=f"next month money_in < {SHOCK_INCOME:.0%} of median money_in so far",
                        n=int(len(shock)), P_shock=Ps.tolist(), P_shock_lo=slo.tolist(), P_shock_hi=shi.tolist(),
                        P_calm=Pc.tolist(), n_calm=int(len(calm)))

    # bank view: latest state of every account, last month's moves, 12 month forecast of the mix
    last = m[m.state.notna()].groupby("account_id").tail(1)
    prev = m[m.state.notna() & (m.t == last.t.max() - 1)][["account_id", "state"]].rename(columns={"state": "prev"})
    moves = last.merge(prev, on="account_id")
    mix = np.bincount(last.state.astype(int), minlength=3).astype(float)
    res["bank"] = dict(month=str(last.t.max()), mix=mix.tolist(),
                       moves_last_month=counts(moves, "prev", "state").astype(int).tolist(),
                       forecast=[(mix @ np.linalg.matrix_power(P, k)).tolist() for k in range(0, 13)])
    last[["account_id", "month", "state", "buffer_months"]].assign(
        state_name=lambda d: d.state.map(dict(enumerate(NAMES)))).to_csv(OUT / "states_latest.csv", index=False)

    with open(OUT / "migration.json", "w") as f:
        json.dump(res, f, indent=1)
    report(res)


def report(res):
    f = lambda M: pd.DataFrame(M, index=NAMES, columns=NAMES).round(3).to_string()
    print(f"train transitions {res['n_train']:,}, test {res['n_test']:,}")
    print("P (train)\n" + f(res["P"]))
    print("95% CI low\n" + f(res["P_lo"]) + "\n95% CI high\n" + f(res["P_hi"]))
    print("P (test)\n" + f(res["P_test"]))
    for k, v in res["markov_check"].items():
        print(f"\nMarkov check k={k} (n={v['n']:,}) max |pred - actual| = {v['max_abs_diff']:.3f}")
        print("pred\n" + f(v["pred"]) + "\nactual\n" + f(v["actual"]))
    print("\nmemory (now Stretched):", json.dumps(res["memory_stretched"], indent=1))
    print("\nprediction:", json.dumps(res["prediction"], indent=1))
    print("\nshock n", res["shock"]["n"], "\nP_shock\n" + f(res["shock"]["P_shock"]) + "\nP_calm\n" + f(res["shock"]["P_calm"]))
    print("\nbank", res["bank"]["month"], "mix", res["bank"]["mix"], "\nmoves\n", np.array(res["bank"]["moves_last_month"]))
    print("forecast 12m", np.round(res["bank"]["forecast"][12], 0))


if __name__ == "__main__":
    main()
