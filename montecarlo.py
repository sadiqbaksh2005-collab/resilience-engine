"""
Monte Carlo survival: "do they make it?" as a probability.

Each simulated month replays one of the customer's own historical months
(money_in, money_out, and the within month dip = month_end_balance - min_balance),
drawn in blocks of consecutive months (moving block bootstrap) so that bad
months cluster the way they really do.

    X_0 = B - C
    X_m = X_{m-1} + income_m - (spend_m + beta*H + E)
    low_m = X_m - dip_m                     (lowest point inside month m)
    survive T  <=>  min_{m<=T} low_m >= 0

No parameter is fitted to outcomes, so it can be validated out of sample directly.
"""
from __future__ import annotations
import json
from pathlib import Path

import numpy as np

POOL_MONTHS = 24     # use the customer's most recent 24 months
BURN_IN = 6          # skip the first 6 months of an account (deposit only "quiet" months)
BLOCK = 3


def block_indices(n: int, T: int, N: int, block: int, rng) -> np.ndarray:
    """(N, T) indices into a pool of length n, built from consecutive blocks."""
    b = max(1, min(block, n))
    nb = -(-T // b)                                     # ceil(T / b)
    starts = rng.integers(0, n - b + 1, size=(N, nb))
    idx = (starts[:, :, None] + np.arange(b)[None, None, :]).reshape(N, nb * b)
    return idx[:, :T]


FEEDBACK_FILE = Path(__file__).parent / "outputs" / "spend_feedback.json"


def load_feedback():
    """Spending response curve f(balance / S) estimated on the TRAIN period (fit_feedback)."""
    if FEEDBACK_FILE.exists():
        d = json.loads(FEEDBACK_FILE.read_text())
        return np.array(d["x"]), np.array(d["y"])
    return None


def simulate(pool_in, pool_out, pool_dip, start, T, N=5000, block=BLOCK,
             income_path=None, extra_spend=0.0, seed=0, feedback=None, S=None):
    """Returns balances X (N, T) and within month lows (N, T).
    income_path: length T array of deterministic income (e.g. EI), or None to resample.
    feedback: (x, y) curve; spending in month m is scaled by f(X_{m-1}/S) / f(start/S),
              i.e. customers spend less when their balance is low, as measured in the data."""
    rng = np.random.default_rng(seed)
    idx = block_indices(len(pool_out), T, N, block, rng)
    inc = pool_in[idx] if income_path is None else np.broadcast_to(np.asarray(income_path, float), (N, T))
    base_spend = pool_out[idx]
    if feedback is None or not S or S <= 0:
        X = start + np.cumsum(inc - base_spend - extra_spend, axis=1)
    else:
        fx, fy = feedback
        f = lambda b: np.interp(b / S, fx, fy)
        ref = f(np.asarray(start, float))
        X = np.empty((N, T))
        prev = np.full(N, float(start))
        for m in range(T):
            scale = f(prev) / ref
            prev = prev + inc[:, m] - base_spend[:, m] * scale - extra_spend
            X[:, m] = prev
    low = X - pool_dip[idx]
    return X, low


def survival_curve(low: np.ndarray) -> np.ndarray:
    """P(no overdraft through month m), m = 1..T."""
    return (np.minimum.accumulate(low, axis=1) >= 0).mean(axis=0)


def first_broke_month(low: np.ndarray) -> np.ndarray:
    """Month index (1 based) of the first overdraft per path, np.inf if none."""
    hit = low < 0
    first = np.where(hit.any(axis=1), hit.argmax(axis=1) + 1, np.inf)
    return first


def make_pool(hist):
    """hist: DataFrame of one account's months (sorted), with money_in, money_out,
    month_end_balance, min_balance. Returns pool arrays (most recent POOL_MONTHS after burn in)."""
    h = hist.iloc[BURN_IN:] if len(hist) > BURN_IN + 2 else hist
    h = h.tail(POOL_MONTHS)
    dip = (h["month_end_balance"] - h["min_balance"]).clip(lower=0).to_numpy()
    return h["money_in"].to_numpy(float), h["money_out"].to_numpy(float), dip


def fit_feedback(ms, train_end="2023-09", n_bins=20):
    """Median of money_out / own median spending, by last month's balance in months of
    spending. Uses TRAIN months only and each account's own train period medians."""
    import pandas as pd
    d = ms[ms["month"] <= train_end].sort_values(["account_id", "month"]).copy()
    g = d.groupby("account_id")
    d["tenure"] = g.cumcount() + 1
    S = g["money_out"].transform("median")
    d["out_rel"] = d["money_out"] / S
    d["bal_rel"] = g["month_end_balance"].shift(1) / S
    d = d[(d["tenure"] > BURN_IN + 1) & (S > 0)].dropna(subset=["bal_rel"])
    q = pd.qcut(d["bal_rel"], n_bins, duplicates="drop")
    t = d.groupby(q, observed=True).agg(x=("bal_rel", "median"), y=("out_rel", "median"))
    FEEDBACK_FILE.write_text(json.dumps(dict(x=t["x"].round(4).tolist(), y=t["y"].round(4).tolist(),
                                             note="train period only; f(balance_prev / S) = median spend / S")))
    return t
