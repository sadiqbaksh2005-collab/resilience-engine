"""
Layer 1: Gaussian HMM on monthly behaviour, with forward filtering.

Fit       train period only (t <= TRAIN_END), all months incl. post event.
Select K  BIC on train, held out log likelihood on val, stability across seeds.
Score     FILTERED probabilities P(S_t | x_1:t)   (never predict_proba: it smooths
          with future months), plus the first passage probability
          P(enter stress within 3 months | x_1:t) from the transition matrix.

Outputs   outputs/hmm_probs.pkl, outputs/hmm_params.json
"""
import json
from itertools import permutations
from pathlib import Path

import numpy as np
import pandas as pd
from hmmlearn.hmm import GaussianHMM
from scipy.optimize import linear_sum_assignment
from scipy.special import logsumexp
from sklearn.metrics import adjusted_rand_score

from features import DATA_DIR, TRAIN_END, VAL_END, HORIZON, MIN_TENURE

OUT = Path(__file__).parent / "outputs"
K_GRID = [2, 3, 4, 5]
SEEDS = [0, 1, 2, 3, 4]
EPS = 1.0

# Scale free behaviour features, so states describe "how the customer is doing"
# rather than "how rich the customer is".
# Chosen because they DRIFT in the months before an event (see pre event profile);
# cash_wd_share and log_ntx were dropped: they describe customer type, are flat
# before events, and produced artifact states (quiet month vs busy month).
HMM_FEATURES = ["buf_rel", "log_out_in_r3", "slog_cover", "z_min_balance", "slog_bal_chg_rel"]


def slog(x):
    return np.sign(x) * np.log1p(np.abs(x))


def build_obs():
    ms = pd.read_pickle(OUT / "monthly_full.pkl")
    tx = pd.read_csv(DATA_DIR / "transactions.csv", usecols=["account_id", "date"])
    ntx = tx.assign(month=tx["date"].str[:7]).groupby(["account_id", "month"]).size().rename("n_tx")
    ms = ms.merge(ntx, on=["account_id", "month"], how="left").fillna({"n_tx": 0})
    ms = ms.sort_values(["account_id", "t"]).reset_index(drop=True)

    inc = ms.groupby("account_id")["money_in"].transform(lambda s: s.expanding().mean())
    ms["log_out_in"] = np.log((ms["money_out"] + EPS) / (ms["money_in"] + EPS))
    ms["slog_cover"] = slog(ms["months_cover"])
    ms["slog_bal_chg_rel"] = slog(ms["bal_chg_3m"].fillna(0) / (inc + EPS))
    ms["log_ntx"] = np.log1p(ms["n_tx"])
    g = ms.groupby("account_id")
    ms["buf_rel"] = slog(ms["min_balance"] / (inc + EPS))       # min balance in months of income
    r3 = lambda c: g[c].transform(lambda s: s.rolling(3, min_periods=1).sum())
    ms["log_out_in_r3"] = np.log((r3("money_out") + EPS) / (r3("money_in") + EPS))
    for c in ["z_net_flow", "z_min_balance"]:
        ms[c] = ms[c].fillna(0.0)          # no history yet: assume no deviation
    # burn in: sequences start once the account has MIN_TENURE months (same rule as panel)
    return ms[ms["tenure"] >= MIN_TENURE].reset_index(drop=True)


def standardize(ms):
    tr = ms["t"] <= TRAIN_END
    lo, hi = ms.loc[tr, HMM_FEATURES].quantile(0.005), ms.loc[tr, HMM_FEATURES].quantile(0.995)
    W = ms[HMM_FEATURES].clip(lo, hi, axis=1)       # winsorize at TRAIN quantiles
    mu, sd = W[tr].mean(), W[tr].std()
    X = ((W - mu) / sd).to_numpy()
    return X, mu, sd


def lengths_of(ids):
    _, idx, counts = np.unique(ids, return_index=True, return_counts=True)
    return counts[np.argsort(idx)]


def fit(X, lengths, K, seed):
    m = GaussianHMM(n_components=K, covariance_type="diag", n_iter=300, tol=1e-3,
                    random_state=seed, init_params="stmc", params="stmc")
    m.fit(X, lengths)
    return m


def align(ref, m):
    """Permutation of m's states that best matches ref's state means."""
    cost = ((ref.means_[:, None, :] - m.means_[None, :, :]) ** 2).sum(-1)
    _, col = linear_sum_assignment(cost)
    return col


def forward_filter(m, X, lengths):
    """Log space forward recursion. Returns P(S_t = k | x_1..x_t) for every row."""
    logB = m._compute_log_likelihood(X)           # log b_k(x_t), shape (T, K)
    logA, logpi = np.log(m.transmat_ + 1e-300), np.log(m.startprob_ + 1e-300)
    out = np.empty_like(logB)
    start = 0
    for L in lengths:
        la = logpi + logB[start]
        out[start] = la - logsumexp(la)
        for t in range(1, L):
            la = logsumexp(out[start + t - 1][:, None] + logA, axis=0) + logB[start + t]
            out[start + t] = la - logsumexp(la)
        start += L
    return np.exp(out)


def first_passage(P, A, stress, h=HORIZON):
    """P(S_{t+j} = stress for some j in 1..h | x_1:t) = 1 - p_t A[:,ns] A[ns,ns]^(h-1) 1."""
    ns = [k for k in range(A.shape[0]) if k != stress]
    stay = A[:, ns] @ np.linalg.matrix_power(A[np.ix_(ns, ns)], h - 1) @ np.ones(len(ns))
    return 1.0 - P @ stay


def main():
    ms = build_obs()
    X, mu, sd = standardize(ms)
    tr, va = (ms["t"] <= TRAIN_END).to_numpy(), ((ms["t"] > TRAIN_END) & (ms["t"] <= VAL_END)).to_numpy()
    ids = ms["account_id"].to_numpy()
    Ltr, Lva = lengths_of(ids[tr]), lengths_of(ids[va])

    # ---------- model selection ----------
    rows, models = [], {}
    for K in K_GRID:
        fits = []
        for s in SEEDS:
            m = fit(X[tr], Ltr, K, s)
            fits.append((m.score(X[tr], Ltr), m))
        fits.sort(key=lambda z: -z[0])
        best = fits[0][1]
        models[K] = best
        # stability: decoded train states of each seed vs best fit (ARI, label free)
        z_best = best.predict(X[tr], Ltr)
        ari = [adjusted_rand_score(z_best, m.predict(X[tr], Ltr)) for _, m in fits[1:]]
        A_diff = [np.abs(best.transmat_ - m.transmat_[np.ix_(align(best, m), align(best, m))]).max()
                  for _, m in fits[1:]]
        rows.append(dict(K=K, train_ll=fits[0][0], bic=best.bic(X[tr], Ltr),
                         val_ll_per_obs=best.score(X[va], Lva) / va.sum(),
                         ari_median=np.median(ari), ari_min=np.min(ari),
                         A_maxdiff_median=np.median(A_diff)))
        print(rows[-1])
    sel = pd.DataFrame(rows)
    print(sel.round(4).to_string(index=False))
    sel.to_csv(OUT / "hmm_selection.csv", index=False)


K_FINAL = 3   # largest K that is identical across all seeds (see hmm_selection.csv)
STATE_NAMES = {3: ["stable", "tight", "stressed"]}


def final():
    ms = build_obs()
    X, mu, sd = standardize(ms)
    tr = (ms["t"] <= TRAIN_END).to_numpy()
    ids = ms["account_id"].to_numpy()
    Ltr, Lall = lengths_of(ids[tr]), lengths_of(ids)

    m = max((fit(X[tr], Ltr, K_FINAL, s) for s in SEEDS), key=lambda z: z.score(X[tr], Ltr))

    # ---- forward filter on full sequences (row t only uses months <= t) ----
    P = forward_filter(m, X, Lall)
    # check: at a sequence's last month, filtered == smoothed
    ends = np.cumsum(Lall) - 1
    sm = m.predict_proba(X, Lall)
    assert np.allclose(P[ends], sm[ends], atol=1e-6), "forward filter mismatch"
    print("forward filter check passed; mean |filtered - smoothed| on non end rows:",
          np.abs(np.delete(P, ends, 0) - np.delete(sm, ends, 0)).mean().round(4))

    # ---- order and name states by severity (share of negative balance months, train) ----
    neg = (ms["min_balance"] < 0).to_numpy()
    neg_share = np.array([(P[tr, k] * neg[tr]).sum() / P[tr, k].sum() for k in range(K_FINAL)])
    order = np.argsort(neg_share)                     # least to most severe
    P = P[:, order]
    A = m.transmat_[np.ix_(order, order)]
    pi = m.startprob_[order]
    means_std = m.means_[order]
    names = STATE_NAMES.get(K_FINAL, [f"state_{k}" for k in range(K_FINAL)])
    stress = K_FINAL - 1

    # ---- profiles in original units (filter weighted, train) ----
    raw = ms.loc[tr, HMM_FEATURES + ["min_balance", "money_in", "money_out", "months_cover"]]
    w = P[tr]
    prof = pd.DataFrame({names[k]: (raw.mul(w[:, k], axis=0).sum() / w[:, k].sum()) for k in range(K_FINAL)}).T
    prof["share_of_months"] = w.mean(0)
    prof["neg_balance_share"] = neg_share[order]
    prof["expected_duration_months"] = 1 / (1 - np.diag(A))
    print("\nState profiles (train, filter weighted):")
    print(prof.round(3).to_string())
    print("\nTransition matrix A (rows = from, cols = to):")
    print(pd.DataFrame(A, index=names, columns=names).round(4).to_string())

    # ---- scores ----
    out = ms[["account_id", "t"]].copy()
    for k in range(K_FINAL):
        out[f"p_state_{k}"] = P[:, k]
    out["p_stress"] = P[:, stress]
    out["p_enter_stress_3m"] = first_passage(P, A, stress)
    out.to_pickle(OUT / "hmm_probs.pkl")

    json.dump(dict(K=K_FINAL, state_names=names, stress_state=stress,
                   transmat=A.tolist(), startprob=pi.tolist(),
                   features=HMM_FEATURES, feature_mean=mu.tolist(), feature_std=sd.tolist(),
                   means_standardized=means_std.tolist(),
                   profiles=prof.round(4).to_dict(orient="index")),
              open(OUT / "hmm_params.json", "w"), indent=2)
    print("saved hmm_probs.pkl, hmm_params.json")


if __name__ == "__main__":
    import sys
    if "--select" in sys.argv:
        main()
    final()
