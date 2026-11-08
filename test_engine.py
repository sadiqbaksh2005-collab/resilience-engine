"""
Independent re-implementation of the judges' definitions (vectorized pandas),
compared against engine.py for every account and every preset twist.
Run:  python3 test_engine.py
"""
import numpy as np
import pandas as pd

import engine as E

ms = pd.read_csv(E.DATA_DIR / "monthly_summary.csv")
ln = pd.read_csv(E.DATA_DIR / "loans.csv")

g = ms.sort_values("month").groupby("account_id")
ref = pd.DataFrame({
    "I": g["money_in"].median(),
    "S": g["money_out"].median(),
    "F": ms.assign(f=ms.household_bills + ms.insurance + ms.loan_payments).groupby("account_id")["f"].median(),
    "B": g.tail(6).groupby("account_id")["avg_daily_balance"].mean(),
    "H": g["household_bills"].median(),
})
running = ln[ln.status.isin(["C", "D"])].set_index("account_id")["monthly_payment"]
ref["P"] = running.reindex(ref.index).fillna(0.0)


def ref_runway(B, one_off, spend, inc):
    left, gap = B - one_off, spend - inc
    out = np.where(left <= 0, 0.0, np.where(gap <= 0, 120.0, np.minimum(left / np.where(gap > 0, gap, 1), 120)))
    return np.round(out, 1)


cases = {
    "none": (1.0, 0.0, 0.0),
    "ei55": (0.55, 0.0, 0.0),
    "furnace": (1.0, 0.0, 1.0),
    "ei55_bills15": (0.55, 0.15, 0.0),
}
for name, (im, bp, oneS) in cases.items():
    ref[f"rw_{name}"] = ref_runway(ref.B, oneS * ref.S, ref.S + bp * ref.H, im * ref.I)

shocks = {"none": E.Shock(), "ei55": E.Shock(income_mult=0.55),
          "furnace": E.Shock(one_off_months_S=1.0), "ei55_bills15": E.Shock(income_mult=0.55, bills_pct=0.15)}

bad = []
for a in ref.index:
    p = E.profile(int(a))
    for k in "IFSBHP":
        if not np.isclose(p[k], ref.at[a, k], atol=1e-6):
            bad.append((a, k, p[k], ref.at[a, k]))
    for name, s in shocks.items():
        v = E.apply_shock(p, s)["runway"]
        if not np.isclose(v, ref.at[a, f"rw_{name}"], atol=1e-9):
            bad.append((a, name, v, ref.at[a, f"rw_{name}"]))

print(f"checked {len(ref)} accounts x ({len('IFSBHP')} numbers + {len(shocks)} runways): {len(bad)} mismatches")
for b in bad[:10]:
    print("  ", b)

# unit tests of the runway branches
assert E.runway(100, 50, 60) == (120.0, "no shortfall (120+)")
assert E.runway(100, 60, 50, one_off=100) == (0.0, "0.0")
assert E.runway(100, 60, 50) == (10.0, "10.0")
assert E.runway(1e6, 60, 50) == (120.0, "120.0")
print("runway branch tests passed")

# population view of the preset twists (for the discovery section)
for name in cases:
    r = ref[f"rw_{name}"]
    print(f"  {name:13s} median runway {r.median():6.1f}  share < 6 months {np.mean(r < 6):.1%}  "
          f"share 120+ {np.mean(r >= 120):.1%}")
assert not bad


# ================= hardening: every account x many shocks =================
import math
import warnings as _w
from dataclasses import replace

rng = np.random.default_rng(7)


def random_shock():
    return E.Shock(
        income_mult=float(rng.choice([1.0, 0.55, 0.8, 0.0, 1.1])),
        income_change=float(rng.choice([0, -2000, 500])),
        ei_months=int(rng.choice([3, 6, 12])) if rng.random() < 0.7 else None,
        income_after=float(rng.choice([0.0, 1.0])),
        bills_pct=float(rng.choice([0, 0.15, 0.5])), bills_abs=float(rng.choice([0, 300])),
        spending_pct=float(rng.choice([0, 0.03])), extra_monthly=float(rng.choice([0, 800])),
        loan_payment_pct=float(rng.choice([0, 0.2])), one_off=float(rng.choice([0, 5000, 1e6])),
        one_off_months_S=float(rng.choice([0, 1])), one_off_months_I=float(rng.choice([0, 0.5])),
        ei_cap=bool(rng.random() < 0.5), keep_pension=bool(rng.random() < 0.5))


def finite_or_inf(x):
    return x is None or isinstance(x, str) or (isinstance(x, (int, float, np.floating)) and not math.isnan(x))


errors, n_checks = [], 0
ids = E.account_ids()
with _w.catch_warnings():
    _w.simplefilter("error", RuntimeWarning)       # any divide by zero / invalid value fails the test
    for a in ids:
        p = E.profile(a)
        for s in [E.Shock(), *E.PRESETS.values(), random_shock(), random_shock()]:
            try:
                r = E.apply_shock(p, s)
                assert 0.0 <= r["runway"] <= 120.0, r
                pl = E.plans(p, s)
                assert pl["goal"]["summary"] and pl["backup"]["summary"]
                for o in pl["backup"]["options"]:
                    assert 0.0 <= o["runway"] <= 120.0
                assert all(finite_or_inf(v) for v in pl["goal"].values() if not isinstance(v, str))
                E.drivers(p, s); E.warnings(p, s)
                df = E.path(p, s, 24)
                assert np.isfinite(df.balance).all()
                n_checks += 1
            except Exception as e:
                errors.append((a, s.describe(), repr(e)))
        if a % 7 == 0:   # Monte Carlo on a sample of accounts (it is the slow part)
            try:
                sv = E.survival(p, random_shock(), T=12, N=500)
                assert np.all((sv["curve"] >= 0) & (sv["curve"] <= 1)) and np.all(np.diff(sv["curve"]) <= 1e-12)
                assert np.isfinite(sv["fan"].drop(columns="month").to_numpy()).all()
                n_checks += 1
            except Exception as e:
                errors.append((a, "survival", repr(e)))

print(f"hardening: {n_checks:,} checks over {len(ids)} accounts, {len(errors)} errors")
for e in errors[:10]:
    print("  ", e)

# monotonicity: a worse shock never gives a longer runway
viol = 0
for a in ids[::5]:
    p = E.profile(a)
    for field, values in [("income_mult", [1.0, 0.8, 0.55, 0.0]), ("bills_pct", [0, 0.15, 0.5]),
                          ("one_off", [0, 1000, 10000]), ("extra_monthly", [0, 300, 1000]),
                          ("spending_pct", [0, 0.05, 0.2])]:
        rws = [E.apply_shock(p, replace(E.Shock(), **{field: v}))["runway"] for v in values]
        viol += sum(b > a_ + 1e-9 for a_, b in zip(rws, rws[1:]))
    # the cap can only lower EI income, pension kept can only raise it
    s = E.Shock(income_mult=0.55)
    viol += E.apply_shock(p, replace(s, ei_cap=True))["runway"] > E.apply_shock(p, s)["runway"] + 1e-9
    viol += E.apply_shock(p, replace(s, keep_pension=True))["runway"] < E.apply_shock(p, s)["runway"] - 1e-9
print(f"monotonicity violations: {viol}")

# investing layer: never above capacity, never invests in deficit, numbers finite
import portfolio as PF
mkt, inv_bad, inv_n = PF.load_market(), [], 0
for a in ids[::3]:
    p = E.profile(a)
    for tol in PF.LEVELS:
        try:
            pl = PF.invest_plan(p, tol, a % 2 == 0, n_paths=200, r=mkt)
            inv_n += 1
            assert PF.LEVELS.index(pl["level"]) <= min(PF.LEVELS.index(tol), PF.LEVELS.index(PF.CAPACITY_MAX[pl["capacity"]]))
            assert pl["upfront"] >= 0 and pl["weekly"] >= 0
            if pl["investable"]:
                assert pl["surplus"] > 0 and np.isfinite(pl["fan"].drop(columns="month").to_numpy()).all()
                assert pl["end_p10"] <= pl["end_p50"] <= pl["end_p90"]
        except Exception as e:
            inv_bad.append((a, tol, repr(e)))
print(f"investing: {inv_n:,} plans checked, {len(inv_bad)} errors {inv_bad[:3]}")
assert not errors and viol == 0 and not inv_bad
print("ALL TESTS PASSED")
