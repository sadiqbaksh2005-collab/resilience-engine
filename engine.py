"""
Resilience engine: know the customer, take a shock, make a plan.

Part A and the headline runway follow the judges' definitions EXACTLY
(monthly_summary.csv, loans.csv). Everything beyond that (month by month
path, plans) is our extension and is labelled as such.

    from engine import profile, Shock, apply_shock, path, plans
    p = profile(20111)
    r = apply_shock(p, Shock(income_mult=0.55, bills_pct=0.15))
"""
from __future__ import annotations

from dataclasses import dataclass, asdict
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd

import os


def _find_data_dir() -> Path:
    """RESILIENCE_DATA env var, else ./data next to this file, else ~/Downloads/data."""
    for c in [os.environ.get("RESILIENCE_DATA"), Path(__file__).parent / "data", Path.home() / "Downloads" / "data"]:
        if c and (Path(c) / "monthly_summary.csv").exists():
            return Path(c)
    raise FileNotFoundError("monthly_summary.csv not found: set RESILIENCE_DATA or put the CSVs in ./data")


DATA_DIR = _find_data_dir()
RUNWAY_CAP = 120.0

# EI regular benefits, 2026 (canada.ca/en/services/benefits/ei/ei-regular-benefit/benefit-amount.html):
# 55% of average insurable weekly earnings, max $729/week, max insurable earnings $68,900/yr.
EI_RATE = 0.55
EI_MAX_WEEKLY = 729.0
EI_MAX_MONTHLY = EI_MAX_WEEKLY * 52 / 12     # = 3159.00

FIXED_COLS = ["household_bills", "insurance", "loan_payments"]
# money_out components a customer can realistically cut (fixed bills, fees excluded)
FLEX_COLS = ["cash_withdrawals_genuine", "card_withdrawals", "transfers_out_other"]
SPEND_COLS = ["household_bills", "insurance", "loan_payments", "bank_fees"] + FLEX_COLS
INCOME_COLS = ["cash_deposits", "transfers_in", "pension_in", "interest_in"]


# ---------------- data ----------------
@lru_cache(maxsize=1)
def _data():
    ms = pd.read_csv(DATA_DIR / "monthly_summary.csv")
    ms = ms.sort_values(["account_id", "month"])
    ln = pd.read_csv(DATA_DIR / "loans.csv")
    return {a: g.reset_index(drop=True) for a, g in ms.groupby("account_id")}, ln.set_index("account_id")


def account_ids():
    return sorted(_data()[0])


# ---------------- A. know the customer ----------------
def profile(account_id: int) -> dict:
    by_acc, ln = _data()
    if account_id not in by_acc:
        raise KeyError(f"account_id {account_id} not found")
    g = by_acc[account_id]

    I = g["money_in"].median()
    S = g["money_out"].median()
    F = (g["household_bills"] + g["insurance"] + g["loan_payments"]).median()
    B = g["avg_daily_balance"].tail(6).mean()          # last 6 rows, sorted by month
    H = g["household_bills"].median()
    P = 0.0
    if account_id in ln.index and ln.loc[account_id, "status"] in ("C", "D"):
        P = float(ln.loc[account_id, "monthly_payment"])   # running at end of 2025

    return dict(
        account_id=account_id, I=I, F=F, S=S, B=B, H=H, P=P,
        n_months=len(g), first_month=g["month"].iloc[0], last_month=g["month"].iloc[-1],
        surplus=I - S,
        runway=runway(B, S, I),
        # extensions (not judge numbers): category medians and history for charts/plans
        cat=g[SPEND_COLS + INCOME_COLS].median().to_dict(),
        flex=g[FLEX_COLS].sum(axis=1).median(),
        history=g[["month", "money_in", "money_out", "month_end_balance", "avg_daily_balance", "min_balance"]],
        loan=(ln.loc[account_id, ["amount", "duration_months", "monthly_payment", "status", "status_meaning"]].to_dict()
              if account_id in ln.index else None),
    )


def runway(buffer_left: float, spending: float, income: float, one_off: float = 0.0):
    """Judges' runway. Returns (value, label). value is a float in [0, 120]."""
    left = buffer_left - one_off
    gap = spending - income
    if left <= 0:
        return 0.0, "0.0"
    if gap <= 0:
        return RUNWAY_CAP, "no shortfall (120+)"
    v = round(min(left / gap, RUNWAY_CAP), 1)
    return v, f"{v:.1f}"


# ---------------- B. take a shock ----------------
@dataclass
class Shock:
    """Every twist is a combination of these. Defaults = no shock.
    Judges' formula:  income = income_mult * I,  spending = S + bills_pct * H,  one off = repair."""
    # income
    income_mult: float = 1.0          # 0.55 = laid off on EI; 0.8 = income drops 20%; 1.1 = raise
    income_change: float = 0.0        # $ per month added to income (negative = partner loses $2,000)
    ei_months: int | None = None      # months the income_mult applies (path / Monte Carlo only)
    income_after: float = 0.0         # income multiple once ei_months end (0 = no income)
    # spending
    bills_pct: float = 0.0            # +0.15 = household bills / rent up 15%   (adds pct * H)
    bills_abs: float = 0.0            # +300  = rent up $300 per month
    spending_pct: float = 0.0         # +0.03 = everything costs 3% more (CPI); applies to all of S
    extra_monthly: float = 0.0        # new recurring cost, $ per month (new child, car payment)
    loan_payment_pct: float = 0.0     # +0.20 = loan payment up 20% (rate reset); applies to P
    # one off
    one_off: float = 0.0              # surprise repair, $
    one_off_months_S: float = 0.0     # repair sized in months of typical spending
    one_off_months_I: float = 0.0     # repair sized in months of typical income
    # realism options (OFF by default so the headline number matches the judges)
    ei_cap: bool = False              # EI income capped at EI_MAX_MONTHLY
    keep_pension: bool = False        # pension + interest continue; EI replaces only the rest

    def describe(self) -> str:
        parts = []
        if self.income_mult != 1:
            d = f" for {self.ei_months} months" if self.ei_months else ""
            parts.append(f"income at {self.income_mult:.0%}{d}")
        if self.income_change:
            parts.append(f"income {self.income_change:+,.0f}/month")
        if self.bills_pct:
            parts.append(f"household bills {self.bills_pct:+.0%}")
        if self.bills_abs:
            parts.append(f"household bills {self.bills_abs:+,.0f}/month")
        if self.spending_pct:
            parts.append(f"all spending {self.spending_pct:+.0%}")
        if self.extra_monthly:
            parts.append(f"new cost {self.extra_monthly:+,.0f}/month")
        if self.loan_payment_pct:
            parts.append(f"loan payment {self.loan_payment_pct:+.0%}")
        if self.one_off:
            parts.append(f"one off ${self.one_off:,.0f}")
        if self.one_off_months_S:
            parts.append(f"one off = {self.one_off_months_S:g} month(s) of spending")
        if self.one_off_months_I:
            parts.append(f"one off = {self.one_off_months_I:g} month(s) of income")
        if self.ei_cap:
            parts.append(f"EI capped at ${EI_MAX_MONTHLY:,.0f}/month")
        if self.keep_pension:
            parts.append("pension kept")
        return ", ".join(parts) or "no shock"


def shocked_income(p: dict, s: Shock) -> float:
    """Monthly income while the shock lasts (judges: income_mult * I)."""
    if s.income_mult == 1.0:
        return p["I"] + s.income_change
    protected = (p["cat"]["pension_in"] + p["cat"]["interest_in"]) if s.keep_pension else 0.0
    protected = min(protected, p["I"])
    replaced = s.income_mult * (p["I"] - protected)
    if s.ei_cap:
        replaced = min(replaced, EI_MAX_MONTHLY)
    return protected + replaced + s.income_change


def income_after_shock(p: dict, s: Shock) -> float:
    """Monthly income once ei_months have passed (path / Monte Carlo only)."""
    return s.income_after * p["I"] + s.income_change


def added_spending(p: dict, s: Shock) -> float:
    """Recurring $ added to spending (judges: bills_pct * H)."""
    return (s.spending_pct * p["S"] + s.bills_pct * p["H"] + s.bills_abs
            + s.extra_monthly + s.loan_payment_pct * p["P"])


def shocked_flows(p: dict, s: Shock):
    income = shocked_income(p, s)
    spending = p["S"] + added_spending(p, s)
    one_off = s.one_off + s.one_off_months_S * p["S"] + s.one_off_months_I * p["I"]
    return income, spending, one_off


def apply_shock(p: dict, s: Shock) -> dict:
    income, spending, one_off = shocked_flows(p, s)
    val, label = runway(p["B"], spending, income, one_off)
    return dict(shock=s.describe(), income=income, spending=spending, one_off=one_off,
                buffer_left=p["B"] - one_off, gap=spending - income,
                runway=val, runway_label=label,
                runway_before=p["runway"][0], runway_change=val - p["runway"][0])


# ---------------- extension: month by month path ----------------
def path(p: dict, s: Shock, months: int = 24) -> pd.DataFrame:
    """Deterministic balance path. Unlike the single ratio, income can change when
    EI ends (ei_months, income_after). Balance starts at B minus one off costs."""
    income0, spending, one_off = shocked_flows(p, s)
    bal = p["B"] - one_off
    rows = []
    for m in range(1, months + 1):
        inc = income0 if (s.ei_months is None or m <= s.ei_months) else income_after_shock(p, s)
        bal = bal + inc - spending
        rows.append(dict(month=m, income=inc, spending=spending, balance=bal))
    return pd.DataFrame(rows)


def months_until_broke(df: pd.DataFrame, start_balance: float) -> float | None:
    """First (fractional) month the balance drops below 0, or None if it never does."""
    prev = start_balance
    for r in df.itertuples():
        if r.balance < 0:
            if prev <= 0:
                return float(r.month - 1)
            return round(r.month - 1 + prev / (prev - r.balance), 1)
        prev = r.balance
    return None


# ---------------- C. do they make it? (Monte Carlo, our extension) ----------------
def survival(p: dict, s: Shock | None = None, T: int = 12, N: int = 5000, seed: int = 0) -> dict:
    """Probability of never going below zero (inside any month) through each month 1..T.
    Fixed seed so a live rerun gives identical numbers."""
    import montecarlo as MC
    s = s or Shock()
    pin, pout, pdip = MC.make_pool(p["history"])
    _, _, one_off = shocked_flows(p, s)
    # spending: the customer's own months, scaled by spending_pct; fixed $ additions on top
    add = s.bills_pct * p["H"] + s.bills_abs + s.extra_monthly + s.loan_payment_pct * p["P"]
    pout = pout * (1 + s.spending_pct)
    if s.income_mult != 1.0:          # job loss style: income is a known benefit, not resampled
        inc_shock, inc_after = shocked_income(p, s), income_after_shock(p, s)
        income_path = np.array([inc_shock if (s.ei_months is None or m <= s.ei_months) else inc_after
                                for m in range(1, T + 1)])
    else:                             # income keeps its own swings, shifted by income_change
        income_path = None
        add -= s.income_change
    X, low = MC.simulate(pin, pout, pdip, p["B"] - one_off, T, N=N, income_path=income_path,
                         extra_spend=add, seed=seed)
    curve = MC.survival_curve(low)
    first = MC.first_broke_month(low)
    fan = pd.DataFrame(np.percentile(X, [5, 25, 50, 75, 95], axis=0).T,
                       columns=["p05", "p25", "p50", "p75", "p95"]).assign(month=np.arange(1, T + 1))
    pick = lambda m: float(curve[m - 1]) if m <= T else None
    return dict(p_survive_3=pick(3), p_survive_6=pick(6), p_survive_12=pick(12),
                curve=curve, fan=fan,
                median_months_to_broke=(float(np.median(first)) if np.isfinite(np.median(first)) else None),
                n_paths=N, pool_months=len(pout))


# ---------------- D. make a plan ----------------
CAT_LABELS = {
    "household_bills": "household bills", "insurance": "insurance", "loan_payments": "loan payments",
    "bank_fees": "bank fees", "cash_withdrawals_genuine": "cash withdrawals",
    "card_withdrawals": "card spending", "transfers_out_other": "other transfers out",
}


def _safe_div(a, b, default=np.inf):
    return a / b if b else default


def plans(p: dict, s: Shock | None = None, fund_months: float = 3.0,
          save_share: float = 0.5, cut_options=(0.10, 0.20, 0.30)) -> dict:
    """Goal plan (emergency fund) and backup plan (if the shock hits),
    built only from the customer's own medians. Never divides by zero; every
    branch returns a plain language 'summary'."""
    S, I, B, flex = p["S"], p["I"], p["B"], p["flex"]

    # ---- goal: emergency fund of fund_months x typical spending ----
    target = fund_months * S
    short = max(0.0, target - B)
    surplus = I - S
    goal = dict(target=target, current=B, shortfall=short, fund_months=fund_months,
                funded_pct=min(1.0, max(0.0, _safe_div(B, target, 1.0))),
                monthly_saving=0.0, months_to_goal=0.0, requires_flex_cut=0.0)
    if S <= 0:
        goal["summary"] = "No regular spending recorded, so no emergency fund target can be set from this history."
    elif short == 0:
        goal["summary"] = f"Already funded: the buffer covers {B / S:.1f} months of spending (target {fund_months:g})."
    else:
        save = save_share * surplus if surplus > 0 else 0.0
        if save <= 0 and flex > 0:          # no surplus: free cash by trimming flexible spending
            save, goal["requires_flex_cut"] = 0.10 * flex, 0.10
        goal["monthly_saving"] = save
        goal["months_to_goal"] = _safe_div(short, save)
        if save <= 0:
            goal["summary"] = (f"Needs ${short:,.0f} more, but spending already exceeds income and there is no "
                               f"flexible spending to trim: the plan has to start with income or fixed costs.")
        elif goal["requires_flex_cut"]:
            goal["summary"] = (f"Spending exceeds income, so trim flexible spending 10% (${save:,.0f}/month) and save it: "
                               f"${short:,.0f} to go, about {goal['months_to_goal']:.0f} months.")
        else:
            goal["summary"] = (f"Save {save_share:.0%} of the ${surplus:,.0f} monthly surplus (${save:,.0f}/month): "
                               f"${short:,.0f} to go, about {goal['months_to_goal']:.0f} months.")

    backup = None
    if s is not None:
        income, spending, one_off = shocked_flows(p, s)
        left, gap = B - one_off, spending - income
        base = apply_shock(p, s)
        options = []
        if flex > 0:
            for c in cut_options:
                v, lab = runway(left, spending - c * flex, income)
                options.append(dict(action=f"cut flexible spending {c:.0%} (${c * flex:,.0f}/month)",
                                    monthly_saving=c * flex, runway=v, runway_label=lab))
        if p["P"] > 0:
            v, lab = runway(left, spending - p["P"], income)
            options.append(dict(action=f"loan payment deferral (${p['P']:,.0f}/month)",
                                monthly_saving=p["P"], runway=v, runway_label=lab))
            if flex > 0:
                v, lab = runway(left, spending - p["P"] - 0.2 * flex, income)
                options.append(dict(action="deferral + 20% flexible cut",
                                    monthly_saving=p["P"] + 0.2 * flex, runway=v, runway_label=lab))
        T = s.ei_months or 6
        if left <= 0:
            needed, summary = np.inf, (f"The one off cost (${one_off:,.0f}) wipes out the ${B:,.0f} buffer immediately: "
                                       f"they need a payment plan or short term credit for ${-left:,.0f} on day one.")
        elif gap <= 0:
            needed, summary = 0.0, "Income still covers spending after the shock: no cuts needed, the buffer is untouched."
        else:
            needed = max(0.0, gap - left / T)
            share = _safe_div(needed, flex)
            if needed == 0:
                summary = f"The buffer lasts {base['runway_label']} months, which covers the {T} month horizon."
            elif flex > 0 and share <= 0.5:
                summary = (f"To last {T} months, cut ${needed:,.0f}/month, which is {share:.0%} of flexible spending "
                           f"(${flex:,.0f}/month).")
            else:
                extra = " Loan deferral would cover part of it." if p["P"] > 0 else ""
                summary = (f"To last {T} months they need ${needed:,.0f}/month less spending, more than flexible "
                           f"spending cuts can deliver (${flex:,.0f}/month in total).{extra}")
        backup = dict(base_runway=base["runway_label"], horizon=T, needed_monthly_cut=needed,
                      needed_cut_share_of_flex=_safe_div(needed, flex) if np.isfinite(needed) else np.inf,
                      feasible_with_flex_cuts=bool(np.isfinite(needed) and needed <= 0.5 * flex),
                      options=sorted(options, key=lambda o: -o["runway"]), summary=summary)
    return dict(goal=goal, backup=backup)


def drivers(p: dict, s: Shock | None = None) -> pd.DataFrame:
    """What the money goes on (category medians) and how many months of runway a 20%
    cut in each flexible category would buy under the shock. Answers 'why is this customer fragile?'"""
    s = s or Shock()
    income, spending, one_off = shocked_flows(p, s)
    left = p["B"] - one_off
    base = runway(left, spending, income)[0]
    rows = []
    for c in SPEND_COLS:
        amt = p["cat"][c]
        flexible = c in FLEX_COLS
        gain = (runway(left, spending - 0.2 * amt, income)[0] - base) if flexible and amt > 0 else np.nan
        rows.append(dict(category=CAT_LABELS[c], monthly=amt, share_of_spending=_safe_div(amt, p["S"], 0.0),
                         flexible=flexible, runway_gain_from_20pct_cut=gain))
    return pd.DataFrame(rows).sort_values("monthly", ascending=False).reset_index(drop=True)


def warnings(p: dict, s: Shock | None = None) -> list[str]:
    """Plain language caveats to show next to the numbers (and to say out loud live)."""
    s = s or Shock()
    w = []
    if p["S"] <= 0:
        w.append("No regular spending in this history (median money_out is 0): runway is not meaningful.")
    if p["B"] <= 0:
        w.append(f"The buffer is already {p['B']:,.0f}: runway is 0 before any shock.")
    if p["I"] < p["S"]:
        w.append(f"Already spending more than they earn: typical gap ${p['S'] - p['I']:,.0f}/month before any shock.")
    if (s.bills_pct or s.bills_abs) and p["H"] == 0:
        w.append("No household bills are paid through this account (H = 0), so a % bills shock changes nothing. "
                 "The judges' formula gives the same result; use a $ amount if the bills are paid elsewhere.")
    if s.income_mult != 1 and not s.ei_cap and s.income_mult * p["I"] > EI_MAX_MONTHLY:
        capped = apply_shock(p, Shock(**{**asdict(s), "ei_cap": True}))
        w.append(f"Real EI is capped at ${EI_MAX_WEEKLY:,.0f}/week (${EI_MAX_MONTHLY:,.0f}/month), below "
                 f"{s.income_mult:.0%} of income here: with the cap, runway is {capped['runway_label']} months.")
    if s.income_mult != 1 and not s.keep_pension and p["cat"]["pension_in"] > 0:
        w.append(f"This customer receives a pension (${p['cat']['pension_in']:,.0f}/month typical), which a layoff "
                 f"would not cut. Turn on 'pension kept' for a realistic view.")
    if p["flex"] <= 0:
        w.append("No flexible spending to cut: backup options are limited to loan relief or income.")
    return w


# ---------------- presets for the live twist ----------------
PRESETS = {
    "Laid off, EI 55% for 6 months": Shock(income_mult=0.55, ei_months=6),
    "Furnace repair = 1 month of spending": Shock(one_off_months_S=1.0),
    "Job loss (EI 55%) + bills up 15%": Shock(income_mult=0.55, ei_months=6, bills_pct=0.15),
}


if __name__ == "__main__":
    import sys
    acc = int(sys.argv[1]) if len(sys.argv) > 1 else account_ids()[0]
    p = profile(acc)
    print(f"account {acc}: I={p['I']:,.2f} F={p['F']:,.2f} S={p['S']:,.2f} B={p['B']:,.2f} "
          f"H={p['H']:,.2f} P={p['P']:,.2f} runway={p['runway'][1]}")
    for name, s in PRESETS.items():
        r = apply_shock(p, s)
        print(f"  {name:40s} runway {r['runway_label']:>20s}  (gap {r['gap']:,.2f}/month, buffer left {r['buffer_left']:,.2f})")
