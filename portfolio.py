"""
Investment layer: after the cushion, invest the rest. Illustration, not advice.

1. Cushion target: months of typical spending, set by how irregular the customer's income is
   (income CV over the last 12 months: < 0.25 -> 3 months, < 0.5 -> 4.5, else 6).
2. Risk capacity (what they can afford, from their own data, no demographics):
     Low    if any: flagged by the early warning model, overdrawn in the last 12 months,
                    buffer under 1 month of spending, income CV >= 0.5 without a buffer of 2x the cushion,
                    loan payment >= 30% of income
     High   if all: buffer >= 2x cushion, income CV < 0.25, loan payment < 10% of income,
                    monthly surplus >= 10% of income
     Medium otherwise
   Portfolio = the LOWER of capacity and the customer's stated risk tolerance.
3. Only if income exceeds spending. Upfront = buffer above the cushion. Weekly = half the monthly surplus (x 12 / 52), starting once
   the cushion is full (until then that half fills the cushion).
4. Three model portfolios of broad index ETFs (no stock picking), fixed strategic weights.
   Projection: 10 years, 12 month blocks of the ETFs' real monthly total returns (bootstrap),
   compared with the same money in savings at an assumed rate.

Market data: yfinance, fetched once into outputs/market_returns.csv; the app only reads that file.
Run: python3 portfolio.py          (refresh data + population summary)
"""
from __future__ import annotations
from pathlib import Path
import numpy as np
import pandas as pd

OUT = Path(__file__).parent / "outputs"
MARKET = OUT / "market_returns.csv"
TICKERS = {"XIC.TO": "Canadian stocks (XIC)", "XUU.TO": "US stocks (XUU)",
           "XEF.TO": "International stocks (XEF)", "XBB.TO": "Canadian bonds (XBB)"}
# equity split fixed at 30% Canada / 45% US / 25% international; bonds fill the rest
PORTFOLIOS = {
    "Conservative": {"XIC.TO": 0.09, "XUU.TO": 0.135, "XEF.TO": 0.075, "XBB.TO": 0.70},   # 30% stocks
    "Balanced":     {"XIC.TO": 0.18, "XUU.TO": 0.27,  "XEF.TO": 0.15,  "XBB.TO": 0.40},   # 60% stocks
    "Growth":       {"XIC.TO": 0.27, "XUU.TO": 0.405, "XEF.TO": 0.225, "XBB.TO": 0.10},   # 90% stocks
}
LEVELS = ["Conservative", "Balanced", "Growth"]
CAPACITY_MAX = {"Low": "Conservative", "Medium": "Balanced", "High": "Growth"}
SAVINGS_RATE = 0.025     # assumed savings account rate, editable in the app


# ---------------- market data ----------------
def fetch_market():
    import yfinance as yf
    px = yf.download(list(TICKERS), start="2010-01-01", interval="1mo", auto_adjust=True, progress=False)["Close"]
    px = px.dropna()
    px = px[px.index < pd.Timestamp.today().normalize().replace(day=1)]     # drop the unfinished month
    r = px.pct_change().dropna()
    r.index = r.index.strftime("%Y-%m")
    r.to_csv(MARKET)
    return r


def load_market() -> pd.DataFrame:
    return pd.read_csv(MARKET, index_col=0)


def portfolio_returns(r: pd.DataFrame, level: str) -> pd.Series:
    w = pd.Series(PORTFOLIOS[level])
    return r[w.index] @ w


def portfolio_stats(r: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for lv in LEVELS:
        x = portfolio_returns(r, lv)
        growth = (1 + x).cumprod()
        roll12 = (1 + x).rolling(12).apply(np.prod, raw=True) - 1
        rows.append(dict(portfolio=lv, stocks=f"{1 - PORTFOLIOS[lv]['XBB.TO']:.0%}",
                         annual_return=float(growth.iloc[-1] ** (12 / len(x)) - 1),
                         volatility=float(x.std() * np.sqrt(12)),
                         worst_12_months=float(roll12.min()),
                         max_drawdown=float((growth / growth.cummax() - 1).min())))
    return pd.DataFrame(rows)


# ---------------- the customer ----------------
def cushion_months(cv: float) -> float:
    return 3.0 if cv < 0.25 else 4.5 if cv < 0.5 else 6.0


def risk_capacity(p: dict, ew_flag: bool = False) -> dict:
    h = p["history"].tail(12)
    inc = h["money_in"]
    cv = float(inc.std() / inc.mean()) if inc.mean() > 0 else np.inf
    S, I, B, P = p["S"], p["I"], p["B"], p["P"]
    cm = cushion_months(cv)
    cushion = cm * S
    buf_m = B / S if S > 0 else np.inf
    loan = P / I if I > 0 else (np.inf if P > 0 else 0.0)
    surplus = I - S
    recent_od = bool((h["min_balance"] < 0).any())
    low = {"flagged by the early warning model": ew_flag,
           "overdrawn in the last 12 months": recent_od,
           "buffer under 1 month of spending": buf_m < 1,
           f"irregular income (variation {cv:.0%}) and buffer under 2x the cushion": cv >= 0.5 and B < 2 * cushion,
           f"loan payment {loan:.0%} of income": loan >= 0.3}
    high = {"buffer at least 2x the cushion": B >= 2 * cushion,
            "steady income": cv < 0.25,
            "light or no loan payment": loan < 0.1,
            "surplus at least 10% of income": I > 0 and surplus >= 0.1 * I}
    reasons_low = [k for k, v in low.items() if v]
    if reasons_low:
        cap, why = "Low", reasons_low
    elif all(high.values()):
        cap, why = "High", list(high)
    else:
        cap, why = "Medium", [f"not {k}" for k, v in high.items() if not v]
    return dict(capacity=cap, why=why, income_cv=cv, cushion_months=cm, cushion=cushion,
                buffer_months=buf_m, loan_share=loan, surplus=surplus)


def invest_plan(p: dict, tolerance: str = "Balanced", ew_flag: bool = False,
                save_share: float = 0.5, years: int = 10, savings_rate: float = SAVINGS_RATE,
                n_paths: int = 5000, seed: int = 0, r: pd.DataFrame | None = None) -> dict:
    rc = risk_capacity(p, ew_flag)
    level = LEVELS[min(LEVELS.index(tolerance), LEVELS.index(CAPACITY_MAX[rc["capacity"]]))]
    B, surplus = p["B"], rc["surplus"]
    upfront = max(0.0, B - rc["cushion"])
    monthly = save_share * surplus if surplus > 0 else 0.0
    gap = max(0.0, rc["cushion"] - B)
    fill_months = int(np.ceil(gap / monthly)) if gap > 0 and monthly > 0 else (0 if gap == 0 else None)
    weekly = monthly * 12 / 52
    plan = dict(rc, level=level, capped=level != tolerance, tolerance=tolerance, upfront=upfront,
                monthly=monthly, weekly=weekly, cushion_gap=gap, fill_months=fill_months)
    T = years * 12
    if surplus <= 0:
        plan.update(investable=False, upfront=0.0,
                    summary=(f"Spending exceeds income by ${-surplus:,.0f} a month, so the buffer is being drawn down. "
                             "Close that gap before investing anything."))
        return plan
    if fill_months is None or (upfront == 0 and monthly == 0):
        plan.update(investable=False,
                    summary=("Nothing to invest yet: spending uses all of income and the buffer is below the "
                             f"{rc['cushion_months']:g} month cushion. The plan is the cushion first."))
        return plan
    if fill_months >= T:
        plan.update(investable=False, summary=(f"Filling the {rc['cushion_months']:g} month cushion takes "
                                               f"{fill_months} months, so investing waits until then."))
        return plan

    r = load_market() if r is None else r
    x = portfolio_returns(r, level).to_numpy()
    contrib = np.where(np.arange(T) >= fill_months, monthly, 0.0)
    rng = np.random.default_rng(seed)
    nb = int(np.ceil(T / 12))
    starts = rng.integers(0, len(x) - 12 + 1, size=(n_paths, nb))
    R = x[(starts[:, :, None] + np.arange(12)).reshape(n_paths, -1)[:, :T]]
    V = np.empty((n_paths, T))
    v = np.full(n_paths, upfront)
    for t in range(T):
        v = (v + contrib[t]) * (1 + R[:, t])
        V[:, t] = v
    cash, c = np.empty(T), upfront
    for t in range(T):
        c = (c + contrib[t]) * (1 + savings_rate / 12)
        cash[t] = c
    paid = upfront + np.cumsum(contrib)
    fan = pd.DataFrame(np.percentile(V, [10, 25, 50, 75, 90], axis=0).T, columns=["p10", "p25", "p50", "p75", "p90"])
    fan["month"], fan["savings"], fan["paid_in"] = np.arange(1, T + 1), cash, paid
    end = V[:, -1]
    plan.update(investable=True, fan=fan, end_p10=float(np.percentile(end, 10)), end_p50=float(np.median(end)),
                end_p90=float(np.percentile(end, 90)), end_savings=float(cash[-1]), paid_in=float(paid[-1]),
                p_below_paid=float((end < paid[-1]).mean()), p_beats_savings=float((end > cash[-1]).mean()),
                market_months=len(x), market_from=r.index[0], market_to=r.index[-1])
    start = "starting now" if fill_months == 0 else f"starting after {fill_months} months of filling the cushion"
    first = (f"Put ${upfront:,.0f} in now and ${weekly:,.0f} a week ({start})" if upfront > 0
             else f"Invest ${weekly:,.0f} a week ({start})")
    plan["summary"] = (f"{first} into the {level} portfolio. "
                       f"In {years} years: likely ${plan['end_p50']:,.0f} (bad case ${plan['end_p10']:,.0f}, good case "
                       f"${plan['end_p90']:,.0f}), against ${cash[-1]:,.0f} in savings at {savings_rate:.1%}.")
    return plan


# ---------------- population ----------------
def main():
    import engine as E
    r = fetch_market()
    print(f"market data: {r.index[0]} to {r.index[-1]}, {len(r)} months")
    print(portfolio_stats(r).round(3).to_string(index=False))
    bank = pd.read_csv(OUT / "bank_view.csv")
    flag = dict(zip(bank.account_id, bank.flag))
    rows = []
    for a in E.account_ids():
        p = E.profile(a)
        if p["S"] <= 0:
            continue
        rc = risk_capacity(p, bool(flag.get(a, False)))
        upfront = max(0.0, p["B"] - rc["cushion"]) if rc["surplus"] > 0 else 0.0
        rows.append(dict(account_id=a, capacity=rc["capacity"], cushion_months=rc["cushion_months"],
                         income_cv=rc["income_cv"], upfront=upfront, surplus=rc["surplus"],
                         below_cushion=p["B"] < rc["cushion"], nothing=(rc["surplus"] <= 0)))
    d = pd.DataFrame(rows)
    d.to_csv(OUT / "capacity.csv", index=False)
    print("\ncapacity:", d.capacity.value_counts(normalize=True).round(3).to_dict())
    print("cushion months:", d.cushion_months.value_counts(normalize=True).round(3).sort_index().to_dict())
    print(f"below their cushion: {d.below_cushion.mean():.1%}; nothing to invest at all: {d.nothing.mean():.1%}; "
          f"median upfront (those with some): ${d.loc[d.upfront > 0, 'upfront'].median():,.0f}")
    three = 3 * pd.Series([E.profile(a)["S"] for a in d.account_id])
    print(f"vs the flat 3 month rule: {(d.cushion_months > 3).mean():.1%} need MORE than 3 months because of irregular income")
    for a in (25257, 56795, 28667, 29010):
        pl = invest_plan(E.profile(a), "Growth", bool(flag.get(a, False)), r=r)
        print(a, pl["capacity"], pl["level"], "|", pl["summary"])


if __name__ == "__main__":
    main()
