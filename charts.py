"""
Pitch charts (static PNG, 16:9) -> outputs/charts/
  1_blind_spot.png      how well each score ranks real overdrafts (AUC + 95% CI)
  2_warning_signal.png  buffer and spending drift in the 12 months before an overdraft
  3_fragility_age.png   share under 6 months runway on EI, judges' formula vs realistic EI
  4_borrowers.png       share under 6 months runway on EI by loan status
  5_fan_<id>.png        one customer's balance paths after a layoff (live demo visual)
"""
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import engine as E

OUT = Path(__file__).parent / "outputs"
CH = OUT / "charts"
CH.mkdir(parents=True, exist_ok=True)

# reference palette, light mode (validated: blue/orange pass all checks)
SURF, INK, INK2, MUTED, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#8a8984", "#e6e5e1"
BLUE, ORANGE, GRAY = "#2a78d6", "#eb6834", "#b4b3ad"
SEQ = ["#cde2fb", "#9ec5f4", "#5598e7", "#256abf"]      # sequential blue for the fan

plt.rcParams.update({
    "figure.facecolor": SURF, "axes.facecolor": SURF, "savefig.facecolor": SURF,
    "font.family": "sans-serif", "font.size": 13, "text.color": INK,
    "axes.edgecolor": GRID, "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2,
    "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True,
    "grid.color": GRID, "grid.linewidth": 0.8, "axes.axisbelow": True,
})


def title(ax, t, sub):
    ax.set_title(t, loc="left", fontsize=18, fontweight="bold", color=INK, pad=34)
    ax.text(0, 1.035, sub, transform=ax.transAxes, fontsize=12.5, color=INK2, va="bottom")


def source(fig, s):
    fig.text(0.012, -0.04, s, fontsize=9.5, color=MUTED)


def save(fig, name):
    fig.savefig(CH / name, dpi=200, bbox_inches="tight", pad_inches=0.3)
    plt.close(fig)
    print("saved", CH / name)


# ---------- 1. blind spot ----------
def chart_blind_spot():
    rv = pd.read_csv(OUT / "runway_validation.csv").set_index("score")
    mc = pd.read_csv(OUT / "mc_validation.csv").set_index("score")
    rows = [("Runway (judges' formula)", rv.loc["runway_base"], GRAY),
            ("Runway under EI 55% stress", rv.loc["runway_ei55"], GRAY),
            ("Monte Carlo on own history", mc.loc["mc_od_current"], BLUE),
            ("Early warning model", rv.loc["model_no_snapshot"], BLUE)]
    fig, ax = plt.subplots(figsize=(12, 5.6))
    y = np.arange(len(rows))[::-1]
    for yi, (lab, r, c) in zip(y, rows):
        ax.barh(yi, r.auc - 0.5, left=0.5, height=0.5, color=c)
        ax.plot([r.auc_lo, r.auc_hi], [yi, yi], color=INK, lw=1.5)
        ax.text(r.auc_hi + 0.008, yi, f"{r.auc:.2f}", va="center", fontsize=14, fontweight="bold", color=INK)
    ax.set_yticks(y, [r[0] for r in rows], fontsize=13.5, color=INK)
    ax.set_xlim(0.5, 1.0)
    ax.set_xlabel("AUC on held out 2024 to 2025 months (0.5 = coin flip, 1.0 = perfect); line = 95% CI")
    ax.grid(axis="y", visible=False)
    title(ax, "Runway alone misses most real overdrafts",
          "79% of customers who overdrew showed 12+ months of runway three months before.  "
          "Gray = judges' formula, blue = our engine")
    source(fig, "Test set: 48,400 account months, 96 first overdrafts. All scores computed with data up to the prediction month only.")
    save(fig, "1_blind_spot.png")


# ---------- 2. warning signal ----------
def chart_warning_signal():
    ms = pd.read_pickle(OUT / "monthly_full.pkl").sort_values(["account_id", "t"])
    g = ms.groupby("account_id")
    inc = g["money_in"].transform(lambda s: s.expanding().mean())
    ms["buffer"] = ms["min_balance"] / (inc + 1)                       # lowest balance, in months of income
    r3 = lambda c: g[c].transform(lambda s: s.rolling(3, min_periods=1).sum())
    ms["spend_ratio"] = r3("money_out") / (r3("money_in") + 1)          # 3 month spending / income
    ms = ms[ms["tenure"] >= 7]
    ms["k"] = (ms.event_t.dt.year - ms.t.dt.year) * 12 + (ms.event_t.dt.month - ms.t.dt.month)
    base = ms[ms.event_t.isna()]
    ev = ms[ms.k.between(1, 12)]
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.4), sharex=True)
    for ax, col, lab, fmt, c in [(axes[0], "buffer", "Lowest balance, in months of income", "{:.1f}", BLUE),
                                 (axes[1], "spend_ratio", "Spending / income, last 3 months", "{:.2f}", ORANGE)]:
        m = ev.groupby("k")[col].median().sort_index()
        ref = base[col].median()
        ax.axhline(ref, color=MUTED, lw=1.5, ls="--")
        ax.text(-12.4, ref, "customers who never overdrew", color=INK2, fontsize=11, va="bottom")
        ax.plot(-m.index, m.values, color=c, lw=2.5, marker="o", ms=6)
        for k in (12, 6, 1):
            ax.annotate(fmt.format(m.loc[k]), (-k, m.loc[k]), textcoords="offset points", xytext=(0, 9),
                        ha="center", fontsize=11.5, color=INK)
        ax.axvspan(-6.5, -0.5, color="#f0efec", zorder=0)
        ax.set_xticks([-12, -9, -6, -3, -1], ["12", "9", "6", "3", "1"])
        ax.set_xlim(-12.6, -0.4)
        ax.set_xlabel("Months before first overdraft")
        ax.set_title(lab, loc="left", fontsize=13.5, color=INK)
    for ax, top in ((axes[0], False), (axes[1], True)):
        lo, hi = ax.get_ylim()
        ax.text(-4.2 if top else -3.5, hi if top else lo, "warning window\n(last 6 months)", color=INK2,
                fontsize=11, ha="center", va="top" if top else "bottom")
    fig.suptitle("The warning signs start about 6 months out", x=0.065, ha="left", fontsize=18, fontweight="bold", y=1.04)
    fig.text(0.065, 0.965, "Median of the 288 customers who overdrew, by months before the event (dashed = everyone else)",
             fontsize=12.5, color=INK2)
    source(fig, "monthly_summary.csv; months 7+ of each account. Medians.")
    save(fig, "2_warning_signal.png")


# ---------- 3. fragility by age ----------
def chart_fragility_age():
    seg = pd.read_csv(OUT / "population_segments.csv")
    a = seg[seg.segment == "age_band"].set_index("group").loc[["under 25", "25 to 34", "35 to 49", "50 to 64", "65+"]]
    allrow = seg[seg.segment == "all"].iloc[0]
    fig, ax = plt.subplots(figsize=(12, 5.6))
    x = np.arange(len(a)); w = 0.36
    b1 = ax.bar(x - w / 2 - 0.01, a.short_ei55 * 100, w, color=GRAY, label="Judges' formula: EI = 55% of income")
    b2 = ax.bar(x + w / 2 + 0.01, a.short_ei_realistic * 100, w, color=BLUE, label="Realistic EI: $729/week cap, pension kept")
    for bars in (b1, b2):
        for r in bars:
            ax.text(r.get_x() + r.get_width() / 2, r.get_height() + 1.2, f"{r.get_height():.0f}%",
                    ha="center", fontsize=12, color=INK)
    labels = [f"{g}\n${inc / 1e3:.1f}k income\n{bm:.1f} mo buffer" for g, inc, bm in
              zip(a.index, a.median_income, a.median_buffer_months)]
    ax.set_xticks(x, labels, fontsize=11.5)
    ax.set_ylim(0, 75); ax.set_ylabel("Share with under 6 months of runway")
    ax.yaxis.set_major_formatter(lambda v, _: f"{v:.0f}%")
    ax.grid(axis="x", visible=False)
    ax.legend(loc="upper right", frameon=False, fontsize=12)
    title(ax, "The most fragile customers are middle aged, higher earners",
          f"Laid off on EI: share who run out within 6 months. All customers: {allrow.short_ei55:.0%} by the formula, "
          f"{allrow.short_ei_realistic:.0%} with the real EI cap")
    source(fig, "End of 2025 position, judges' definitions. EI maximum: canada.ca EI regular benefits (2026).")
    save(fig, "3_fragility_age.png")


# ---------- 4. borrowers ----------
def chart_borrowers():
    seg = pd.read_csv(OUT / "population_segments.csv")
    l = seg[seg.segment == "loan"].set_index("group").loc[["no loan", "repaid", "running OK", "running in debt", "defaulted"]]
    fig, ax = plt.subplots(figsize=(12, 4.8))
    y = np.arange(len(l))[::-1]
    cols = [GRAY, BLUE, BLUE, BLUE, BLUE]
    ax.barh(y, l.short_ei55 * 100, height=0.55, color=cols)
    for yi, v, n in zip(y, l.short_ei55 * 100, l.accounts):
        ax.text(v + 1, yi, f"{v:.0f}%", va="center", fontsize=13, fontweight="bold", color=INK)
    ax.set_yticks(y, [f"{g}  (n={n:,})" for g, n in zip(l.index, l.accounts)], fontsize=12.5, color=INK)
    ax.set_xlim(0, 100); ax.xaxis.set_major_formatter(lambda v, _: f"{v:.0f}%")
    ax.set_xlabel("Share with under 6 months of runway if laid off (EI 55%)")
    ax.grid(axis="y", visible=False)
    title(ax, "Borrowers in good standing are already twice as exposed",
          "Even loans that are 'running OK' leave 61% of borrowers unable to last 6 months on EI")
    source(fig, "loans.csv status joined to the judges' runway. Loan status is never a model input.")
    save(fig, "4_borrowers.png")


# ---------- 5. fan chart for one customer ----------
def chart_fan(acc=20111, months=12):
    p = E.profile(acc)
    s = E.Shock(income_mult=0.55, ei_months=6)
    sv = E.survival(p, s, T=months, N=5000)
    det = E.path(p, s, months)
    judge = E.apply_shock(p, s)
    f = sv["fan"]
    m = np.r_[0, f.month]
    start = p["B"]
    q = {c: np.r_[start, f[c]] for c in ["p05", "p25", "p50", "p75", "p95"]}
    fig, ax = plt.subplots(figsize=(12, 5.8))
    ax.fill_between(m, q["p05"] / 1e3, q["p95"] / 1e3, color=SEQ[0], lw=0, label="90% of simulated paths")
    ax.fill_between(m, q["p25"] / 1e3, q["p75"] / 1e3, color=SEQ[1], lw=0, label="middle 50%")
    ax.plot(m, q["p50"] / 1e3, color=SEQ[3], lw=2, label="median path")
    ax.plot(np.r_[0, det.month], np.r_[start, det.balance] / 1e3, color=ORANGE, lw=2.5, ls="--",
            label="judges' straight line (uses median income and spending)")
    ax.axhline(0, color=INK, lw=1.2)
    ax.axvline(6, color=MUTED, lw=1, ls=":")
    ax.text(6.1, ax.get_ylim()[1] * 0.92, "EI ends", color=INK2, fontsize=11.5)
    ax.set_xlim(0, months); ax.set_xticks(range(0, months + 1))
    ax.set_xlabel("Months after layoff"); ax.set_ylabel("Balance ($ thousands)")
    ax.legend(loc="lower left", frameon=False, fontsize=11.5)
    title(ax, f"Account {acc}: laid off, EI at 55% for 6 months",
          f"Formula runway {judge['runway_label']} months.  Chance of no overdraft: "
          f"{sv['p_survive_3']:.0%} at 3 months, {sv['p_survive_6']:.0%} at 6 months (conservative: past swings repeat)")
    source(fig, "5,000 block bootstrap paths of this customer's own months (income fixed at EI, spending resampled).")
    save(fig, f"5_fan_{acc}.png")


if __name__ == "__main__":
    chart_blind_spot()
    chart_warning_signal()
    chart_fragility_age()
    chart_borrowers()
    chart_fan()
