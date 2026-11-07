"""
Pick one demo account per story (deterministic rules), with its numbers under the
three example twists. Output: outputs/demo_accounts.csv
"""
from pathlib import Path
import pandas as pd
import engine as E

OUT = Path(__file__).parent / "outputs"
pop = pd.read_csv(OUT / "population_accounts.csv")
bv = pd.read_csv(OUT / "bank_view.csv")[["account_id", "risk", "flag", "past_overdraft", "why"]]
d = pop.merge(bv, on="account_id")
d["pension"] = [E.profile(a)["cat"]["pension_in"] for a in d.account_id]
d["flex"] = [E.profile(a)["flex"] for a in d.account_id]
clean = d[d.past_overdraft.isna() & (d.S > 1000) & (d.I > 1000) & (d.B > 0)]   # real, active accounts

stories = {
    "Blind spot: formula says safe, model flags": (
        clean[clean.flag & (clean.runway_base >= 120)].sort_values("risk", ascending=False),
        "No shortfall by the formula, yet the early warning model flags them now"),
    "Fragile middle aged borrower": (
        clean[(clean.age_band == "35 to 49") & (clean.loan == "running OK") & (clean.H > 0) & (clean.flex > 0)
              & (clean.runway_ei55 < 3)].sort_values("I", ascending=False),
        "High income, thin buffer, loan payment: the segment our population test found most exposed"),
    "Retiree the formula misjudges": (
        clean[(clean.age_band == "65+") & (clean.pension > 0) & (clean.runway_ei55 < 6)
              & (clean.runway_ei_realistic >= 120)].sort_values("runway_ei55"),
        "The layoff twist cuts their pension by 45%; with the pension kept they are fine"),
    "Bills shock does nothing (H = 0)": (
        clean[(clean.H == 0) & (clean.runway_ei55 < 6) & (clean.runway_ei55 > 1) & (clean.ei_cap_binds == 1)]
        .sort_values("runway_ei55"),
        "No bills paid from this account, so +15% bills changes nothing; EI cap shortens runway"),
    "Already in deficit, living off savings": (
        clean[(clean.I < clean.S) & (clean.buffer_months > 6) & (clean.runway_base < 24)]
        .sort_values("buffer_months", ascending=False),
        "Spends more than they earn every month; the buffer is quietly running down"),
    "Comfortable saver": (
        clean[(clean.runway_ei55 >= 120)].sort_values("buffer_months", ascending=False),
        "Survives a layoff with no shortfall at all: the control case"),
}

rows = []
for name, (cands, story) in stories.items():
    if cands.empty:
        print("no candidate for", name); continue
    a = int(cands.iloc[0].account_id)
    p = E.profile(a)
    r = {k: E.apply_shock(p, s)["runway_label"] for k, s in E.PRESETS.items()}
    real = E.apply_shock(p, E.Shock(income_mult=0.55, ei_months=6, ei_cap=True, keep_pension=True))["runway_label"]
    sv = E.survival(p, E.PRESETS["Laid off, EI 55% for 6 months"], T=6)
    rows.append(dict(story=name, account_id=a, why_it_matters=story, I=round(p["I"]), S=round(p["S"]), B=round(p["B"]),
                     H=round(p["H"]), P=round(p["P"]), runway_now=p["runway"][1],
                     layoff=r["Laid off, EI 55% for 6 months"], furnace=r["Furnace repair = 1 month of spending"],
                     layoff_bills=r["Job loss (EI 55%) + bills up 15%"], layoff_realistic=real,
                     p_survive6_layoff=round(sv["p_survive_6"], 2),
                     ew_risk=round(float(cands.iloc[0].risk), 3), n_candidates=len(cands)))
out = pd.DataFrame(rows)
out.to_csv(OUT / "demo_accounts.csv", index=False)
pd.set_option("display.width", 250); pd.set_option("display.max_colwidth", 60)
print(out.drop(columns="why_it_matters").to_string(index=False))

# ---- model showcase accounts (out of sample: first overdraft in 2025, model trained on data <= 2023-09) ----
# Chosen from outputs/risk_history.pkl: flagged continuously in the months before the first overdraft.
#   73023: 5 months sustained warning, zero false alarms in the 57 earlier months
#   90939: 6 months sustained warning while the judges' runway said "no shortfall (120+)"
extra = []
for a, story, why in [
        (73023, "Early warning: flagged 5 months ahead", "Model flagged 5 months before the first overdraft, no false alarms before"),
        (90939, "Early warning while runway said safe", "Flagged 6 months ahead while the formula showed no shortfall")]:
    p = E.profile(a)
    r = {k: E.apply_shock(p, s)["runway_label"] for k, s in E.PRESETS.items()}
    extra.append(dict(story=story, account_id=a, why_it_matters=why, I=round(p["I"]), S=round(p["S"]), B=round(p["B"]),
                      H=round(p["H"]), P=round(p["P"]), runway_now=p["runway"][1],
                      layoff=r["Laid off, EI 55% for 6 months"], furnace=r["Furnace repair = 1 month of spending"],
                      layoff_bills=r["Job loss (EI 55%) + bills up 15%"]))
out = pd.concat([pd.DataFrame(extra), out], ignore_index=True)
out.to_csv(OUT / "demo_accounts.csv", index=False)
print("added model showcase accounts:", [e["account_id"] for e in extra])
