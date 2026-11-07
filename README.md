# Resilience Engine

Sheridan Datathon 2026, Tangerine track "Resilience Rewired".

For any customer: **know them** (income, bills, buffer, runway), **apply a shock** (job loss on EI, higher bills, a surprise repair, or any mix), and **make a plan** (emergency fund goal plus a backup plan), all from their own numbers. Every headline number follows the judges' definitions exactly. On top of that we add a validated early warning model and a Monte Carlo survival probability.

**Biggest finding:** the runway formula misses most real overdrafts. 79% of customers who later overdrew showed 12+ months of runway three months before (AUC 0.65). Our early warning model reaches AUC 0.955 on held out months.

## Run the app

Needs **Python 3.11 or newer** (pandas 3 requires it). The competition data is not in the repo.

```bash
git clone https://github.com/sadiqbaksh2005-collab/resilience-engine.git
cd resilience-engine
python3 -m venv .venv && source .venv/bin/activate      # optional but recommended
pip install -r requirements.txt
mkdir data    # copy the 9 competition CSVs into ./data
streamlit run app.py
```

The app looks for the CSVs in `RESILIENCE_DATA` (if set), then `./data`, then `~/Downloads/data`.
Everything else the app needs (model, scores, demo accounts, charts) is already in `outputs/`.

## Rebuild everything from the raw data

```bash
python3 features.py          # monthly feature panel, label, time split
python3 baselines.py         # rule / logistic regression / gradient boosting baselines
python3 test_engine.py       # judge definitions on all 4,500 accounts + edge case and monotonicity tests
python3 validate_runway.py   # runway vs real overdrafts and loan defaults (as of each month)
python3 validate_mc.py       # Monte Carlo out of sample check
python3 population.py        # population stress test + fairness audit
python3 robustness.py        # headline across 3, 6, 12 month horizons
python3 early_warning.py     # validated model, bank view, latest features
python3 demo_accounts.py     # demo accounts per story
python3 charts.py            # pitch charts -> outputs/charts
```

## Files

| File | What it does |
|---|---|
| `engine.py` | Profile (I, F, S, B, H, P), shocks, judges' runway, path, survival, plans, drivers, warnings |
| `montecarlo.py` | Block bootstrap survival simulation |
| `app.py` | Streamlit app |
| `features.py`, `baselines.py`, `early_warning.py` | Early warning model |
| `validate_*.py`, `robustness.py`, `population.py` | Evidence |
| `hmm.py` | Hidden Markov model experiment (secondary) |

## Data and sources

- PKDD'99 Discovery Challenge Financial dataset (real anonymized bank data), CTU Prague Relational Dataset Repository: https://relational.fel.cvut.cz/dataset/Financial
- Employment Insurance regular benefits, 2026 maximum ($729/week): https://www.canada.ca/en/services/benefits/ei/ei-regular-benefit/benefit-amount.html
