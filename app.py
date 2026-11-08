"""
Resilience engine: Streamlit app.
Run:  streamlit run app.py
"""
from dataclasses import astuple
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import streamlit as st

import engine as E

OUT = Path(__file__).parent / "outputs"
st.set_page_config(page_title="Resilience Engine", layout="wide")
st.markdown("""<style>
section[data-testid="stSidebar"] .stButton button {padding: 0.1rem 0.6rem; min-height: 1.9rem; font-size: 0.85rem;
  justify-content: flex-start;}
section[data-testid="stSidebar"] .stButton {margin-bottom: -0.55rem;}
div[data-testid="stMetricValue"] {font-size: 1.6rem;}
</style>""", unsafe_allow_html=True)

# palette (validated reference palette, light mode)
INK, INK2, MUTED, GRID = "#0b0b0b", "#52514e", "#8a8984", "#e6e5e1"
BLUE, ORANGE, GRAY = "#2a78d6", "#eb6834", "#b4b3ad"
SEQ = ["#cde2fb", "#9ec5f4", "#5598e7", "#256abf"]
plt.rcParams.update({"axes.spines.top": False, "axes.spines.right": False, "axes.grid": True,
                     "grid.color": GRID, "axes.edgecolor": GRID, "axes.axisbelow": True,
                     "xtick.color": INK2, "ytick.color": INK2, "axes.labelcolor": INK2, "font.size": 10})

money = lambda x: f"${x:,.0f}"


# ---------------- cached data ----------------
@st.cache_data
def load_profile(acc):
    return E.profile(acc)


@st.cache_data
def load_ids():
    return E.account_ids()


@st.cache_data
def load_csv(name):
    f = OUT / name
    return pd.read_csv(f) if f.exists() else None


@st.cache_resource
def load_model():
    import pickle
    f = OUT / "ew_model.pkl"
    return pickle.load(open(f, "rb")) if f.exists() else None


@st.cache_data
def load_latest_features():
    return pd.read_pickle(OUT / "latest_features.pkl")


@st.cache_data
def load_risk_history():
    f = OUT / "risk_history.pkl"
    return pd.read_pickle(f) if f.exists() else None


@st.cache_data
def load_market():
    import portfolio as PF
    return PF.load_market() if PF.MARKET.exists() else None


@st.cache_data
def load_migration():
    import json
    f = OUT / "migration.json"
    return json.loads(f.read_text()) if f.exists() else None


@st.cache_data
def run_invest(acc, tolerance, years, flag):
    import portfolio as PF
    return PF.invest_plan(load_profile(acc), tolerance, flag, years=years, r=load_market())


@st.cache_data
def run_survival(acc, shock_tuple, T=12):
    return E.survival(load_profile(acc), E.Shock(*shock_tuple), T=T, N=5000)


# ---------------- sidebar: customer ----------------
DEFAULTS = dict(inc_pct=100, ei_months=6, income_after_pct=0, income_change=0, bills_pct=0, bills_abs=0,
                spending_pct=0, extra_monthly=0, loan_pct=0, one_off=0, one_off_S=0.0,
                ei_cap=False, keep_pension=False)
PRESETS = {
    "Laid off: EI 55% for 6 months": dict(inc_pct=55, ei_months=6),
    "Furnace: 1 month of spending": dict(one_off_S=1.0),
    "Laid off + bills up 15%": dict(inc_pct=55, ei_months=6, bills_pct=15),
}
for k, v in DEFAULTS.items():
    st.session_state.setdefault(k, v)


def set_preset(values):
    for k, v in DEFAULTS.items():
        st.session_state[k] = v
    for k, v in values.items():
        st.session_state[k] = v


LIMITS = dict(inc_pct=(0, 150), ei_months=(0, 24), income_after_pct=(0, 100), bills_pct=(0, 100),
              spending_pct=(0, 50), loan_pct=(0, 100), one_off_S=(0.0, 6.0))


def read_card():
    from twist import parse
    text = st.session_state.get("twist_text", "")
    params, understood, unread = parse(text)
    notes = list(understood)
    if "one_off_months_I" in params:                       # needs this customer's income
        k = params.pop("one_off_months_I")
        I = load_profile(st.session_state["acc"])["I"]
        params["one_off"] = params.get("one_off", 0) + k * I
        i = next((j for j, n in enumerate(notes) if "of typical income" in n), len(notes) - 1)
        notes[i] += f" (= ${k * I:,.0f} for this customer)"
    for key, (lo, hi) in LIMITS.items():
        if key in params and not lo <= params[key] <= hi:
            notes.append(f"{key} {params[key]} is outside the slider range, set to {min(max(params[key], lo), hi)}")
            params[key] = min(max(params[key], lo), hi)
    for key in ("inc_pct", "ei_months", "income_after_pct", "bills_pct", "spending_pct", "loan_pct"):
        if key in params:
            params[key] = int(round(params[key]))
    if "one_off_S" in params:
        params["one_off_S"] = float(round(params["one_off_S"] * 2) / 2)
    for key in ("income_change", "bills_abs", "extra_monthly", "one_off"):
        if key in params:
            params[key] = int(round(params[key]))
    set_preset(params)
    st.session_state["twist_notes"] = (notes, unread, bool(params))


ids = load_ids()
demos = load_csv("demo_accounts.csv")
id_set = set(ids)
demo_map = {} if demos is None else dict(zip(demos.account_id.astype(int), demos.story))
st.session_state.setdefault("acc", 20111)


_bank = load_csv("bank_view.csv")
risk_of = {} if _bank is None else dict(zip(_bank.account_id.astype(int), _bank.risk))
STATUS = {
    "At risk now": "Never overdrawn, but the early warning model flags them for the next 3 months.",
    "Not at risk": "Never overdrawn and not flagged.",
    "Already overdrew": "Has overdrawn before; the model scores only customers who never have.",
}
_status = {}
if _bank is not None:
    for r in _bank.itertuples():
        _status[int(r.account_id)] = ("Already overdrew" if isinstance(r.past_overdraft, str)
                                      else "At risk now" if r.flag else "Not at risk")


def status_of(a):
    return _status.get(a, "Not at risk")


def choose(a):
    st.session_state["acc"] = int(a)


def pick_random():
    choose(np.random.default_rng().choice(ids))


@st.dialog("Choose a customer", width="large")
def account_picker():
    cur = st.session_state["acc"]
    st.markdown("**Type an account_id** (fastest when a judge reads one out)")
    q = st.text_input("account_id", placeholder="e.g. 29010", label_visibility="collapsed").strip()
    if q:
        if q.isdigit() and int(q) in id_set:
            if st.button(f"Open account {q}", type="primary"):
                choose(q); st.rerun()
        else:
            matches = [a for a in ids if str(a).startswith(q)][:40]
            if matches:
                hit = st.pills("Matches", matches, key="pick_search")
                if hit and hit != cur:
                    choose(hit); st.rerun()
            else:
                st.error(f"No account starts with {q}")

    st.markdown("**Featured demo accounts**")
    if demos is not None:
        cols = st.columns(2)
        for i, r in enumerate(demos.itertuples()):
            if cols[i % 2].button(f"{r.account_id}  ·  {r.story}", key=f"demo_{r.account_id}", width="stretch"):
                choose(r.account_id); st.rerun()

    if st.button("Random account", key="dlg_random"):
        pick_random(); st.rerun()

    st.markdown("**Browse accounts**")
    counts = {k: sum(1 for a in ids if status_of(a) == k) for k in STATUS}
    opts = ["All"] + [k for k in STATUS]
    show = st.segmented_control("Show", opts, default="All", key="pick_status",
                                format_func=lambda k: f"{k} ({len(ids) if k == 'All' else counts[k]:,})")
    pool = ids if show in (None, "All") else [a for a in ids if status_of(a) == show]
    label = lambda a: f"{a}  {risk_of.get(a, 0):.1%}" + (" (demo)" if a in demo_map else "")
    if show and show != "All":
        st.caption(STATUS[show])
    if len(pool) <= 150:
        # short list: show everything, highest risk first
        pool = sorted(pool, key=lambda a: -risk_of.get(a, 0))
        st.caption(f"{len(pool)} accounts, highest 3 month risk first. The % is the early warning score.")
        hit = st.pills("Account", pool, key=f"pick_flat_{show}", format_func=label)
        if hit and hit != cur:
            choose(hit); st.rerun()
    else:
        digits = sorted({str(a)[0] for a in pool})
        lead = st.pills("First digit", [f"{d}xxxx" for d in digits], default=f"{digits[0]}xxxx", key=f"pick_lead_{show}")
        if lead:
            d = lead[0]
            blocks = sorted({str(a)[:2] for a in pool if str(a)[0] == d})
            blk = st.pills("Range", [f"{b}xxx" for b in blocks], default=f"{blocks[0]}xxx", key=f"pick_blk_{show}_{d}")
            if blk:
                group = [a for a in pool if str(a)[:2] == blk[:2]]
                st.caption(f"{len(group)} accounts in {blk}. The % is the early warning score; demo accounts are marked.")
                hit = st.pills("Account", group, key=f"pick_{show}_{blk}", format_func=label)
                if hit and hit != cur:
                    choose(hit); st.rerun()

st.sidebar.title("Resilience Engine")
with st.sidebar:
    st.caption("Pick a customer, apply a twist, read the result at the top of the page. "
               "Headline runway uses the judges' definitions exactly; realistic EI and the simulation are our extensions.")
    st.subheader("1 · Customer")
    acc = st.session_state.get("acc", 20111)
    demo_story = demo_map.get(acc)
    st.markdown(f"**Account {acc}**" + (f"  \n<span style='color:{INK2};font-size:0.85rem'>Demo: {demo_story}</span>"
                                         if demo_story else ""), unsafe_allow_html=True)
    b1, b2 = st.columns(2)
    if b1.button("Choose account", width="stretch", type="primary"):
        account_picker()
    b2.button("Random account", width="stretch", on_click=pick_random)

    st.subheader("2 · Twist")
    st.text_area("Read the twist card", key="twist_text", height=80,
                 placeholder="e.g. They lose their job (EI at 55%) and their household bills go up 15%")
    st.button("Apply twist card", on_click=read_card, width="stretch", type="primary")
    if "twist_notes" in st.session_state:
        notes, unread, any_hit = st.session_state["twist_notes"]
        if any_hit:
            st.markdown("<span style='font-size:0.85rem'>Understood as:<br>" +
                        "<br>".join(f"&nbsp;&nbsp;{n}" for n in notes) + "</span>", unsafe_allow_html=True)
        else:
            st.warning("Couldn't read a twist from that text. Use the buttons or sliders below.")
        for u in unread:
            st.warning(f"Not understood: \"{u}\". Set it with the sliders.")
    st.caption("Or one click:")
    for name, vals in PRESETS.items():
        st.button(name, on_click=set_preset, args=(vals,), width="stretch")
    st.button("Reset (no shock)", on_click=set_preset, args=({},), width="stretch")
    st.write("")

    changed = lambda *keys: any(st.session_state[k] != DEFAULTS[k] for k in keys)
    with st.expander("Income", expanded=changed("inc_pct", "income_change")):
        st.slider("Income while shocked (% of usual)", 0, 150, key="inc_pct", step=1)
        st.slider("For how many months (0 = indefinitely)", 0, 24, key="ei_months")
        st.slider("Income after that (% of usual)", 0, 100, key="income_after_pct", step=1)
        st.number_input("Other income change ($/month, e.g. -2000)", step=100, key="income_change")
    with st.expander("Costs", expanded=changed("bills_pct", "bills_abs", "spending_pct", "extra_monthly", "loan_pct")):
        st.slider("Household bills / rent up (%)", 0, 100, key="bills_pct", step=1)
        st.number_input("Household bills / rent up ($/month)", min_value=0, step=50, key="bills_abs")
        st.slider("All spending up (%, e.g. inflation)", 0, 50, key="spending_pct")
        st.number_input("New recurring cost ($/month)", min_value=0, step=50, key="extra_monthly")
        st.slider("Loan payment up (%)", 0, 100, key="loan_pct", step=1)
    with st.expander("One off cost", expanded=changed("one_off", "one_off_S")):
        st.number_input("Surprise cost ($)", min_value=0, step=500, key="one_off")
        st.slider("Surprise cost (months of typical spending)", 0.0, 6.0, key="one_off_S", step=0.5)
    with st.expander("Realism options (our extension)", expanded=changed("ei_cap", "keep_pension")):
        st.checkbox("Cap EI at $729/week (canada.ca, 2026)", key="ei_cap")
        st.checkbox("Keep pension income during a layoff", key="keep_pension")

ss = st.session_state
shock = E.Shock(
    income_mult=ss.inc_pct / 100, income_change=float(ss.income_change),
    ei_months=(ss.ei_months or None) if ss.inc_pct != 100 else None, income_after=ss.income_after_pct / 100,
    bills_pct=ss.bills_pct / 100, bills_abs=float(ss.bills_abs), spending_pct=ss.spending_pct / 100,
    extra_monthly=float(ss.extra_monthly), loan_payment_pct=ss.loan_pct / 100,
    one_off=float(ss.one_off), one_off_months_S=float(ss.one_off_S), ei_cap=ss.ei_cap, keep_pension=ss.keep_pension)
judge_shock = E.Shock(**{**shock.__dict__, "ei_cap": False, "keep_pension": False})
real_shock = E.Shock(**{**shock.__dict__, "ei_cap": True, "keep_pension": True})

p = load_profile(acc)
r_j = E.apply_shock(p, judge_shock)
r_r = E.apply_shock(p, real_shock)
bank = load_csv("bank_view.csv")
ew = bank.set_index("account_id").loc[acc] if bank is not None and acc in set(bank.account_id) else None

sv = run_survival(acc, astuple(shock))
fmt_rw = lambda lab: lab if "120+" in lab else f"{lab} months"

st.title(f"Account {acc}")
st.markdown(f"**Twist:** {shock.describe()}  \n"
            f"<span style='color:{INK2}'>{p['n_months']} months of history, {p['first_month']} to {p['last_month']}</span>",
            unsafe_allow_html=True)
top = st.columns(6)
top[0].metric("Runway now", fmt_rw(p["runway"][1]), help="Judges' formula, no shock")
top[1].metric("Runway after twist", fmt_rw(r_j["runway_label"]),
              delta=None if ("120+" in r_j["runway_label"] or r_j["runway_change"] == 0) else f"{r_j['runway_change']:+.1f} months",
              help="Judges' formula: (B - one off) / (spending after - income after)")
top[2].metric("Realistic EI runway", fmt_rw(r_r["runway_label"]), help="EI capped at $729/week, pension kept (our extension)")
top[3].metric("Monthly gap", money(r_j["gap"]), help="Spending after the twist minus income after the twist")
top[4].metric("No overdraft, 6 months", f"{sv['p_survive_6']:.0%}",
              help="Monte Carlo on the customer's own months (conservative)")
if ew is None:
    ew_txt = "n/a"
elif isinstance(ew["past_overdraft"], str):
    ew_txt = "Overdrew before"
else:
    ew_txt = "Flagged" if ew["flag"] else "Not flagged"
top[5].metric("Early warning", ew_txt, help="Risk of a first overdraft in the next 3 months, model validated out of sample")
st.divider()

tab_c, tab_s, tab_v, tab_st, tab_m, tab_f = st.tabs(["Customer", "Twist and plan", "Investing", "States", "Model", "Proof"])
tab_p, tab_b, tab_i = tab_s, tab_m, tab_f        # merged: plan under the twist, bank view under the model, calculations under findings

# ---------------- customer ----------------
with tab_c:
    m = st.columns(7)
    m[0].metric("Income I", money(p["I"]), help="Median of money_in")
    m[1].metric("Fixed costs F", money(p["F"]), help="Median of household_bills + insurance + loan_payments")
    m[2].metric("Spending S", money(p["S"]), help="Median of money_out")
    m[3].metric("Buffer B", money(p["B"]), help="Mean avg_daily_balance, last 6 months")
    m[4].metric("Household bills H", money(p["H"]))
    m[5].metric("Loan payment P", money(p["P"]), help="Running at end of 2025")
    m[6].metric("Runway now", p["runway"][1] + ("" if "120+" in p["runway"][1] else " mo"))

    left, right = st.columns([3, 2])
    with left:
        h = p["history"].copy()
        fig, ax = plt.subplots(figsize=(8, 3.2))
        ax.plot(range(len(h)), h["month_end_balance"] / 1e3, color=BLUE, lw=2, label="month end balance")
        ax.plot(range(len(h)), h["min_balance"] / 1e3, color=GRAY, lw=1.5, label="lowest balance in month")
        ax.axhline(0, color=INK, lw=1)
        step = max(1, len(h) // 8)
        ax.set_xticks(range(0, len(h), step), h["month"].iloc[::step], rotation=0, fontsize=8)
        ax.set_ylabel("$ thousands"); ax.legend(frameon=False, fontsize=9, loc="upper left")
        ax.set_title("Balance history", loc="left", fontsize=11, color=INK)
        st.pyplot(fig, width="stretch"); plt.close(fig)

        fig, ax = plt.subplots(figsize=(8, 2.6))
        x = np.arange(len(h))
        ax.bar(x - 0.2, h["money_in"] / 1e3, 0.4, color=BLUE, label="money in")
        ax.bar(x + 0.2, h["money_out"] / 1e3, 0.4, color=ORANGE, label="money out")
        ax.set_xticks(range(0, len(h), step), h["month"].iloc[::step], fontsize=8)
        ax.set_ylabel("$ thousands"); ax.legend(frameon=False, fontsize=9, loc="upper left")
        ax.set_title("Money in and out each month", loc="left", fontsize=11, color=INK)
        st.pyplot(fig, width="stretch"); plt.close(fig)
    with right:
        st.markdown("#### Early warning")
        if ew is not None:
            if isinstance(ew["past_overdraft"], str):
                st.warning(f"Already overdrew (first in {ew['past_overdraft']}). Risk score {ew['risk']:.1%}.")
            elif ew["flag"]:
                st.error(f"Flagged: {ew['risk']:.1%} chance of a first overdraft in the next 3 months "
                         f"(alert threshold 2.9%). Why: {ew['why']}.")
            else:
                st.success(f"Not flagged: {ew['risk']:.2%} chance of a first overdraft in the next 3 months.")
            st.caption("Logistic regression, validated out of sample: AUC 0.955, catches 72% of first overdrafts "
                       "with a median 3 months warning, alerts on 2.2% of customers. No demographics used.")
        st.markdown("#### Where the money goes")
        dr = E.drivers(p, shock)
        dr = dr[dr.monthly > 0]
        st.dataframe(dr.assign(monthly=dr.monthly.map(money), share_of_spending=dr.share_of_spending.map("{:.0%}".format),
                               runway_gain_from_20pct_cut=dr.runway_gain_from_20pct_cut.map(
                                   lambda v: "" if pd.isna(v) else f"+{v:.1f} mo"))
                     .rename(columns={"runway_gain_from_20pct_cut": "20% cut buys"}),
                     hide_index=True, width="stretch")
        st.caption("Typical month per category (medians, so they need not add up to S).")

# ---------------- shock ----------------
with tab_s:
    st.markdown("### What the twist does")
    with st.expander("Show the math", expanded=True):
        st.latex(
            r"\text{runway}=\frac{B-C}{S'-I'}=\frac{%s-%s}{%s-%s}=\frac{%s}{%s}"
            % (f"{p['B']:,.0f}", f"{r_j['one_off']:,.0f}", f"{r_j['spending']:,.0f}", f"{r_j['income']:,.0f}",
               f"{r_j['buffer_left']:,.0f}", f"{r_j['gap']:,.0f}")
            + (r"\Rightarrow \textbf{%s}" % r_j["runway_label"].replace("(", "").replace(")", "")))
        st.caption(f"I′ = {money(r_j['income'])}/month, S′ = {money(r_j['spending'])}/month, "
                   f"one off C = {money(r_j['one_off'])}. Capped at 120 months; 0 if the one off cost exceeds B.")

    for w in E.warnings(p, judge_shock):
        st.info(w)

    T = 12
    det = E.path(p, shock, T)
    f = sv["fan"]
    start = p["B"] - r_j["one_off"]
    mm = np.r_[0, f.month]
    q = {k: np.r_[start, f[k]] / 1e3 for k in ["p05", "p25", "p50", "p75", "p95"]}
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.fill_between(mm, q["p05"], q["p95"], color=SEQ[0], lw=0, label="90% of simulated paths")
    ax.fill_between(mm, q["p25"], q["p75"], color=SEQ[1], lw=0, label="middle 50%")
    ax.plot(mm, q["p50"], color=SEQ[3], lw=2, label="median simulated path")
    ax.plot(np.r_[0, det.month], np.r_[start, det.balance] / 1e3, color=ORANGE, lw=2.5, ls="--",
            label="straight line (median income and spending)")
    ax.axhline(0, color=INK, lw=1)
    if shock.ei_months and shock.income_mult != 1:
        ax.axvline(shock.ei_months, color=MUTED, lw=1, ls=":")
        ax.text(shock.ei_months + 0.1, ax.get_ylim()[1] * 0.9, "reduced income ends", color=INK2, fontsize=9)
    ax.set_xlim(0, T); ax.set_xticks(range(T + 1))
    ax.set_xlabel("Months after the shock"); ax.set_ylabel("Balance ($ thousands)")
    ax.legend(frameon=False, fontsize=9, loc="lower left")
    ax.set_title("What happens month by month", loc="left", fontsize=12, color=INK)
    st.pyplot(fig, width="stretch"); plt.close(fig)
    st.caption(f"Chance of no overdraft: {sv['p_survive_3']:.0%} at 3 months, {sv['p_survive_6']:.0%} at 6, "
               f"{sv['p_survive_12']:.0%} at 12. 5,000 paths replaying this customer's own months; conservative, "
               f"because real customers cut back when money gets tight.")

# ---------------- plan ----------------
with tab_p:
    st.divider()
    st.markdown("### The plan")
    fund_m = st.radio("Emergency fund target", [3, 6], horizontal=True, format_func=lambda v: f"{v} months of spending")
    pl = E.plans(p, shock, fund_months=fund_m)
    g, b = pl["goal"], pl["backup"]
    left, right = st.columns(2)
    with left:
        st.markdown("#### Goal plan: build the emergency fund")
        st.progress(float(g["funded_pct"]), text=f"{g['funded_pct']:.0%} funded: {money(g['current'])} of {money(g['target'])}")
        st.write(g["summary"])
    with right:
        st.markdown("#### Backup plan: if the shock hits")
        st.write(b["summary"])
        if b["options"]:
            opt = pd.DataFrame(b["options"])
            st.dataframe(opt.assign(monthly_saving=opt.monthly_saving.map(money))[["action", "monthly_saving", "runway_label"]]
                         .rename(columns={"monthly_saving": "frees up", "runway_label": "runway (months)"}),
                         hide_index=True, width="stretch")
        st.caption(f"Baseline runway under this twist: {b['base_runway']}. All numbers from this customer's own medians.")


# ---------------- model ----------------
with tab_m:
    M, RH = load_model(), load_risk_history()
    if M is None or RH is None:
        st.warning("Run early_warning.py and the risk history step first.")
    else:
        met = M["metrics"]
        st.markdown("#### Early warning model")
        st.markdown(
            "Predicts the chance that a customer who has **never overdrawn** has their **first overdraft in the next 3 months**. "
            "It reads the customer's own monthly pattern: buffer relative to income, spending relative to income, "
            "how the balance moves against their own normal, volatility, loan burden. "
            "It never uses age, gender, region, loan status or the end of data standing orders snapshot. "
            "Logistic regression, trained on months up to Sept 2023, alert threshold fixed on Oct 2023 to Sept 2024, "
            "and scored once on Oct 2024 to Sept 2025.")
        k = st.columns(5)
        k[0].metric("AUC, test period", "0.955", delta="+0.30 vs runway (0.65)")
        k[1].metric("First overdrafts caught", f"{met['test_event_recall']:.0%}", help="Flagged 1 to 3 months ahead")
        k[2].metric("Median warning", f"{met['test_median_lead']:.0f} months")
        k[3].metric("Customers alerted", f"{met['test_alert_rate']:.1%}", help="Share of non event months flagged")
        k[4].metric("Precision per alert", f"{met['test_precision']:.0%}", delta=f"vs {met['test_base_rate']:.1%} base rate",
                    delta_color="off")

        h = RH[RH.account_id == acc].sort_values("t").reset_index(drop=True)
        if h.empty:
            st.info("This account has under 7 months of history, so the model does not score it.")
        else:
            thr = M["threshold"]
            x = np.arange(len(h))
            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 5.6), sharex=True, gridspec_kw=dict(height_ratios=[3, 2]))
            n_train = int((h.period == "train").sum())
            for ax in (ax1, ax2):
                if n_train:
                    ax.axvspan(-0.5, n_train - 0.5, color="#f0efec", zorder=0)
            neg = h.min_balance < 0
            fe = int(np.argmax(neg.to_numpy())) if neg.any() else None
            # the model only covers customers who have never overdrawn: stop the line before the first overdraft
            risk = h.risk.where(np.arange(len(h)) < fe) if fe is not None else h.risk
            ax1.plot(x, risk * 100, color=BLUE, lw=2, label="model risk, next 3 months")
            hi = (risk >= thr).to_numpy()
            ax1.scatter(x[hi], risk[hi] * 100, color=ORANGE, s=28, zorder=3, label="flagged")
            ax1.axhline(thr * 100, color=MUTED, ls="--", lw=1.2)
            ax1.text(len(h) - 0.5, thr * 100, f" alert threshold {thr:.1%}", color=INK2, fontsize=8.5, va="bottom", ha="right")
            ax1.set_yscale("log")
            ax1.yaxis.set_major_formatter(lambda v, _: f"{v:g}%")
            ax1.set_ylabel("Risk (log scale)")
            if fe is not None:
                for ax in (ax1, ax2):
                    ax.axvline(fe, color="#d03b3b", lw=1.5)
                ax1.text(fe, ax1.get_ylim()[1], " first overdraft\n (model stops here)", color="#d03b3b", fontsize=9, va="top")
                lead = 0
                for j in range(fe - 1, -1, -1):
                    if h.risk.iloc[j] >= thr:
                        lead += 1
                    else:
                        break
            if n_train:
                ax1.text(0, ax1.get_ylim()[1], " model trained on this period", color=INK2, fontsize=8.5, va="top")
                ax1.text(n_train, ax1.get_ylim()[1], " never seen by the model", color=INK2, fontsize=8.5, va="top")
            ax1.legend(frameon=False, fontsize=8.5, loc="center left")
            ax1.set_title("This customer's risk, month by month", loc="left", fontsize=12, color=INK)
            ax2.plot(x, h.runway_asof.clip(upper=120), color=GRAY, lw=2)
            ax2.set_ylim(0, 125); ax2.set_yticks([0, 12, 60, 120], ["0", "12", "60", "120+"])
            ax2.set_ylabel("Runway (months)")
            ax2.set_title("Judges' runway at the time (no shock)", loc="left", fontsize=10, color=INK2)
            step = max(1, len(h) // 10)
            ax2.set_xticks(x[::step], h.month.iloc[::step], fontsize=8)
            fig.tight_layout()
            st.pyplot(fig, width="stretch"); plt.close(fig)
            if fe is not None:
                when = "out of sample" if h.period.iloc[fe] != "train" else "inside the training period"
                if lead:
                    st.success(f"The model flagged this customer continuously for the {lead} month(s) before their first "
                               f"overdraft in {h.month.iloc[fe]} ({when}).")
                else:
                    st.warning(f"The model did not flag this customer in the month before their first overdraft "
                               f"in {h.month.iloc[fe]} ({when}). It catches about 72% of first overdrafts.")

            # what drives the latest score
            from early_warning import contributions
            row = load_latest_features()
            row = row[row.account_id == acc]
            C = pd.Series(contributions(M["model"], row[M["features"]])[0], index=M["features"])
            C = C[C.abs() > 0.05].sort_values()
            ups, downs = C[C > 0].tail(4), C[C < 0].head(4)
            show = pd.concat([downs, ups])
            if len(show):
                fig, ax = plt.subplots(figsize=(10, 0.45 * len(show) + 0.8))
                ax.barh(range(len(show)), show.values, color=[BLUE if v < 0 else ORANGE for v in show.values], height=0.6)
                ax.set_yticks(range(len(show)), [M["nice"].get(f, f) for f in show.index], fontsize=9)
                ax.axvline(0, color=INK, lw=1)
                ax.set_xlabel("Effect on the risk score (log odds)")
                ax.grid(axis="y", visible=False)
                ax.set_title(f"What drives the latest score ({row.month.iloc[0]}): orange raises risk, blue lowers it",
                             loc="left", fontsize=11, color=INK)
                fig.tight_layout()
                st.pyplot(fig, width="stretch"); plt.close(fig)

# ---------------- bank view ----------------
with tab_b:
    st.divider()
    st.markdown("### Bank view: who to call this month")
    if bank is None:
        st.warning("Run early_warning.py first.")
    else:
        fresh = bank[bank.past_overdraft.isna()]
        c = st.columns(4)
        c[0].metric("Flagged today", int(bank.flag.sum()))
        c[1].metric("…never overdrawn before", int(fresh.flag.sum()))
        blind = fresh[fresh.flag & (fresh.runway_base >= 12)]
        c[2].metric("…of which runway says 12+ months", len(blind))
        c[3].metric("Validated catch rate", "72%", help="Share of first overdrafts flagged 1 to 3 months ahead, test period")
        show_all = st.checkbox("Include customers who already overdrew", value=False)
        view = (bank if show_all else fresh)
        view = view[view.flag] if st.checkbox("Flagged only", value=True) else view
        st.dataframe(view[["rank", "account_id", "risk", "why", "runway_base", "runway_ei55", "age_band", "loan"]]
                     .assign(risk=view.risk.map("{:.1%}".format))
                     .rename(columns={"risk": "3 month risk", "runway_base": "runway now", "runway_ei55": "runway if laid off"}),
                     hide_index=True, width="stretch", height=420)

# ---------------- findings ----------------
with tab_f:
    st.markdown("### Findings")
    for img, cap in [("1_blind_spot.png", "Runway alone misses most real overdrafts"),
                     ("3_fragility_age.png", "Who is fragile"),
                     ("2_warning_signal.png", "The warning signs"),
                     ("4_borrowers.png", "Borrowers")]:
        f = OUT / "charts" / img
        if f.exists():
            st.image(str(f), caption=cap, width="stretch")

# ---------------- how it's calculated ----------------
with tab_i:
    st.divider()
    st.markdown("### How every number is calculated")
    st.caption("Each section shows the formula, then this account and the current twist plugged in, then the evidence.")

    with st.expander("1. The customer's numbers (judges' definitions, exact)", expanded=True):
        st.latex(r"I=\operatorname{median}(\text{money\_in}),\quad S=\operatorname{median}(\text{money\_out}),\quad "
                 r"H=\operatorname{median}(\text{household\_bills})")
        st.latex(r"F=\operatorname{median}(\text{household\_bills}+\text{insurance}+\text{loan\_payments}),\quad "
                 r"B=\tfrac{1}{6}\textstyle\sum_{\text{last 6 months}}\text{avg\_daily\_balance}")
        st.latex(r"P=\text{monthly\_payment if the loan is still running at the end of 2025 (status C or D), else } 0")
        st.markdown(f"**Account {acc}:** I = {money(p['I'])}, S = {money(p['S'])}, H = {money(p['H'])}, "
                    f"F = {money(p['F'])}, B = {money(p['B'])}, P = {money(p['P'])}  "
                    f"(from {p['n_months']} rows of monthly_summary.csv)")
        st.success("Verified: an independent implementation of these definitions matches the engine on all 4,500 "
                   "accounts with 0 mismatches. 27,637 further checks over every account and random combined twists "
                   "found no errors, and a worse twist never gives a longer runway.")

    with st.expander("2. Runway and the twist", expanded=True):
        st.latex(r"\text{runway}=\begin{cases}0 & B-C\le 0\\ \text{no shortfall (120+)} & S'-I'\le 0\\ "
                 r"\min\!\left(\dfrac{B-C}{S'-I'},\,120\right) & \text{otherwise}\end{cases}")
        st.latex(r"I'=\rho I+\Delta I,\qquad S'=S(1+\sigma)+\beta H+\Delta H+E+\lambda P,\qquad C=C_0+kS")
        st.markdown("ρ income multiple (EI = 0.55), ΔI income change, σ all spending up, β bills up %, ΔH bills up $, "
                    "E new recurring cost, λ loan payment up %, C one off cost. The judges' twists use only ρ, β and C.")
        st.latex(r"\text{Now: }\frac{%s-%s}{%s-%s}=\frac{%s}{%s}\;\Rightarrow\;\textbf{%s}"
                 % (f"{p['B']:,.0f}", f"{r_j['one_off']:,.0f}", f"{r_j['spending']:,.0f}", f"{r_j['income']:,.0f}",
                    f"{r_j['buffer_left']:,.0f}", f"{r_j['gap']:,.0f}",
                    r_j["runway_label"].replace("(", "").replace(")", "")))
        st.markdown("**Realistic EI (our extension).** EI pays 55% of insurable earnings up to $729 per week "
                    "(canada.ca, 2026), and a layoff does not stop a pension:")
        st.latex(r"I'=\Pi+\min\big(0.55\,(I-\Pi),\;\bar E\big),\qquad \bar E=729\times\tfrac{52}{12}=\$3{,}159\text{ per month}")
        st.markdown(f"Π = typical pension + interest income. For this account Π = {money(min(p['cat']['pension_in'] + p['cat']['interest_in'], p['I']))}; "
                    f"realistic runway under this twist: **{r_r['runway_label']}**. The cap binds for 57% of customers.")

    with st.expander("3. Month by month path and the survival probability"):
        st.latex(r"X_0=B-C,\qquad X_m=X_{m-1}+I'_m-S'_m,\qquad I'_m=\begin{cases}I' & m\le T_{EI}\\ "
                 r"\rho_{\text{after}}I & m>T_{EI}\end{cases}")
        st.markdown("The straight line uses median income and spending every month. The simulation replays the "
                    "customer's own months instead (blocks of 3 consecutive real months from the last 24, after a "
                    "6 month burn in), and also subtracts each month's dip below its month end balance:")
        st.latex(r"\hat P(\text{no overdraft through } T)=\frac{1}{N}\sum_{n=1}^{N}"
                 r"\mathbb 1\Big\{\min_{m\le T}\big(X^{(n)}_m-\text{dip}^{(n)}_m\big)\ge 0\Big\},\qquad N=5{,}000")
        mc = load_csv("mc_validation.csv")
        if mc is not None:
            r = mc.set_index("score").loc["mc_od_current"]
            st.markdown(f"**Evidence:** as a ranking of who overdraws next, the simulation scores AUC "
                        f"{r.auc:.2f} [{r.auc_lo:.2f}, {r.auc_hi:.2f}] out of sample, against 0.65 for runway. "
                        "**Caveat:** its probabilities are conservative (too high), because real customers cut "
                        "spending when money gets tight (they spend 71% of their usual amount at the lowest "
                        "balances). Read it as a stress view, and for comparing scenarios.")

    with st.expander("4. Early warning model"):
        st.latex(r"y_{i,t}=\mathbb 1\Big\{\min_{h\in\{1,2,3\}}\text{min\_balance}_{i,t+h}<0\Big\},\qquad "
                 r"P(y_{i,t}=1)=\sigma\big(w_0+\mathbf w^\top \mathbf x_{i,t}\big)")
        st.markdown(
            "- **Who is scored:** customers who have never overdrawn, with at least 6 months of history.\n"
            "- **Features:** only months up to t. Levels, flows, rolling 3 and 6 month dynamics, and each customer "
            "compared with their own past: \\(z_{i,t}=(x_{i,t}-\\bar x_{i,<t})/(s_{i,<t}+1)\\).\n"
            "- **Never used:** age, gender, region, loan status (an outcome), standing orders (an end of data "
            "snapshot), and the penalty interest that is charged *because* of an overdraft.\n"
            "- **Split by time:** train on months up to 2023-09, fix the alert threshold on 2023-10 to 2024-09 "
            "(catch 50% of risk months), score once on 2024-10 to 2025-09.\n"
            "- **Uncertainty:** 95% intervals from resampling whole accounts.")
        M = load_model()
        if M is not None:
            coef = pd.Series(M["model"].named_steps["lr"].coef_[0], index=M["features"])
            top = coef.reindex(coef.abs().sort_values(ascending=False).index).head(10)
            st.markdown("**Largest standardized weights** (positive = more risk):")
            st.dataframe(pd.DataFrame({"feature": [M["nice"].get(f, f) for f in top.index],
                                       "weight": top.round(3).values}), hide_index=True, width="stretch")

    with st.expander("5. Evidence: runway against real overdrafts"):
        rv = load_csv("runway_validation.csv")
        if rv is not None:
            names = {"runway_base": "Runway, judges' formula", "runway_ei55": "Runway under EI 55%",
                     "runway_ei55_bills15": "Runway under EI 55% + bills 15%", "runway_ei_realistic": "Runway, realistic EI",
                     "runway_furnace": "Runway after furnace repair", "model_no_snapshot": "Early warning model",
                     "model_no_snapshot_plus_runway": "Model + runway as extra inputs", "model_full": "Model incl. snapshot feature (not used)"}
            t = rv.assign(score=rv.score.map(names).fillna(rv.score),
                          AUC=rv.auc.round(3), CI=[f"[{a:.2f}, {b:.2f}]" for a, b in zip(rv.auc_lo, rv.auc_hi)])
            st.markdown("Every score recomputed with data up to the prediction month only, test period "
                        "(48,400 account months, 96 first overdrafts). AUC 0.5 is a coin flip, 1.0 is perfect.")
            st.dataframe(t[["score", "AUC", "CI"]], hide_index=True, width="stretch")
        rb = load_csv("robustness.csv")
        if rb is not None:
            st.markdown("**Robustness:** share of people who overdrew that runway showed as safe (12+ months) beforehand.")
            rb2 = rb.assign(runway=rb.runway.map({"rw_judge": "judges' buffer", "rw_current": "actual balance",
                                                  "rw_judge_ei55": "judges' buffer, EI 55%"}),
                            looked_safe=(rb.safe_ge12 * 100).round(0).astype(int).astype(str) + "%", AUC=rb.auc.round(3))
            st.dataframe(rb2[["horizon", "runway", "events", "AUC", "looked_safe"]]
                         .rename(columns={"horizon": "horizon (months)"}), hide_index=True, width="stretch")

    with st.expander("6. Fairness check"):
        fa = load_csv("fairness.csv")
        if fa is not None:
            st.markdown("Early warning model on the test period at the fixed threshold. Demographics are joined only "
                        "here, never used as inputs. Groups with fewer than 10 events are too small to judge.")
            st.dataframe(fa.assign(event_recall=fa.event_recall.round(2), alert_rate=(fa.alert_rate * 100).round(1))
                         [["segment", "group", "events", "event_recall", "alert_rate", "small_sample"]]
                         .rename(columns={"alert_rate": "alert rate %"}), hide_index=True, width="stretch")

    with st.expander("7. Data and sources"):
        st.markdown(
            "- PKDD'99 Discovery Challenge Financial dataset (real anonymized bank data), CTU Prague Relational "
            "Dataset Repository: relational.fel.cvut.cz/dataset/Financial. Amounts in CAD as provided.\n"
            "- EI regular benefits, maximum weekly benefit $729 and maximum insurable earnings $68,900 (2026): "
            "canada.ca/en/services/benefits/ei/ei-regular-benefit/benefit-amount.html\n"
            "- Only monthly_summary.csv, loans.csv, accounts, dispositions, clients, cards and districts are used; "
            "transactions.csv only to remove penalty interest and count transactions.")

# ---------------- investing ----------------
with tab_v:
    import portfolio as PF
    st.markdown("### Investing: cushion first, then invest the rest")
    st.caption("Illustration, not advice. Model portfolios of broad index ETFs, no stock picking. "
               "Past returns do not guarantee future returns.")
    if load_market() is None:
        st.warning("Run portfolio.py once to download the market data.")
    else:
        cc = st.columns([2, 1])
        tol = cc[0].segmented_control("Customer's risk tolerance (their answer to a questionnaire)", PF.LEVELS,
                                      default="Growth", key="tolerance")
        yrs = cc[1].segmented_control("Horizon", [5, 10, 20], default=10, key="inv_years",
                                      format_func=lambda v: f"{v} years")
        flag_now = bool(ew is not None and not isinstance(ew["past_overdraft"], str) and ew["flag"])
        pl = run_invest(acc, tol or "Growth", int(yrs or 10), flag_now)

        st.markdown("#### 1. The cushion")
        k = st.columns(4)
        k[0].metric("Cushion target", f"{pl['cushion_months']:g} months", help="3 months if income is steady, 4.5 if "
                    "somewhat irregular, 6 if irregular (income variation over the last 12 months)")
        k[1].metric("In dollars", money(pl["cushion"]))
        k[2].metric("Buffer now", money(p["B"]))
        k[3].metric("Income variation", f"{pl['income_cv']:.0%}", help="Standard deviation / mean of money_in, last 12 months")
        st.progress(float(min(1.0, p["B"] / pl["cushion"])) if pl["cushion"] > 0 else 1.0,
                    text="Cushion full" if pl["cushion_gap"] == 0 else
                    f"{money(pl['cushion_gap'])} short of the cushion")

        st.markdown("#### 2. Risk capacity: what they can afford")
        k = st.columns([1, 3])
        k[0].metric("Risk capacity", pl["capacity"])
        k[1].markdown("**Why:** " + "; ".join(pl["why"]) + ".  \n"
                      f"**Portfolio:** {pl['level']}" +
                      (f" (they asked for {pl['tolerance']}, capped by {pl['capacity'].lower()} capacity)" if pl["capped"] else
                       " (their tolerance, within their capacity)"))

        st.markdown("#### 3. The plan")
        if not pl["investable"]:
            st.info(pl["summary"])
        else:
            k = st.columns(4)
            k[0].metric("Upfront", money(pl["upfront"]), help="Buffer above the cushion")
            k[1].metric("Weekly", money(pl["weekly"]), help="Half the monthly surplus, x 12 / 52" +
                        ("" if not pl["fill_months"] else f"; starts after {pl['fill_months']} months of filling the cushion"))
            k[2].metric(f"Likely value in {yrs or 10} years", money(pl["end_p50"]))
            k[3].metric("Same money in savings", money(pl["end_savings"]), help=f"At {PF.SAVINGS_RATE:.1%} a year")
            st.write(pl["summary"])
            f = pl["fan"]
            fig, ax = plt.subplots(figsize=(10, 3.6))
            yrs_x = f.month / 12
            ax.fill_between(yrs_x, f.p10 / 1e3, f.p90 / 1e3, color=SEQ[0], lw=0, label="80% of simulated outcomes")
            ax.fill_between(yrs_x, f.p25 / 1e3, f.p75 / 1e3, color=SEQ[1], lw=0, label="middle 50%")
            ax.plot(yrs_x, f.p50 / 1e3, color=SEQ[3], lw=2, label=f"{pl['level']} portfolio, median")
            ax.plot(yrs_x, f.savings / 1e3, color=ORANGE, lw=2, ls="--", label=f"savings account at {PF.SAVINGS_RATE:.1%}")
            ax.plot(yrs_x, f.paid_in / 1e3, color=GRAY, lw=1.5, label="money put in")
            ax.set_xlabel("Years"); ax.set_ylabel("$ thousands"); ax.legend(frameon=False, fontsize=9, loc="upper left")
            ax.set_title("Projected value", loc="left", fontsize=12, color=INK)
            st.pyplot(fig, width="stretch"); plt.close(fig)
            st.caption(f"5,000 paths replaying 12 month blocks of the ETFs' real monthly total returns "
                       f"({pl['market_from']} to {pl['market_to']}, {pl['market_months']} months, from yfinance). "
                       f"Chance of ending below the money put in: {pl['p_below_paid']:.0%}. "
                       f"Chance of beating savings: {pl['p_beats_savings']:.0%}.")

        st.markdown("#### The three model portfolios")
        stats = PF.portfolio_stats(load_market())
        hold = pd.DataFrame(PF.PORTFOLIOS).T.rename(columns=PF.TICKERS).map("{:.0%}".format)
        st.dataframe(hold, width="stretch")
        st.dataframe(stats.assign(**{c: stats[c].map("{:.1%}".format) for c in
                                     ["annual_return", "volatility", "worst_12_months", "max_drawdown"]})
                     .rename(columns={"annual_return": "return per year", "worst_12_months": "worst 12 months",
                                      "max_drawdown": "worst drop"}), hide_index=True, width="stretch")
        st.caption("Historical, 2015 to 2026, in Canadian dollars. 2022 hit every portfolio because bonds and stocks fell "
                   "together, which is why even Conservative lost 11% over its worst 12 months.")
        cap = load_csv("capacity.csv")
        if cap is not None:
            st.markdown(f"**Across all customers:** {cap.below_cushion.mean():.0%} are below their own cushion; "
                        f"{(cap.cushion_months > 3).mean():.0%} need more than the usual 3 months because their income "
                        f"is irregular; {cap.nothing.mean():.0%} have nothing to invest yet. Risk capacity: "
                        + ", ".join(f"{k} {v:.0%}" for k, v in cap.capacity.value_counts(normalize=True).items()) + ".")

# ---------------- states ----------------
with tab_st:
    import migration as MG
    mg = load_migration()
    st.markdown("### Financial state: Stable, Stretched, Distressed")
    st.caption("Each month, from history up to that month only. Buffer months = judges' B / S as of that month. "
               "Stable: 3+ months. Stretched: 1 to 3. Distressed: under 1, or overdrawn that month. "
               "The first 6 months of an account are a burn in.")
    if mg is None:
        st.warning("Run migration.py first.")
    else:
        h = MG.add_states(p["history"])
        COL = {0: BLUE, 1: "#f2a37f", 2: "#c4441a"}
        fig, ax = plt.subplots(figsize=(10, 1.5))
        for i, s_ in enumerate(h.state):
            ax.bar(i, 1, width=1, color=GRID if np.isnan(s_) else COL[int(s_)], edgecolor="white", lw=0.3)
        step = max(1, len(h) // 8)
        ax.set_xticks(range(0, len(h), step), h.month.iloc[::step], fontsize=8)
        ax.set_yticks([]); ax.grid(False); ax.set_xlim(-0.5, len(h) - 0.5)
        for k_, nm in enumerate(MG.NAMES):
            ax.bar(0, 0, color=COL[k_], label=nm)
        ax.bar(0, 0, color=GRID, label="burn in")
        ax.legend(frameon=False, fontsize=8, ncol=4, loc="upper left", bbox_to_anchor=(0, 1.45))
        st.pyplot(fig, width="stretch"); plt.close(fig)
        cur = h.state.dropna()
        if len(cur):
            s0 = int(cur.iloc[-1])
            run = (h.state != h.state.shift()).cumsum()
            since = h.loc[run == run.iloc[-1], "month"].iloc[0]
            P, lo, hi = (np.array(mg[k_]) for k_ in ("P", "P_lo", "P_hi"))
            k = st.columns(4)
            k[0].metric("State now", MG.NAMES[s0], help=f"Since {since}")
            for j in range(3):
                k[j + 1].metric(f"Next month: {MG.NAMES[j]}", f"{P[s0, j]:.1%}",
                                help=f"95% CI {lo[s0, j]:.1%} to {hi[s0, j]:.1%}, from {mg['n_train']:,} monthly moves (train years)")

        st.markdown("#### Finding: distress is sticky")
        mk = mg["markov_check"]
        sticky = pd.DataFrame({"months later": [3, 6, 12],
                               "a memoryless model predicts": [f"{mk[str(k_)]['pred'][2][2]:.0%}" for k_ in (3, 6, 12)],
                               "actually still Distressed": [f"{mk[str(k_)]['actual'][2][2]:.0%}" for k_ in (3, 6, 12)]})
        c1, c2 = st.columns([2, 3])
        c1.dataframe(sticky, hide_index=True, width="stretch")
        mem, sh, pr = mg["memory_stretched"], mg["shock"], mg["prediction"]
        c2.markdown(
            f"- Once Distressed, customers stay there far longer than a standard migration matrix predicts, so we "
            f"**do not** use it to forecast months ahead. The lesson: catch customers while they are Stretched.\n"
            f"- **Relapse:** a Stretched customer who just left Distressed has a {mem['Distressed']['to_distressed']:.0%} "
            f"chance of falling back next month, against {mem['Stretched']['to_distressed']:.0%} for one already Stretched.\n"
            f"- **Shocks:** in a month where income falls below 60% of usual, Stretched to Distressed rises from "
            f"{sh['P_calm'][1][2]:.1%} to {sh['P_shock'][1][2]:.1%}.\n"
            f"- **Warning value:** within 3 months, Distressed customers overdraw at "
            f"{pr['overdraft_rate_by_state']['Distressed']['rate']:.1%} against "
            f"{pr['overdraft_rate_by_state']['Stable']['rate']:.1%} for Stable. The state ranks risk better than runway "
            f"(AUC {pr['auc_state']:.2f} vs {pr['auc_runway']:.2f}), but well below the early warning model (0.955).")
        st.caption("All on the held out test year (Oct 2024 to Sep 2025); matrix estimated on the train years.")

        st.markdown(f"#### The bank this month ({mg['bank']['month']})")
        c1, c2 = st.columns(2)
        mix = mg["bank"]["mix"]
        c1.dataframe(pd.DataFrame({"state": MG.NAMES, "customers": [int(x) for x in mix],
                                   "share": [f"{x / sum(mix):.0%}" for x in mix]}), hide_index=True, width="stretch")
        mv = pd.DataFrame(mg["bank"]["moves_last_month"], index=[f"from {n}" for n in MG.NAMES],
                          columns=[f"to {n}" for n in MG.NAMES])
        c2.dataframe(mv, width="stretch")
        c2.caption("Moves between last month and this month.")
        with st.expander("Monthly migration matrix with 95% confidence intervals"):
            P, lo, hi = (np.array(mg[k_]) for k_ in ("P", "P_lo", "P_hi"))
            st.dataframe(pd.DataFrame([[f"{P[i, j]:.1%} ({lo[i, j]:.1%} to {hi[i, j]:.1%})" for j in range(3)]
                                       for i in range(3)], index=[f"from {n}" for n in MG.NAMES],
                                      columns=[f"to {n}" for n in MG.NAMES]), width="stretch")
            Pt = np.array(mg["P_test"])
            st.caption(f"Train years, {mg['n_train']:,} moves. Same matrix on the test year: Stable stays "
                       f"{Pt[0, 0]:.1%}, Stretched stays {Pt[1, 1]:.1%}, Distressed stays {Pt[2, 2]:.1%} "
                       f"(train {P[2, 2]:.1%}): distress has become stickier recently.")
