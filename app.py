"""
Resilience engine: Streamlit app.
Run:  streamlit run app.py
"""
from dataclasses import astuple
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

import engine as E
import viz as V

OUT = Path(__file__).parent / "outputs"
st.set_page_config(page_title="Resilience Engine", layout="wide")
st.markdown("""<style>
:root { --sage:#565A4B; --sage2:#6E7361; --terra:#B5715A; --sand:#F6F1EA; --card:#FBF8F3; --line:#DDD2C3;
        --ink:#2E3029; --ink2:#6E6A60; --serif: Georgia, 'Times New Roman', serif; }
.block-container, div[data-testid="stMainBlockContainer"] {padding-top: 3.6rem; max-width: 1280px;}
h1, h2, h3, h4 {font-family: var(--serif) !important; letter-spacing: -0.01em; color: var(--ink);}
/* sidebar */
section[data-testid="stSidebar"] h1 {font-size: 1.15rem; letter-spacing: 0.28em; text-transform: uppercase; font-weight: 400;}
section[data-testid="stSidebar"] h3 {font-size: 0.72rem; letter-spacing: 0.18em; text-transform: uppercase;
  font-family: inherit !important; color: var(--sage); margin-top: 0.6rem;}
section[data-testid="stSidebar"] .stButton button {padding: 0.1rem 0.6rem; min-height: 1.9rem; font-size: 0.82rem;
  justify-content: flex-start;}
section[data-testid="stSidebar"] .stButton {margin-bottom: -0.55rem;}
/* buttons */
div[data-testid="stMain"] .stButton button[kind="primary"] p {letter-spacing: 0.1em; text-transform: uppercase; font-size: 0.74rem;}
section[data-testid="stSidebar"] .stButton button[kind="primary"] {justify-content: center;}
/* tabs as a quiet top nav */
div[role="tablist"] {gap: 2.2rem; border-bottom: 1px solid var(--line);}
div[data-testid="stTab"] p {font-size: 0.74rem !important; letter-spacing: 0.2em; text-transform: uppercase; color: var(--ink2);}
div[data-testid="stTab"][aria-selected="true"] p {color: var(--ink); font-weight: 600;}
div[data-testid="stTab"] .react-aria-SelectionIndicator {background-color: var(--sage) !important;}
/* cards */
div[data-testid="stVerticalBlockBorderWrapper"]:has(> div > div[data-testid="stVerticalBlock"]) {background: var(--card);}
div[data-testid="stMetric"] label p {font-size: 0.68rem; letter-spacing: 0.14em; text-transform: uppercase; color: var(--ink2);}
div[data-testid="stMetricValue"] {font-family: var(--serif);}
/* custom blocks */
.eyebrow {font-size: 0.68rem; letter-spacing: 0.22em; text-transform: uppercase; color: var(--sage2); margin-bottom: 0.15rem;}
.hero h1 {font-size: 2.7rem; font-weight: 400; margin: 0 0 0.2rem 0; padding: 0;}
.hero h1 em {color: var(--terra); font-style: italic;}
.hero .sub {color: var(--ink2); font-size: 0.95rem;}
.kpis {display: grid; grid-template-columns: repeat(auto-fit, minmax(150px, 1fr)); background: var(--card);
  border: 1px solid var(--line); border-radius: 0.3rem; margin: 1.1rem 0 0.9rem 0;}
.kpi {padding: 0.85rem 1.1rem; border-left: 1px solid var(--line);}
.kpi:first-child {border-left: none;}
.kpi .l {font-size: 0.66rem; letter-spacing: 0.16em; text-transform: uppercase; color: var(--ink2);}
.kpi .v {font-family: var(--serif); font-size: 1.65rem; color: var(--ink); line-height: 1.25; margin-top: 0.2rem;}
.kpi .n {font-size: 0.78rem; color: var(--ink2); margin-top: 0.1rem;}
.kpi .n.bad {color: #A8492F;} .kpi .n.good {color: #56704A;}
.verdict {font-family: var(--serif); font-size: 1.12rem; line-height: 1.5; color: var(--ink); background: #F0E8DC;
  border-left: 3px solid var(--sage); padding: 0.75rem 1.1rem; border-radius: 0 0.3rem 0.3rem 0; margin-bottom: 1.4rem;}
.verdict.alert {border-left-color: var(--terra);}
.sec {margin: 1.6rem 0 0.6rem 0;}
.sec h3 {font-size: 1.55rem; font-weight: 400; margin: 0; padding: 0;}
.sec p {color: var(--ink2); margin: 0.15rem 0 0 0; font-size: 0.92rem;}
.cardtitle {font-family: var(--serif); font-size: 1.12rem; color: var(--ink); margin-bottom: 0.1rem;}
.cardsub {font-size: 0.82rem; color: var(--ink2); margin-bottom: 0.4rem;}
</style>""", unsafe_allow_html=True)

INK, INK2 = "#2E3029", "#6E6A60"


def section(eyebrow, title, sub=None):
    st.markdown(f"<div class='sec'><div class='eyebrow'>{eyebrow}</div><h3>{title}</h3>"
                + (f"<p>{sub}</p>" if sub else "") + "</div>", unsafe_allow_html=True)


def card_title(title, sub=None):
    st.markdown(f"<div class='cardtitle'>{title}</div>" + (f"<div class='cardsub'>{sub}</div>" if sub else ""),
                unsafe_allow_html=True)


def kpi_band(items):
    """items: (label, value, note, tone) with tone in '', 'good', 'bad'."""
    cells = "".join(f"<div class='kpi' title='{n if n else ''}'><div class='l'>{l}</div><div class='v'>{v}</div>"
                    f"<div class='n {t}'>{n}</div></div>" for l, v, n, t in items)
    st.markdown(f"<div class='kpis'>{cells}</div>", unsafe_allow_html=True)


def chart(c):
    st.altair_chart(c, width="stretch", theme=None)


money = lambda x: f"-${-x:,.0f}" if x < 0 else f"${x:,.0f}"


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
if "acc" not in st.session_state and st.query_params.get("acc", "").isdigit():
    st.session_state["acc"] = int(st.query_params["acc"])
st.session_state.setdefault("acc", 20111)
st.session_state.setdefault("started", "acc" in st.query_params)


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
    st.session_state["started"] = True
    st.query_params["acc"] = str(int(a))


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

def go_home():
    st.session_state["started"] = False
    st.query_params.clear()


def landing():
    st.markdown("""<style>
section[data-testid="stSidebar"], div[data-testid="stSidebarCollapsedControl"], header[data-testid="stHeader"] {display: none !important;}
div[data-testid="stMainBlockContainer"], .block-container {max-width: 1320px; padding-top: 0 !important; padding-bottom: 0;}
.topbar {display: flex; justify-content: space-between; align-items: center; padding: 1.6rem 0 1.4rem 0;
  border-bottom: 1px solid var(--line); margin-bottom: 4.5vh;}
.topbar .brand {font-family: var(--serif); letter-spacing: 0.32em; text-transform: uppercase; font-size: 1.05rem; color: var(--ink);}
.topbar .nav {display: flex; gap: 2.2rem; font-size: 0.72rem; letter-spacing: 0.2em; text-transform: uppercase; color: var(--ink2);}
.land .title {font-family: var(--serif); font-weight: 400; font-size: clamp(3.4rem, 7vw, 6rem); line-height: 1.0;
  letter-spacing: -0.02em; margin: 0.5rem 0 0 0; color: var(--sage);}
.land .title em {color: var(--terra); font-style: italic; display: block;}
@media (max-width: 640px) {.topbar .nav {display: none;}}
.land .rule {width: 52px; height: 1px; background: var(--line); margin: 1.8rem 0 1.4rem 0;}
.land .tag {color: var(--ink2); font-size: 1.02rem; max-width: 34ch; line-height: 1.55; margin-bottom: 1.6rem;}
div[data-testid="stForm"] {border: none; padding: 0; background: transparent;}
.land-btns .stButton button, div[data-testid="stForm"] button {border-radius: 0.15rem; min-height: 2.9rem;}
div[data-testid="stForm"] button p, .land-btns .stButton button p {letter-spacing: 0.16em; text-transform: uppercase; font-size: 0.72rem;}
.band {margin-top: 6vh; border-top: 1px solid var(--line); border-bottom: 1px solid var(--line); overflow: hidden;
  white-space: nowrap; padding: 0.9rem 0; background: #EFE7DC;}
.band .track {display: inline-block; animation: slide 38s linear infinite;}
.band span {font-size: 0.74rem; letter-spacing: 0.26em; text-transform: uppercase; color: var(--ink2); margin: 0 1.4rem;}
.band b {color: var(--terra); font-weight: 400;}
@keyframes slide {from {transform: translateX(0);} to {transform: translateX(-50%);}}
@media (prefers-reduced-motion: reduce) {.band .track {animation: none;}}
.land-foot {color: var(--ink2); font-size: 0.8rem; padding: 1.2rem 0 2rem 0;}
</style>""", unsafe_allow_html=True)
    st.markdown("<div class='topbar'><div class='brand'>Resilience Engine</div>"
                "<div class='nav'><span>Know</span><span>Shock</span><span>Warn</span><span>Plan</span></div></div>",
                unsafe_allow_html=True)
    left, right = st.columns([1, 1.1], gap="large", vertical_alignment="center")
    with left:
        st.markdown("<div class='land'><div class='eyebrow'>Tangerine · Resilience Rewired</div>"
                    "<div class='title'>Resilience<em>Engine.</em></div><div class='rule'></div>"
                    "<div class='tag'>Know any customer, apply any shock, and see the plan, in seconds.</div></div>",
                    unsafe_allow_html=True)
        with st.form("land_search", border=False):
            q = st.text_input("Account ID", placeholder="Search an account ID, e.g. 29010", label_visibility="collapsed")
            c1, c2 = st.columns(2)
            go = c1.form_submit_button("Open account  ↗", type="primary", width="stretch")
            rnd = c2.form_submit_button("Random account", width="stretch")
        if go:
            q = q.strip()
            if q.isdigit() and int(q) in id_set:
                choose(q); st.rerun()
            elif q:
                near = [a for a in ids if str(a).startswith(q)][:8]
                st.error(f"No account {q}." + (f" Close matches: {', '.join(map(str, near))}." if near else
                                                f" IDs run from {ids[0]} to {ids[-1]}."))
            else:
                st.error("Type an account ID first, or press Random account.")
        if rnd:
            pick_random(); st.rerun()
    with right:
        demo = 29010
        pd_ = load_profile(demo)
        sv_ = run_survival(demo, astuple(E.Shock(income_mult=0.55, ei_months=6)))
        st.altair_chart(V.hero_fan(sv_["fan"], pd_["B"]), width="stretch", theme=None)
        st.markdown("<div style='color:var(--ink2);font-size:0.78rem;letter-spacing:0.08em'>Account 29010, laid off on EI: "
                    "5,000 simulated futures from their own months. Dashed line: $0.</div>", unsafe_allow_html=True)
    words = ["Know the customer", "Apply the shock", "Warn early", "Make a plan", "Build the cushion", "Then invest"]
    run = "".join(f"<span>{w}</span><span><b>◆</b></span>" for w in words)
    st.markdown(f"<div class='band'><div class='track'>{run}{run}{run}{run}</div></div>"
                f"<div class='land-foot'>{len(ids):,} customers · judges' definitions, exact · early warning validated on a "
                f"held out year</div>", unsafe_allow_html=True)


if not st.session_state["started"]:
    landing()
    st.stop()

st.sidebar.button("← Home", on_click=go_home, key="home_btn")
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
    if b1.button("Choose", width="stretch", type="primary"):
        account_picker()
    b2.button("Random", width="stretch", on_click=pick_random)

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
kpi_rw = lambda lab: "120+ months" if "120+" in lab else f"{lab} months"

demo_line = f" · demo: {demo_map[acc]}" if acc in demo_map else ""
st.markdown(f"<div class='hero'><div class='eyebrow'>Resilience engine · customer profile</div>"
            f"<h1>Account <em>{acc}</em></h1><div class='sub'>{p['n_months']} months of history, {p['first_month']} "
            f"to {p['last_month']}{demo_line}<br>Twist: {shock.describe()}</div></div>", unsafe_allow_html=True)

if ew is None:
    ew_txt, ew_note, ew_tone = "n/a", "under 7 months of history", ""
elif isinstance(ew["past_overdraft"], str):
    ew_txt, ew_note, ew_tone = "Overdrew before", f"first in {ew['past_overdraft']}", "bad"
elif ew["flag"]:
    ew_txt, ew_note, ew_tone = "Flagged", f"{ew['risk']:.1%} risk, next 3 months", "bad"
else:
    ew_txt, ew_note, ew_tone = "Not flagged", f"{ew['risk']:.2%} risk, next 3 months", "good"
chg = r_j["runway_change"]
kpi_band([
    ("Runway now", kpi_rw(p["runway"][1]), "judges' formula, no shock", ""),
    ("After the twist", kpi_rw(r_j["runway_label"]),
     "" if ("120+" in r_j["runway_label"] or chg == 0) else f"{chg:+.1f} months", "bad" if chg < 0 else ""),
    ("Realistic EI", kpi_rw(r_r["runway_label"]), "EI capped at $729/week, pension kept", ""),
    ("Monthly gap", money(r_j["gap"]), "spending exceeds income" if r_j["gap"] > 0 else "income covers spending",
     "bad" if r_j["gap"] > 0 else "good"),
    ("Early warning", ew_txt, ew_note, ew_tone),
])

twisted = shock.describe() != "no shock"
lab_v = r_j["runway_label"] if twisted else p["runway"][1]
lasts = ("income covers spending, so the buffer <b>never runs down</b>" if "120+" in lab_v
         else f"the buffer lasts <b>{fmt_rw(lab_v)}</b>")
v1 = (f"After this twist, {lasts}" if twisted else f"On the judges' formula, {lasts}")
safe_rw = (r_j["runway"] if twisted else p["runway"][0]) >= 12
if ew is None:
    v2, alert = ".", False
elif isinstance(ew["past_overdraft"], str):
    v2, alert = f", and they have already overdrawn (first in {ew['past_overdraft']}).", True
elif ew["flag"]:
    v2, alert = (f"{', yet' if safe_rw else ', and'} the early warning model flags a <b>{ew['risk']:.1%}</b> chance of a "
                 f"first overdraft within 3 months.", True)
else:
    v2, alert = f", and the early warning model does not flag them ({ew['risk']:.2%}).", False
st.markdown(f"<div class='verdict{' alert' if alert else ''}'>{v1}{v2}</div>", unsafe_allow_html=True)

tab_c, tab_s, tab_v, tab_m, tab_f = st.tabs(["Customer", "Twist and plan", "Investing", "Model", "Proof"])
tab_p, tab_b, tab_i = tab_s, tab_m, tab_f        # merged: plan under the twist, bank view under the model, calculations under findings

# ---------------- customer ----------------
with tab_c:
    section("Know the customer", "Their numbers", "Judges' definitions, computed from this account's own history.")
    kpi_band([("Income I", money(p["I"]), "median money in", ""),
              ("Spending S", money(p["S"]), "median money out", ""),
              ("Fixed costs F", money(p["F"]), "bills + insurance + loans", ""),
              ("Buffer B", money(p["B"]), "avg daily balance, last 6 mo", ""),
              ("Household bills H", money(p["H"]), "median", ""),
              ("Loan payment P", money(p["P"]), "running at end of 2025", "")])

    left, right = st.columns([3, 2], gap="medium")
    with left:
        with st.container(border=True):
            card_title("Balance history", "Month end balance and the lowest point inside each month. Hover for values.")
            chart(V.balance_history(p["history"]))
        with st.container(border=True):
            card_title("Money in and out", "Each month's inflows and outflows.")
            chart(V.money_in_out(p["history"]))
    with right:
        with st.container(border=True):
            card_title("Early warning")
            if ew is not None:
                if isinstance(ew["past_overdraft"], str):
                    st.warning(f"Already overdrew (first in {ew['past_overdraft']}). Risk score {ew['risk']:.1%}.")
                elif ew["flag"]:
                    st.error(f"Flagged: {ew['risk']:.1%} chance of a first overdraft in the next 3 months "
                             f"(alert threshold 2.9%). Why: {ew['why']}.")
                else:
                    st.success(f"Not flagged: {ew['risk']:.2%} chance of a first overdraft in the next 3 months.")
            else:
                st.info("Under 7 months of history, so the model does not score this account.")
            st.caption("Validated on months it never saw: AUC 0.955, catches 72% of first overdrafts with a median "
                       "3 months warning, alerts on 2.2% of customers. No demographics used.")
        with st.container(border=True):
            card_title("Where the money goes", "Typical month per category (medians, so they need not add up to S).")
            dr = E.drivers(p, shock)
            dr = dr[dr.monthly > 0]
            st.dataframe(dr.assign(monthly=dr.monthly.map(money), share_of_spending=dr.share_of_spending.map("{:.0%}".format),
                                   runway_gain_from_20pct_cut=dr.runway_gain_from_20pct_cut.map(
                                       lambda v: "" if pd.isna(v) else f"+{v:.1f} mo"))
                         [["category", "monthly", "share_of_spending", "runway_gain_from_20pct_cut"]]
                         .rename(columns={"share_of_spending": "share", "runway_gain_from_20pct_cut": "20% cut buys"}),
                         hide_index=True, width="stretch")

# ---------------- shock ----------------
with tab_s:
    section("Apply the shock", "What the twist does", shock.describe().capitalize() + ".")
    with st.container(border=True):
        card_title("The judges' runway, with this account's numbers")
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
    q = {k: np.r_[start, f[k]] for k in ["p05", "p25", "p50", "p75", "p95"]}
    with st.container(border=True):
        card_title("Month by month", f"Chance of no overdraft: {sv['p_survive_3']:.0%} at 3 months, "
                   f"{sv['p_survive_6']:.0%} at 6, {sv['p_survive_12']:.0%} at 12. 5,000 paths replaying this customer's "
                   "own months; conservative, because real customers cut back when money gets tight.")
        chart(V.fan(np.r_[0, f.month], (q["p05"], q["p95"]), (q["p25"], q["p75"]), q["p50"],
                    {"straight line (median income and spending)": (np.r_[start, det.balance], V.S2, True)},
                    "months after the shock",
                    marker=(shock.ei_months, "reduced income ends") if (shock.ei_months and shock.income_mult != 1) else None))

# ---------------- plan ----------------
with tab_p:
    section("Make a plan", "The plan", "A backup plan if the shock hits, and a goal to be ready for the next one.")
    import portfolio as PF
    rc_plan = PF.risk_capacity(p)
    fund_m = rc_plan["cushion_months"]
    pl = E.plans(p, shock, fund_months=fund_m)
    g, b = pl["goal"], pl["backup"]
    left, right = st.columns(2, gap="medium")
    with left, st.container(border=True):
        card_title("Goal: build the emergency fund")
        st.progress(float(g["funded_pct"]), text=f"{g['funded_pct']:.0%} funded: {money(g['current'])} of {money(g['target'])}")
        st.write(g["summary"])
        st.caption(f"Target: {fund_m:g} months of spending, sized to this customer's income: 3 months if steady, 4.5 if "
                   f"somewhat irregular, 6 if irregular (income variation here {rc_plan['income_cv']:.0%}). "
                   "Once the cushion is full, the Investing tab shows the next step.")
    with right, st.container(border=True):
        card_title("Backup: if the shock hits")
        st.write(b["summary"])
        if b["options"]:
            opt = pd.DataFrame(b["options"])
            st.dataframe(opt.assign(monthly_saving=opt.monthly_saving.map(money))[["action", "monthly_saving", "runway_label"]]
                         .rename(columns={"monthly_saving": "frees up", "runway_label": "runway"}),
                         hide_index=True, width="stretch")
        st.caption(f"Baseline runway under this twist: {b['base_runway']}. All numbers from this customer's own medians.")


# ---------------- model ----------------
with tab_m:
    M, RH = load_model(), load_risk_history()
    if M is None or RH is None:
        st.warning("Run early_warning.py and the risk history step first.")
    else:
        met = M["metrics"]
        section("Warn early", "The early warning model")
        st.markdown(
            "Predicts the chance that a customer who has **never overdrawn** has their **first overdraft in the next 3 months**. "
            "It reads the customer's own monthly pattern: buffer relative to income, spending relative to income, "
            "how the balance moves against their own normal, volatility, loan burden. "
            "It never uses age, gender, region, loan status or the end of data standing orders snapshot. "
            "Logistic regression, trained on months up to Sept 2023, alert threshold fixed on Oct 2023 to Sept 2024, "
            "and scored once on Oct 2024 to Sept 2025.")
        kpi_band([("AUC, test period", "0.955", "runway alone: 0.65", "good"),
                  ("Overdrafts caught", f"{met['test_event_recall']:.0%}", "flagged 1 to 3 months ahead", ""),
                  ("Median warning", f"{met['test_median_lead']:.0f} months", "before the first overdraft", ""),
                  ("Customers alerted", f"{met['test_alert_rate']:.1%}", "of non event months", ""),
                  ("Precision per alert", f"{met['test_precision']:.0%}", f"vs {met['test_base_rate']:.1%} base rate", "")])

        h = RH[RH.account_id == acc].sort_values("t").reset_index(drop=True)
        if h.empty:
            st.info("This account has under 7 months of history, so the model does not score it.")
        else:
            thr = M["threshold"]
            top_c, bot_c, fe = V.risk_history(h, thr)
            lead = 0
            if fe is not None:
                for j in range(fe - 1, -1, -1):
                    if h.risk.iloc[j] >= thr:
                        lead += 1
                    else:
                        break
            with st.container(border=True):
                card_title("This customer's risk, month by month",
                           "Top: the model's risk (log scale), stopped at the first overdraft. Bottom: the judges' runway "
                           "at the same time. Shaded: the months the model was trained on.")
                chart(top_c)
                chart(bot_c)
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
                with st.container(border=True):
                    card_title(f"What drives the latest score ({row.month.iloc[0]})",
                               "Each factor's push on the risk score, from this customer's own numbers.")
                    chart(V.drivers(show, M["nice"]))

# ---------------- bank view ----------------
with tab_b:
    section("For the bank", "Who to call this month", "Every customer scored at their latest month, highest risk first.")
    if bank is None:
        st.warning("Run early_warning.py first.")
    else:
        fresh = bank[bank.past_overdraft.isna()]
        blind = fresh[fresh.flag & (fresh.runway_base >= 12)]
        kpi_band([("Flagged today", f"{int(bank.flag.sum())}", "customers", ""),
                  ("Never overdrawn", f"{int(fresh.flag.sum())}", "of those flagged", ""),
                  ("Runway says safe", f"{len(blind)}", "flagged, yet 12+ months of runway", "bad"),
                  ("Catch rate", "72%", "first overdrafts, test period", "")])
        show_all = st.checkbox("Include customers who already overdrew", value=False)
        view = (bank if show_all else fresh)
        view = view[view.flag] if st.checkbox("Flagged only", value=True) else view
        st.dataframe(view[["rank", "account_id", "risk", "why", "runway_base", "runway_ei55", "age_band", "loan"]]
                     .assign(risk=view.risk.map("{:.1%}".format))
                     .rename(columns={"risk": "3 month risk", "runway_base": "runway now", "runway_ei55": "runway if laid off"}),
                     hide_index=True, width="stretch", height=420)

# ---------------- findings ----------------
with tab_f:
    section("Proof", "What we found", "Every number below is measured on months the models never saw.")
    rv, mcv, seg, wsig = (load_csv(n) for n in ("runway_validation.csv", "mc_validation.csv", "population_segments.csv",
                                                "warning_signal.csv"))
    if rv is not None and mcv is not None:
        rv, mcv = rv.set_index("score"), mcv.set_index("score")
        rows = pd.DataFrame([("Runway (judges' formula)", rv.loc["runway_base"], "Judges' formula"),
                             ("Runway under EI 55% stress", rv.loc["runway_ei55"], "Judges' formula"),
                             ("Monte Carlo on own history", mcv.loc["mc_od_current"], "Our engine"),
                             ("Early warning model", rv.loc["model_no_snapshot"], "Our engine")],
                            columns=["label", "r", "kind"])
        rows = rows.assign(auc=[r.auc for r in rows.r], lo=[r.auc_lo for r in rows.r], hi=[r.auc_hi for r in rows.r]).drop(columns="r")
        with st.container(border=True):
            card_title("1. Runway alone misses most real overdrafts",
                       "79% of customers who overdrew showed 12+ months of runway three months before. How well each score "
                       "ranks real first overdrafts (line = 95% confidence interval). Test set: 48,400 account months, 96 first overdrafts.")
            chart(V.blind_spot(rows))
    if wsig is not None:
        with st.container(border=True):
            card_title("2. The warning signs start about 6 months out",
                       "Median of the 288 customers who overdrew, by months before the event; dashed = everyone else.")
            w1, w2 = V.warning_signal(wsig)
            c1, c2 = st.columns(2)
            with c1:
                chart(w1)
            with c2:
                chart(w2)
    if seg is not None:
        a_ = seg[seg.segment == "age_band"].set_index("group").loc[["under 25", "25 to 34", "35 to 49", "50 to 64", "65+"]]
        allrow = seg[seg.segment == "all"].iloc[0]
        with st.container(border=True):
            card_title("3. The most fragile customers are middle aged, higher earners",
                       f"Laid off on EI: share who run out within 6 months. All customers: {allrow.short_ei55:.0%} by the "
                       f"formula, {allrow.short_ei_realistic:.0%} with the real EI cap (canada.ca, 2026).")
            chart(V.fragility_age(a_))
        l_ = seg[seg.segment == "loan"].set_index("group").loc[["no loan", "repaid", "running OK", "running in debt", "defaulted"]]
        with st.container(border=True):
            card_title("4. Borrowers in good standing are already twice as exposed",
                       "Loan status from loans.csv joined to the judges' runway. Loan status is never a model input.")
            chart(V.borrowers(l_))
    mg = load_migration()
    if mg is not None:
      with st.container(border=True):
        card_title("5. Distress is sticky")
        mk = mg["markov_check"]
        c1, c2 = st.columns([1, 1])
        c1.dataframe(pd.DataFrame({"months later": [3, 6, 12],
                                   "model predicts": [f"{mk[str(k_)]['pred'][2][2]:.0%}" for k_ in (3, 6, 12)],
                                   "actually distressed": [f"{mk[str(k_)]['actual'][2][2]:.0%}" for k_ in (3, 6, 12)]}),
                     hide_index=True, width="stretch")
        c2.markdown("Customers with under 1 month of buffer, or overdrawn, stay that way far longer than a standard "
                    "migration matrix predicts: 43% are still there a year later. Once a customer falls behind they stay "
                    "behind, which is why the early warning model aims to catch them months before.")
        c2.caption("Held out test year (Oct 2024 to Sep 2025); matrix estimated on the train years. Source: migration.py.")

# ---------------- how it's calculated ----------------
with tab_i:
    section("Show your work", "How every number is calculated",
            "Each section shows the formula, then this account and the current twist plugged in, then the evidence.")

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
    section("Grow", "Cushion first, then invest the rest",
            "Illustration, not advice. Model portfolios of broad index ETFs, no stock picking. "
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

        c1, c2 = st.columns(2, gap="medium")
        with c1, st.container(border=True):
            card_title("1. The cushion", f"{pl['cushion_months']:g} months of spending: 3 if income is steady, 4.5 if "
                       f"somewhat irregular, 6 if irregular. Income variation here: {pl['income_cv']:.0%}.")
            kpi_band([("Target", money(pl["cushion"]), f"{pl['cushion_months']:g} months", ""),
                      ("Buffer now", money(p["B"]), "", "")])
            st.progress(float(min(1.0, p["B"] / pl["cushion"])) if pl["cushion"] > 0 else 1.0,
                        text="Cushion full" if pl["cushion_gap"] == 0 else f"{money(pl['cushion_gap'])} short of the cushion")
        with c2, st.container(border=True):
            card_title("2. Risk capacity: what they can afford", "From their own cash flows, never demographics.")
            kpi_band([("Capacity", pl["capacity"], "", "bad" if pl["capacity"] == "Low" else ""),
                      ("Portfolio", pl["level"], f"asked for {pl['tolerance']}" if pl["capped"] else "their choice", "")])
            st.markdown("<span style='font-size:0.88rem'>Why: " + "; ".join(pl["why"]) + ".</span>", unsafe_allow_html=True)

        with st.container(border=True):
            card_title("3. The plan")
            if not pl["investable"]:
                st.info(pl["summary"])
            else:
                kpi_band([("Upfront", money(pl["upfront"]), "buffer above the cushion", ""),
                          ("Weekly", money(pl["weekly"]), "half the surplus" + (f", after {pl['fill_months']} months"
                                                                                if pl["fill_months"] else ""), ""),
                          (f"Likely in {yrs or 10} years", money(pl["end_p50"]), f"bad case {money(pl['end_p10'])}", "good"),
                          ("Same money in savings", money(pl["end_savings"]), f"at {PF.SAVINGS_RATE:.1%} a year", "")])
                st.write(pl["summary"])
                f = pl["fan"]
                chart(V.fan(f.month / 12, (f.p10, f.p90), (f.p25, f.p75), f.p50,
                            {f"savings account at {PF.SAVINGS_RATE:.1%}": (f.savings, V.S2, True),
                             "money put in": (f.paid_in, V.REF, False)},
                            "years", names=("80% of simulated outcomes", "middle 50%", f"{pl['level']} portfolio, median"),
                            zero=False, height=280))
                st.caption(f"5,000 paths replaying 12 month blocks of the ETFs' real monthly total returns "
                           f"({pl['market_from']} to {pl['market_to']}, {pl['market_months']} months, from yfinance). "
                           f"Chance of ending below the money put in: {pl['p_below_paid']:.0%}. "
                           f"Chance of beating savings: {pl['p_beats_savings']:.0%}.")

        with st.container(border=True):
            card_title("The three model portfolios", "Historical, 2015 to 2026, in Canadian dollars. 2022 hit every portfolio "
                       "because bonds and stocks fell together, which is why even Conservative lost 11% over its worst 12 months.")
            stats = PF.portfolio_stats(load_market())
            hold = pd.DataFrame(PF.PORTFOLIOS).T.rename(columns=PF.TICKERS).map("{:.0%}".format)
            st.dataframe(hold, width="stretch")
            st.dataframe(stats.assign(**{c: stats[c].map("{:.1%}".format) for c in
                                         ["annual_return", "volatility", "worst_12_months", "max_drawdown"]})
                         .rename(columns={"annual_return": "return per year", "worst_12_months": "worst 12 months",
                                          "max_drawdown": "worst drop"}), hide_index=True, width="stretch")
        cap = load_csv("capacity.csv")
        if cap is not None:
            st.markdown(f"**Across all customers:** {cap.below_cushion.mean():.0%} are below their own cushion; "
                        f"{(cap.cushion_months > 3).mean():.0%} need more than the usual 3 months because their income "
                        f"is irregular; {cap.nothing.mean():.0%} have nothing to invest yet. Risk capacity: "
                        + ", ".join(f"{k} {v:.0%}" for k, v in cap.capacity.value_counts(normalize=True).items()) + ".")
