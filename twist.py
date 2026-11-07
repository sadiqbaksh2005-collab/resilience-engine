"""
Twist card parser: plain English -> slider settings. Rule based, so it is instant,
works offline and always gives the same answer for the same card.

    params, understood, unread = parse("They lose their job (EI at 55%) and their household bills go up 15%")
    # params    -> {"inc_pct": 55, "bills_pct": 15}
    # understood -> ["Income drops to 55% of usual (job loss on EI)", "Household bills up 15%"]

Keys match the app's sliders: inc_pct, ei_months, income_after_pct, income_change, bills_pct,
bills_abs, spending_pct, extra_monthly, loan_pct, one_off, one_off_S, plus one_off_months_I
(converted to $ by the app, since it needs the customer's income).
"""
from __future__ import annotations
import re

WORDNUM = {"a": 1, "an": 1, "one": 1, "a full": 1, "a whole": 1, "full": 1, "two": 2, "three": 3, "four": 4,
           "five": 5, "six": 6, "seven": 7, "eight": 8, "nine": 9, "ten": 10, "twelve": 12, "eighteen": 18,
           "half a": 0.5, "half": 0.5, "a half": 0.5, "one and a half": 1.5, "a couple of": 2, "couple of": 2,
           "a few": 3, "several": 3}
_NUM = r"(\d+(?:\.\d+)?|" + "|".join(sorted(map(re.escape, WORDNUM), key=len, reverse=True)) + r")"

PCT = re.compile(r"(\d+(?:\.\d+)?)\s*(?:%|per\s?cent)")
MONEY = re.compile(r"\$\s?(\d[\d,]*(?:\.\d+)?)\s*(k|thousand)?|(\d[\d,]*(?:\.\d+)?)\s*(k|thousand)?\s*(?:dollars|\$)")
MONTHS = re.compile(_NUM + r"\s*(?:-|\s)?\s*months?\b")
FOR_MONTHS = re.compile(r"(?:for|lasting|over|during)\s+(?:the\s+(?:next|first)\s+)?" + _NUM + r"\s*months?")
MONTHS_OF = re.compile(_NUM + r"\s+months?\s+(?:worth\s+)?of\s+(?:their\s+|his\s+|her\s+)?(?:typical\s+|usual\s+|normal\s+|average\s+|monthly\s+)?"
                       r"(spending|expenses|costs|income|pay|salary|earnings)")
PER_MONTH = re.compile(r"(?:per|a|each|every|/)\s*month|monthly|/mo\b")

JOB_LOSS = re.compile(r"laid off|lay ?off|lose[s]? (?:their|his|her|the)? ?job|lost (?:their|his|her) job|job loss|unemploy|fired|"
                      r"\bei\b|employment insurance|out of work")
PARTNER = re.compile(r"partner|spouse|husband|wife|second income|other earner")
INCOME_CUT = re.compile(r"pay cut|income (?:drops|falls|goes down|cut|is cut|decreases)|hours (?:cut|reduced)|"
                        r"salary (?:cut|drops)|earn(?:s|ing)? less|(?:drop|fall|cut) in (?:income|pay)")
RAISE = re.compile(r"raise|pay rise|promotion|income (?:rises|goes up|increases)|earn(?:s)? more")
BACK_TO_WORK = re.compile(r"back to work|finds? (?:a )?(?:new )?job|re-?employ|return(?:s)? to work")
BILLS = re.compile(r"household bills?|bills?|rent|utilit|hydro|heating|electric|mortgage(?! payment)")
INFLATION = re.compile(r"inflation|prices?|cost of living|everything costs|all (?:their )?spending|groceries")
LOAN = re.compile(r"loan payment|loan|interest rate|rate (?:hike|rise|reset)|mortgage payment")
ONE_OFF = re.compile(r"repair|furnace|roof|car (?:breaks|dies|repair)|breaks? down|dies|broke|emergency|surprise|"
                     r"unexpected|one[- ]?(?:off|time)|vet|medical|dental|funeral|fine\b|ticket|replace")
NEW_COST = re.compile(r"baby|child|daycare|childcare|new car|car payment|tuition|caregiv|elder|new (?:monthly|recurring) cost|"
                      r"subscription|moves? in|pet")


def _num(tok: str) -> float:
    tok = tok.strip()
    try:
        return float(tok)
    except ValueError:
        return float(WORDNUM[tok])


def _money(text: str) -> float | None:
    m = MONEY.search(text)
    if not m:
        return None
    val = float((m.group(1) or m.group(3)).replace(",", ""))
    if m.group(2) or m.group(4):
        val *= 1000
    return val


def _clauses(text: str) -> list[str]:
    t = text.lower().replace("’", "'")
    t = re.sub(r"[()\[\]\"]", " , ", t)
    parts = re.split(r"(?<!\d)[.;!?](?!\d)|,(?=\s)|\band (?:then|also)?\b|\bplus\b|\bwhile\b|\bmeanwhile\b|\bat the same time\b", t)
    return [re.sub(r"\s+", " ", p).strip() for p in parts if p and p.strip()]


def parse(text: str):
    params: dict[str, float] = {}
    understood: list[str] = []
    unread: list[str] = []
    pending_one_off = False
    pending_partner = False
    job_loss_seen = False

    def note(msg):
        if msg not in understood:
            understood.append(msg)

    for c in _clauses(text):
        hit = False
        pct = PCT.search(c)
        money = _money(c)
        dur = FOR_MONTHS.search(c)
        mo = MONTHS_OF.search(c)

        # amount for a partner job loss stated in the next clause ("..., losing $2,000 a month")
        if pending_partner and (money or pct):
            pending_partner = False
            if money:
                params["income_change"] = -money
                note(f"Partner's income lost: -${money:,.0f}/month")
            else:
                params["inc_pct"] = int(round(100 - float(pct.group(1))))
                note(f"Household income drops to {params['inc_pct']}% (partner's share)")
            continue

        # ---- partner loses income (dollar amount, or half of income as a stated assumption) ----
        if PARTNER.search(c) and (JOB_LOSS.search(c) or INCOME_CUT.search(c) or "income" in c):
            hit = True
            if money:
                params["income_change"] = -money
                note(f"Partner's income lost: -${money:,.0f}/month")
            elif pct:
                params["inc_pct"] = int(round(100 - float(pct.group(1))))
                note(f"Household income drops to {params['inc_pct']}% (partner's share)")
            else:
                pending_partner = True
        # ---- job loss / EI ----
        elif JOB_LOSS.search(c):
            hit = True
            job_loss_seen = True
            rate = float(pct.group(1)) if pct else params.get("inc_pct", 55)
            params["inc_pct"] = int(round(rate))
            note(f"Income drops to {params['inc_pct']}% of usual (job loss on EI)")
        # ---- pay cut / raise ----
        elif INCOME_CUT.search(c) and pct:
            hit = True
            params["inc_pct"] = int(round(100 - float(pct.group(1))))
            note(f"Income drops {pct.group(1)}% (to {params['inc_pct']}% of usual)")
        elif INCOME_CUT.search(c) and money:
            hit = True
            params["income_change"] = params.get("income_change", 0) - money
            note(f"Income down ${money:,.0f}/month")
        elif RAISE.search(c) and (pct or money):
            hit = True
            if pct:
                params["inc_pct"] = int(round(100 + float(pct.group(1))))
                note(f"Income up {pct.group(1)}%")
            else:
                params["income_change"] = params.get("income_change", 0) + money
                note(f"Income up ${money:,.0f}/month")
        # ---- loan payment ----
        elif LOAN.search(c) and (pct or money) and not ONE_OFF.search(c):
            hit = True
            if pct:
                params["loan_pct"] = int(round(float(pct.group(1))))
                note(f"Loan payment up {params['loan_pct']}%")
            else:
                params["extra_monthly"] = params.get("extra_monthly", 0) + money
                note(f"Loan payment up ${money:,.0f}/month")
        # ---- bills / rent ----
        elif BILLS.search(c) and (pct or money) and not ONE_OFF.search(c):
            hit = True
            if pct:
                params["bills_pct"] = int(round(float(pct.group(1))))
                note(f"Household bills up {params['bills_pct']}%")
            else:
                params["bills_abs"] = params.get("bills_abs", 0) + money
                note(f"Household bills up ${money:,.0f}/month")
        # ---- inflation ----
        elif INFLATION.search(c) and pct:
            hit = True
            params["spending_pct"] = int(round(float(pct.group(1))))
            note(f"All spending up {params['spending_pct']}%")
        # ---- one off cost ----
        elif ONE_OFF.search(c) or (pending_one_off and (money or mo)):
            hit = True
            if mo and mo.group(2) in ("spending", "expenses", "costs"):
                params["one_off_S"] = params.get("one_off_S", 0) + _num(mo.group(1))
                note(f"One off cost = {_num(mo.group(1)):g} month(s) of typical spending")
                pending_one_off = False
            elif mo:
                params["one_off_months_I"] = params.get("one_off_months_I", 0) + _num(mo.group(1))
                note(f"One off cost = {_num(mo.group(1)):g} month(s) of typical income")
                pending_one_off = False
            elif money:
                params["one_off"] = params.get("one_off", 0) + money
                note(f"One off cost ${money:,.0f}")
                pending_one_off = False
            else:
                pending_one_off = True          # "the furnace dies" -> amount comes in the next clause
        # ---- new recurring cost ----
        elif (NEW_COST.search(c) or PER_MONTH.search(c)) and money:
            hit = True
            params["extra_monthly"] = params.get("extra_monthly", 0) + money
            note(f"New recurring cost ${money:,.0f}/month")

        # ---- duration and what happens after (can sit in any clause) ----
        if dur and (job_loss_seen or "inc_pct" in params):
            params["ei_months"] = int(round(_num(dur.group(1))))
            note(f"Reduced income lasts {params['ei_months']} months")
            hit = True
        if BACK_TO_WORK.search(c):
            params["income_after_pct"] = 100
            note("Back to full income afterwards")
            hit = True

        if not hit and (PCT.search(c) or _money(c) or MONTHS.search(c)):
            unread.append(c)

    if pending_partner:
        params["inc_pct"] = 50
        params.setdefault("ei_months", 0)
        note("Partner's income lost with no amount given: assumed half of household income")
    if pending_one_off:
        params.setdefault("one_off_S", 1.0)
        note("One off cost with no amount given: assumed 1 month of typical spending")
    if "inc_pct" in params and params["inc_pct"] != 100 and "ei_months" not in params:
        params["ei_months"] = 6 if job_loss_seen else 0
        note("Duration not given: 6 months of EI assumed" if job_loss_seen else "Lasts indefinitely")
    return params, understood, unread


if __name__ == "__main__":
    import sys
    print(parse(" ".join(sys.argv[1:])))
