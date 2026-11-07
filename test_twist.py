"""Twist parser tests: the three cards from the brief plus likely variations."""
from twist import parse

CASES = [
    ("This customer was laid off. EI pays 55% of their usual income for 6 months. Do they make it?",
     {"inc_pct": 55, "ei_months": 6}),
    ("The furnace dies, and the repair costs a full month of their typical spending.",
     {"one_off_S": 1.0}),
    ("They lose their job (EI at 55%) and their household bills go up 15% in the same month.",
     {"inc_pct": 55, "bills_pct": 15, "ei_months": 6}),
    ("Rent goes up by $300 a month", {"bills_abs": 300}),
    ("Their income drops 20%", {"inc_pct": 80, "ei_months": 0}),
    ("A surprise car repair costs $2,500", {"one_off": 2500}),
    ("They have a baby, which adds $800 per month", {"extra_monthly": 800}),
    ("Inflation pushes all prices up 4%", {"spending_pct": 4}),
    ("Their loan payment goes up 20% after a rate reset", {"loan_pct": 20}),
    ("Their partner loses their job, losing $2,000 a month", {"income_change": -2000}),
    ("Laid off with EI at 55% for 4 months, then they find a new job", {"inc_pct": 55, "ei_months": 4, "income_after_pct": 100}),
    ("The roof needs replacing: two months of spending", {"one_off_S": 2.0}),
    ("Laid off, EI 55% for six months, rent up 10%, and the car breaks down costing $1.5k",
     {"inc_pct": 55, "ei_months": 6, "bills_pct": 10, "one_off": 1500}),
    ("Medical emergency costing half a month of income", {"one_off_months_I": 0.5}),
    ("They get a 10% raise", {"inc_pct": 110, "ei_months": 0}),
]

fails = 0
for text, want in CASES:
    got, understood, unread = parse(text)
    ok = all(abs(got.get(k, -999) - v) < 1e-9 for k, v in want.items()) and set(got) == set(want)
    if not ok:
        fails += 1
        print(f"FAIL: {text}\n   want {want}\n   got  {got}\n   unread {unread}")
print(f"{len(CASES) - fails}/{len(CASES)} twist cards parsed exactly")
assert fails == 0
