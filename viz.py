"""
Native, themed Altair charts for the app (no images): transparent background so they sit on the
page's cards, the app's fonts, hover tooltips on every chart.

Data palette (validated: lightness band, chroma floor, CVD and normal vision separation, contrast):
    S1  #24699F  denim blue   our engine / the customer
    S2  #C4673F  terracotta   second series, alerts
    REF #8C7A6B  taupe        the judges' formula / reference lines (3.7:1 on the page)
"""
from __future__ import annotations
import altair as alt
import numpy as np
import pandas as pd

S1, S2, REF = "#24699F", "#C4673F", "#8C7A6B"
BAND = ["#D9E4EE", "#AFC7DC"]                       # light tints of S1 for simulation bands
INK, INK2, GRID, SHADE = "#2E3029", "#6E6A60", "#E6DDD0", "#EFE7DC"
SANS = "-apple-system, BlinkMacSystemFont, 'Helvetica Neue', Helvetica, Arial, sans-serif"


def themed(ch, height=None):
    if height:
        ch = ch.properties(height=height)
    return (ch.configure(background="transparent", font=SANS, padding={"left": 2, "right": 8, "top": 6, "bottom": 2})
            .configure_view(stroke=None)
            .configure_axis(grid=True, gridColor=GRID, gridWidth=0.8, domain=False, ticks=False,
                            labelColor=INK2, titleColor=INK2, labelFontSize=11, titleFontSize=11,
                            titleFontWeight="normal", labelPadding=6, titlePadding=8)
            .configure_axisX(grid=False)
            .configure_legend(orient="top", direction="horizontal", title=None, labelColor=INK2, labelFontSize=11,
                              symbolStrokeWidth=2.5, symbolSize=120, columnPadding=18, offset=6, labelLimit=320)
            .configure_text(font=SANS, color=INK2))


def _months(s: pd.Series) -> pd.Series:
    return pd.to_datetime(s.astype(str) + "-01")


def _hover_rule(df, x, xtype="T"):
    """Vertical crosshair that follows the pointer (nearest x)."""
    sel = alt.selection_point(nearest=True, on="pointerover", fields=[x], empty=False, clear="pointerout")
    rule = alt.Chart(df).mark_rule(color=INK2, strokeWidth=1, opacity=0.5).encode(
        x=f"{x}:{xtype}", opacity=alt.condition(sel, alt.value(0.5), alt.value(0))).add_params(sel)
    return rule


def _legend_scale(names, colors):
    return alt.Scale(domain=list(names), range=list(colors))


# ---------------- customer ----------------
def balance_history(h: pd.DataFrame):
    d = pd.DataFrame({"month": _months(h["month"]), "Month end balance": h["month_end_balance"],
                      "Lowest balance in the month": h["min_balance"]})
    scale = _legend_scale(["Month end balance", "Lowest balance in the month"], [S1, REF])
    base = alt.Chart(d).encode(x=alt.X("month:T", title=None, axis=alt.Axis(format="%Y", tickCount="year")))
    yax = alt.Y("Month end balance:Q", title=None, axis=alt.Axis(format="$~s"))
    low = base.mark_line(strokeWidth=1.6, strokeDash=[4, 3], interpolate="linear").encode(
        y="Lowest balance in the month:Q", color=alt.Color("s:N", scale=scale)).transform_calculate(s="'Lowest balance in the month'")
    end = base.mark_line(strokeWidth=2.2, interpolate="linear").encode(
        y=yax, color=alt.Color("s:N", scale=scale)).transform_calculate(s="'Month end balance'")
    lines = alt.layer(end, low)
    tip = alt.Chart(d).mark_rule(opacity=0, strokeWidth=12).encode(
        x="month:T", tooltip=[alt.Tooltip("month:T", format="%b %Y", title="Month"),
                              alt.Tooltip("Month end balance:Q", format="$,.0f"),
                              alt.Tooltip("Lowest balance in the month:Q", format="$,.0f")])
    zero = alt.Chart(pd.DataFrame({"y": [0]})).mark_rule(color=INK, strokeWidth=1).encode(y="y:Q")
    return themed(alt.layer(zero, lines, _hover_rule(d, "month"), tip), 230)


def money_in_out(h: pd.DataFrame):
    d = pd.DataFrame({"month": _months(h["month"]), "Money in": h["money_in"], "Money out": h["money_out"]})
    long = d.melt("month", var_name="series", value_name="amount")
    scale = _legend_scale(["Money in", "Money out"], [S1, S2])
    lines = alt.Chart(long).mark_line(strokeWidth=1.8, interpolate="linear").encode(
        x=alt.X("month:T", title=None, axis=alt.Axis(format="%Y", tickCount="year")),
        y=alt.Y("amount:Q", title=None, axis=alt.Axis(format="$~s")),
        color=alt.Color("series:N", scale=scale))
    tip = alt.Chart(d).mark_rule(opacity=0, strokeWidth=12).encode(
        x="month:T", tooltip=[alt.Tooltip("month:T", format="%b %Y", title="Month"),
                              alt.Tooltip("Money in:Q", format="$,.0f"), alt.Tooltip("Money out:Q", format="$,.0f")])
    return themed(alt.layer(lines, _hover_rule(d, "month"), tip), 190)


# ---------------- fan charts (twist, investing) ----------------
def fan(x, outer, inner, median, lines: dict, x_title: str, names=("90% of simulated paths", "middle 50%",
        "median simulated path"), zero=True, marker=None, height=300, y_fmt="$~s"):
    """outer/inner = (lo, hi) arrays; lines = {name: (values, color, dashed)}."""
    d = pd.DataFrame({"x": x, "lo_o": outer[0], "hi_o": outer[1], "lo_i": inner[0], "hi_i": inner[1], "med": median})
    keys = {k: f"l{j}" for j, k in enumerate(lines)}       # safe column names (labels may contain '.' or '%')
    for k, (vals, _, _) in lines.items():
        d[keys[k]] = vals
    dom = list(names) + list(lines)
    rng = BAND + [S1] + [c for _, c, _ in lines.values()]
    scale = _legend_scale(dom, rng)
    base = alt.Chart(d).encode(x=alt.X("x:Q", title=x_title, axis=alt.Axis(tickMinStep=1, grid=False)))
    yax = alt.Axis(format=y_fmt)
    o = base.mark_area(opacity=1).encode(y=alt.Y("lo_o:Q", title=None, axis=yax), y2="hi_o:Q",
                                         color=alt.Color("s:N", scale=scale)).transform_calculate(s=f"'{names[0]}'")
    i = base.mark_area(opacity=1).encode(y="lo_i:Q", y2="hi_i:Q", color=alt.Color("s:N", scale=scale)
                                         ).transform_calculate(s=f"'{names[1]}'")
    m = base.mark_line(strokeWidth=2.2).encode(y="med:Q", color=alt.Color("s:N", scale=scale)
                                               ).transform_calculate(s=f"'{names[2]}'")
    layers = [o, i, m]
    for k, (_, c, dashed) in lines.items():
        layers.append(base.mark_line(strokeWidth=2.2, strokeDash=[6, 4] if dashed else [1, 0]).encode(
            y=f"{keys[k]}:Q", color=alt.Color("s:N", scale=scale)).transform_calculate(s=f"'{k}'"))
    if zero:   # drawn above the bands, below the lines
        layers.insert(2, alt.Chart(pd.DataFrame({"y": [0]})).mark_rule(color=INK, strokeWidth=1).encode(y="y:Q"))
    if marker is not None:
        mx, mlabel = marker
        mk = pd.DataFrame({"x": [mx], "t": [mlabel]})
        layers.append(alt.Chart(mk).mark_rule(color=INK2, strokeDash=[2, 3]).encode(x="x:Q"))
        layers.append(alt.Chart(mk).mark_text(align="left", dx=5, dy=8, baseline="top", fontSize=11, color=INK2)
                      .encode(x="x:Q", y=alt.value(0), text="t:N"))
    tips = [alt.Tooltip("x:Q", title=x_title), alt.Tooltip("med:Q", format="$,.0f", title="median"),
            alt.Tooltip("lo_o:Q", format="$,.0f", title="bad case"), alt.Tooltip("hi_o:Q", format="$,.0f", title="good case")]
    tips += [alt.Tooltip(f"{keys[k]}:Q", format="$,.0f", title=k) for k in lines]
    layers.append(_hover_rule(d, "x", "Q"))
    layers.append(base.mark_rule(opacity=0, strokeWidth=14).encode(tooltip=tips))
    return themed(alt.layer(*layers), height)


# ---------------- model ----------------
def risk_history(h: pd.DataFrame, thr: float):
    d = h.copy()
    d["date"] = _months(d["month"])
    neg = (d.min_balance < 0).to_numpy()
    fe = int(np.argmax(neg)) if neg.any() else None
    keep = np.arange(len(d)) < fe if fe is not None else np.ones(len(d), bool)
    d["risk_pct"] = (d.risk * 100).where(keep)
    d["flagged"] = d.risk >= thr
    d["runway"] = d.runway_asof.clip(upper=120)
    x = alt.X("date:T", title=None, axis=alt.Axis(format="%Y", tickCount="year"))
    layers_top, layers_bot = [], []
    n_train = int((d.period == "train").sum())
    if n_train:
        sh = pd.DataFrame({"a": [d.date.iloc[0]], "b": [d.date.iloc[n_train - 1]], "t": ["trained on this period"],
                           "t2": ["never seen by the model"], "c": [d.date.iloc[min(n_train, len(d) - 1)]]})
        rect = alt.Chart(sh).mark_rect(color=SHADE).encode(x="a:T", x2="b:T")
        layers_top += [rect, alt.Chart(sh).mark_text(align="left", baseline="top", dx=4, dy=4, fontSize=10, color=INK2)
                       .encode(x="a:T", y=alt.value(0), text="t:N"),
                       alt.Chart(sh).mark_text(align="left", baseline="top", dx=4, dy=4, fontSize=10, color=INK2)
                       .encode(x="c:T", y=alt.value(0), text="t2:N")]
        layers_bot.append(rect)
    scale = _legend_scale(["Risk of a first overdraft, next 3 months", "Flagged"], [S1, S2])
    yr = alt.Y("risk_pct:Q", title="risk, log scale", scale=alt.Scale(type="log"),
                   axis=alt.Axis(values=[0.01, 0.03, 0.1, 0.3, 1, 3, 10, 30, 100], labelExpr="datum.value + '%'"))
    line = alt.Chart(d).mark_line(strokeWidth=2).encode(x=x, y=yr, color=alt.Color("s:N", scale=scale)
                                                       ).transform_calculate(s="'Risk of a first overdraft, next 3 months'")
    pts = alt.Chart(d[d.flagged & d.risk_pct.notna()]).mark_circle(size=42, opacity=1).encode(
        x=x, y="risk_pct:Q", color=alt.Color("s:N", scale=scale)).transform_calculate(s="'Flagged'")
    th = pd.DataFrame({"y": [thr * 100], "t": [f"alert threshold {thr:.1%}"]})
    layers_top += [alt.Chart(th).mark_rule(color=REF, strokeDash=[5, 4]).encode(y="y:Q"),
                   alt.Chart(th).mark_text(align="right", dy=-6, fontSize=10, color=INK2).encode(
                       y="y:Q", x=alt.value("width"), text="t:N"), line, pts]
    tips = [alt.Tooltip("date:T", format="%b %Y", title="Month"), alt.Tooltip("risk:Q", format=".2%", title="risk"),
            alt.Tooltip("runway:Q", format=".1f", title="judges' runway (months)")]
    if fe is not None:
        od = pd.DataFrame({"date": [d.date.iloc[fe]], "t": ["first overdraft"]})
        ov = [alt.Chart(od).mark_rule(color="#9C3D2A", strokeWidth=1.5).encode(x="date:T"),
              alt.Chart(od).mark_text(align="left", baseline="top", dx=4, dy=18, fontSize=10, color="#9C3D2A")
              .encode(x="date:T", y=alt.value(0), text="t:N")]
        layers_top += ov
        layers_bot += ov[:1]
    layers_top.append(alt.Chart(d).mark_rule(opacity=0, strokeWidth=10).encode(x="date:T", tooltip=tips))
    top = alt.layer(*layers_top)
    rw = alt.Chart(d).mark_line(strokeWidth=2, color=REF, interpolate="step-after").encode(
        x=x, y=alt.Y("runway:Q", title="judges' runway", scale=alt.Scale(domain=[0, 125]),
                     axis=alt.Axis(values=[0, 12, 60, 120], labelExpr="datum.value == 120 ? '120+' : datum.label")))
    layers_bot += [rw, alt.Chart(d).mark_rule(opacity=0, strokeWidth=10).encode(x="date:T", tooltip=tips)]
    bot = alt.layer(*layers_bot)
    return themed(top, 230), themed(bot, 120), fe


def drivers(show: pd.Series, nice: dict):
    d = pd.DataFrame({"factor": [nice.get(f, f) for f in show.index], "effect": show.values})
    d["direction"] = np.where(d.effect > 0, "raises risk", "lowers risk")
    scale = _legend_scale(["raises risk", "lowers risk"], [S2, S1])
    bars = alt.Chart(d).mark_bar(cornerRadiusEnd=3, height=14).encode(
        y=alt.Y("factor:N", sort=alt.SortField("effect", order="ascending"), title=None, axis=alt.Axis(grid=False)),
        x=alt.X("effect:Q", title="effect on the risk score (log odds)", axis=alt.Axis(grid=True)),
        color=alt.Color("direction:N", scale=scale),
        tooltip=["factor:N", alt.Tooltip("effect:Q", format="+.2f")])
    zero = alt.Chart(pd.DataFrame({"x": [0]})).mark_rule(color=INK).encode(x="x:Q")
    return themed(alt.layer(bars, zero).configure_axisY(grid=False), 34 * len(d) + 30)


# ---------------- findings ----------------
def blind_spot(rows: pd.DataFrame):
    """rows: label, auc, lo, hi, kind ('judges' | 'ours')."""
    d = rows.copy()
    d["start"] = 0.5
    scale = _legend_scale(["Judges' formula", "Our engine"], [REF, S1])
    y = alt.Y("label:N", sort=list(d.label), title=None, axis=alt.Axis(grid=False, labelLimit=260, labelFontSize=12))
    bars = alt.Chart(d).mark_bar(height=22, cornerRadiusEnd=3).encode(
        y=y, x=alt.X("start:Q", scale=alt.Scale(domain=[0.5, 1.0]), axis=alt.Axis(values=[0.5, 0.6, 0.7, 0.8, 0.9, 1.0]), title="AUC on held out months (0.5 = coin flip, 1 = perfect)"),
        x2="auc:Q", color=alt.Color("kind:N", scale=scale),
        tooltip=["label:N", alt.Tooltip("auc:Q", format=".3f"), alt.Tooltip("lo:Q", format=".3f", title="95% CI low"),
                 alt.Tooltip("hi:Q", format=".3f", title="95% CI high")])
    ci = alt.Chart(d).mark_rule(color=INK, strokeWidth=1.4).encode(y=y, x="lo:Q", x2="hi:Q")
    txt = alt.Chart(d).mark_text(align="left", dx=6, fontSize=13, fontWeight="bold", color=INK).encode(
        y=y, x="hi:Q", text=alt.Text("auc:Q", format=".3f"))
    return themed(alt.layer(bars, ci, txt), 230)


def fragility_age(a: pd.DataFrame):
    d = a.reset_index().rename(columns={"index": "group"})
    d["label"] = d.apply(lambda r: f"{r.group} · ${r.median_income / 1e3:.1f}k income", axis=1)
    long = d.melt(id_vars=["group", "label"], value_vars=["short_ei55", "short_ei_realistic"], var_name="k", value_name="share")
    names = {"short_ei55": "Judges' formula: EI = 55% of income", "short_ei_realistic": "Realistic EI: $729/week cap, pension kept"}
    long["measure"] = long.k.map(names)
    scale = _legend_scale(list(names.values()), [REF, S1])
    x = alt.X("label:N", sort=list(d.label), title=None, axis=alt.Axis(labelAngle=0, grid=False, labelLimit=160))
    bars = alt.Chart(long).mark_bar(cornerRadiusEnd=3, width=26).encode(
        x=x, xOffset=alt.XOffset("measure:N", sort=list(names.values())),
        y=alt.Y("share:Q", title="share with under 6 months of runway", axis=alt.Axis(format="%"), scale=alt.Scale(domain=[0, 0.75])),
        color=alt.Color("measure:N", scale=scale),
        tooltip=["group:N", "measure:N", alt.Tooltip("share:Q", format=".1%")])
    txt = bars.mark_text(dy=-7, fontSize=11, color=INK).encode(text=alt.Text("share:Q", format=".0%"), color=alt.value(INK))
    return themed(alt.layer(bars, txt), 280)


def borrowers(l: pd.DataFrame):
    d = l.reset_index().rename(columns={"index": "group"})
    d["label"] = d.apply(lambda r: f"{r.group} (n={r.accounts:,})", axis=1)
    d["kind"] = np.where(d.group == "no loan", "No loan", "Has or had a loan")
    scale = _legend_scale(["No loan", "Has or had a loan"], [REF, S1])
    y = alt.Y("label:N", sort=list(d.label), title=None, axis=alt.Axis(grid=False, labelLimit=240, labelFontSize=12))
    bars = alt.Chart(d).mark_bar(height=20, cornerRadiusEnd=3).encode(
        y=y, x=alt.X("short_ei55:Q", title="share with under 6 months of runway if laid off (EI 55%)",
                     axis=alt.Axis(format="%"), scale=alt.Scale(domain=[0, 1])),
        color=alt.Color("kind:N", scale=scale), tooltip=["group:N", alt.Tooltip("short_ei55:Q", format=".1%", title="share"), "accounts:Q"])
    txt = alt.Chart(d).mark_text(align="left", dx=6, fontSize=12, fontWeight="bold", color=INK).encode(
        y=y, x="short_ei55:Q", text=alt.Text("short_ei55:Q", format=".0%"))
    return themed(alt.layer(bars, txt), 210)


def warning_signal(w: pd.DataFrame):
    def panel(col, title, fmt):
        d = w.assign(x=-w.months_before)
        win = pd.DataFrame({"a": [-6.5], "b": [-0.5]})
        shade = alt.Chart(win).mark_rect(color=SHADE).encode(x="a:Q", x2="b:Q")
        ref = alt.Chart(d.head(1)).mark_rule(color=REF, strokeDash=[5, 4], strokeWidth=1.5).encode(y=f"{col}_ref:Q")
        reft = alt.Chart(d.head(1)).mark_text(align="left", dx=4, dy=12, fontSize=10, color=INK2).encode(
            x=alt.value(0), y=f"{col}_ref:Q", text=alt.value("customers who never overdrew"))
        ln = alt.Chart(d).mark_line(color=S1, strokeWidth=2.2, point=alt.OverlayMarkDef(color=S1, size=40)).encode(
            x=alt.X("x:Q", title="months before the first overdraft", scale=alt.Scale(domain=[-12.5, -0.5]),
                    axis=alt.Axis(values=[-12, -9, -6, -3, -1], labelExpr="abs(datum.value)")),
            y=alt.Y(f"{col}:Q", title=None, axis=alt.Axis(format=fmt)),
            tooltip=[alt.Tooltip("months_before:Q", title="months before"), alt.Tooltip(f"{col}:Q", format=".2f", title=title)])
        return themed(alt.layer(shade, ref, reft, ln).properties(
            title=alt.Title(title, anchor="start", fontSize=12, fontWeight="normal", color=INK)), 220)
    return (panel("buffer", "Lowest balance, in months of income", ".1f"),
            panel("spend_ratio", "Spending / income, last 3 months", ".2f"))


def hero_fan(f: pd.DataFrame, start: float, height=440):
    """Decorative landing visual: one real customer's simulated balance after a layoff, no axes."""
    d = pd.DataFrame({"x": np.r_[0, f.month], "p05": np.r_[start, f.p05], "p95": np.r_[start, f.p95],
                      "p25": np.r_[start, f.p25], "p75": np.r_[start, f.p75], "p50": np.r_[start, f.p50]})
    x = alt.X("x:Q", axis=None, scale=alt.Scale(domain=[0, d.x.max()], nice=False))
    y = alt.Y("p05:Q", axis=None)
    o = alt.Chart(d).mark_area(color="#DCD3C2", interpolate="monotone").encode(x=x, y=y, y2="p95:Q")
    i = alt.Chart(d).mark_area(color="#C3BCA6", interpolate="monotone").encode(x=x, y="p25:Q", y2="p75:Q")
    m = alt.Chart(d).mark_line(color="#565A4B", strokeWidth=2.5, interpolate="monotone").encode(x=x, y="p50:Q")
    z = alt.Chart(pd.DataFrame({"y": [0]})).mark_rule(color="#A85E40", strokeWidth=1.2, strokeDash=[4, 4]).encode(y="y:Q")
    return (alt.layer(o, i, z, m).properties(height=height)
            .configure(background="transparent").configure_view(stroke=None))
