"""
Streamlit front end for flowtracker.

Run locally:   streamlit run app.py
Deploy:        push to GitHub, connect at share.streamlit.io

Reuses the chart and metric code from flowtracker.py rather than duplicating it,
so fixes to the analysis apply to both the terminal and the web version.
"""

from __future__ import annotations

import datetime as dt
import json

import pandas as pd
import streamlit as st
import streamlit.components.v1 as components

import flowtracker as ft

st.set_page_config(page_title="Sector rotation", layout="wide",
                   initial_sidebar_state="expanded")

VIEWS = {
    "Sectors": "11 GICS sectors with their industry funds",
    "My portfolio": "your positions on their own scale",
    "Full board": "sectors, countries, bonds, commodities, crypto",
    "Custom": "pick your own tickers",
}


@st.cache_data(ttl=6 * 3600, show_spinner="Fetching prices…")
def load_prices(tickers: tuple, years: int) -> dict:
    """One ticker at a time, cached for six hours."""
    ft.LOOKBACK_YEARS = years
    ft.CACHE_DIR = f"cache_{years}y"
    out = ft.fetch_prices(list(tickers), demo=False)
    return {t: df for t, df in out.items()}


def render_chart(prices: dict, bench: pd.Series, hold: list,
                 buckets: dict | None, weeks: int, smooth: int,
                 axes: str, height: int) -> None:
    data = ft.build_trails(prices, bench, weeks, axes, smooth, hold, buckets)
    if not data["series"]:
        st.warning("Not enough history for these tickers.")
        return
    html = ft.TRAILS_HTML.replace("__DATA__", json.dumps(data))
    components.html(html, height=height, scrolling=True)


# --------------------------------------------------------------------------
# sidebar
# --------------------------------------------------------------------------

st.sidebar.title("Rotation")
view = st.sidebar.radio("View", list(VIEWS), index=0,
                        captions=list(VIEWS.values()))

weeks = st.sidebar.slider("Window (weeks)", 8, 104, 26, step=2)
smooth = st.sidebar.slider("Smoothing (days)", 1, 20, 5,
                           help="Rolling mean applied to both axes. Higher is "
                                "calmer but lags more.")
axes = st.sidebar.selectbox(
    "Axes", ["classic", "rrg"],
    format_func=lambda a: ("3m relative strength / % above 200d"
                           if a == "classic"
                           else "relative strength / its momentum"),
    help="The classic vertical axis still carries market beta — a broad rally "
         "lifts every dot at once. The rrg version removes it.")

extra = st.sidebar.text_input(
    "Add tickers", "",
    placeholder="NVDA, ASML, NESN.SW",
    help="Any Yahoo ticker. Non-US listings need a suffix: .SW, .PA, .L")

st.sidebar.divider()
st.sidebar.caption(
    "Price data from Yahoo via yfinance, which fails quietly when it breaks. "
    "Cross-check anything surprising against your broker before acting on it. "
    "Not investment advice."
)

# --------------------------------------------------------------------------
# assemble the ticker set
# --------------------------------------------------------------------------

added = [t.strip().upper() for t in extra.replace(";", ",").split(",") if t.strip()]
for t in added:
    if t not in ft.ALL:
        ft.ALL[t] = (t, "added")

if view == "Sectors":
    bmap = {t: sec for sec, ts in ft.SECTOR_MAP.items() for t in ts}
    wanted = list(bmap) + added
    hold_src = [ts[0] for ts in ft.SECTOR_MAP.values()]
elif view == "My portfolio":
    bmap = None
    wanted = ft.PORTFOLIO_VIEW + added
    hold_src = ft.PORTFOLIO_VIEW
elif view == "Full board":
    bmap = None
    wanted = list(ft.ALL)
    hold_src = ft.PORTFOLIO_VIEW
else:
    bmap = None
    picks = st.multiselect(
        "Tickers", sorted(ft.ALL),
        default=["XLK", "XLE", "XLV", "GLD"],
        format_func=lambda t: f"{t} — {ft.ALL[t][0]}")
    wanted = picks + added
    hold_src = added or picks

if not wanted:
    st.info("Pick at least one ticker.")
    st.stop()

years = 4 if weeks <= 78 else 6
prices = load_prices(tuple(sorted(set(wanted + [ft.BENCHMARK]))), years)

missing = [t for t in wanted if t not in prices]
if ft.BENCHMARK not in prices:
    st.error(f"Benchmark {ft.BENCHMARK} unavailable — Yahoo may be rate-limiting. "
             "Wait a few minutes and reload.")
    st.stop()

bench = prices[ft.BENCHMARK]["close"]
for t in list(prices):
    if t in ft.CRYPTO:
        prices[t] = prices[t].reindex(bench.index).ffill().dropna(subset=["close"])

chart = {t: prices[t] for t in wanted if t in prices}
hold = [t for t in hold_src if t in chart]

# --------------------------------------------------------------------------
# main
# --------------------------------------------------------------------------

st.subheader(view)
if missing:
    st.warning("No data returned for: " + ", ".join(missing))

tab_chart, tab_table = st.tabs(["Chart", "Table"])

with tab_chart:
    render_chart(chart, bench, hold, bmap, weeks, smooth, axes, height=780)

with tab_table:
    rows = []
    for t, df in chart.items():
        px = df["close"].dropna()
        rr = (ft._classic_series(px, bench, smooth) if axes == "classic"
              else ft._rrg_series(px, bench))
        if rr.empty:
            continue
        vol = px.pct_change().tail(126).std() * (252 ** 0.5)
        rows.append({
            "ticker": t,
            "name": ft.ALL[t][0],
            "x": round(float(rr.x.iloc[-1]), 1),
            "y": round(float(rr.y.iloc[-1]), 1),
            "quadrant": ft._quad(rr.x.iloc[-1], rr.y.iloc[-1]),
            "3m %": round(ft._ret(px, 3) * 100, 1) if ft._ret(px, 3) == ft._ret(px, 3) else None,
            "12m %": round(ft._ret(px, 12) * 100, 1) if ft._ret(px, 12) == ft._ret(px, 12) else None,
            "vol %": round(vol * 100, 0),
        })
    if rows:
        tbl = pd.DataFrame(rows).sort_values("x", ascending=False)
        st.dataframe(tbl, use_container_width=True, hide_index=True)
        st.download_button("Download CSV", tbl.to_csv(index=False),
                           f"rotation_{dt.date.today()}.csv", "text/csv")
        st.caption(
            "Volatility is shown because the ranking is not risk-adjusted: "
            "high-volatility names travel further on both axes regardless of "
            "whether anything meaningful is happening."
        )
