#!/usr/bin/env python3
"""
flowtracker.py — a sector / theme relative-strength scanner for a CHF-based investor.

What it actually does
---------------------
It ranks a universe of sector, theme, country and asset-class ETFs by where
capital is *relatively* going, using price and volume as the proxy (real
fund-flow data is not freely available). Everything is measured in CHF as well
as USD, because a Swiss investor's return is the local return times the FX move.

It also enforces the two guardrails that matter for a Swiss tax resident:
  - minimum 6-month holding period before a sale (Circular 36, criterion 1)
  - annual gross transaction volume < 5x opening portfolio value (criterion 2)

Usage
-----
    python flowtracker.py scan                 # rank the universe, write CSV
    python flowtracker.py scan --top 15
    python flowtracker.py portfolio            # concentration + KS36 guardrails
    python flowtracker.py scan --demo          # synthetic data, no network

Files it reads/writes (in the working directory):
    cache/*.csv        price cache, refreshed once a day
    holdings.csv       your positions: ticker,shares,cost_basis_chf,buy_date
    trades.csv         your trade log: date,ticker,side,amount_chf
    scan_YYYY-MM-DD.csv  ranked output

Not investment advice. This is a measurement tool, not a decision.
"""

from __future__ import annotations

import argparse
import datetime as dt
import os
import sys
import time
import warnings

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

# --------------------------------------------------------------------------
# CONFIG
# --------------------------------------------------------------------------

BENCHMARK = "ACWI"          # global equity, the thing a sector must beat
FX_PAIR = "CHF=X"           # USD/CHF: CHF per 1 USD
CACHE_DIR = "cache"
LOOKBACK_YEARS = 3

# ticker: (label, bucket)
# Buckets let you check you are not accidentally buying the same trade twice.
UNIVERSE = {
    # --- US GICS sectors (the standard 11-square chessboard) ---
    "XLK":  ("US Technology",            "tech"),
    "XLC":  ("US Communication Svcs",    "tech"),
    "XLY":  ("US Consumer Discretionary","cyclical"),
    "XLP":  ("US Consumer Staples",      "defensive"),
    "XLV":  ("US Health Care",           "defensive"),
    "XLF":  ("US Financials",            "financials"),
    "XLI":  ("US Industrials",           "cyclical"),
    "XLE":  ("US Energy",                "energy"),
    "XLB":  ("US Materials",             "materials"),
    "XLU":  ("US Utilities",             "defensive"),
    "XLRE": ("US Real Estate",           "real assets"),
    # --- narrower industries: this is where dispersion actually lives ---
    "SMH":  ("Semiconductors",           "tech"),
    "IGV":  ("Software",                 "tech"),
    "XBI":  ("Biotech (equal wt)",       "healthcare"),
    "IHI":  ("Medical Devices",          "healthcare"),
    "KRE":  ("Regional Banks",           "financials"),
    "ITA":  ("Aerospace & Defense",      "defense"),
    "IYT":  ("Transports",               "cyclical"),
    "XME":  ("Metals & Mining",          "materials"),
    "COPX": ("Copper Miners",            "materials"),
    "CPER": ("Copper (metal)",           "commodities"),
    "GDX":  ("Gold Miners",              "precious metals"),
    "URA":  ("Uranium",                  "energy"),
    "XOP":  ("Oil & Gas E&P",            "energy"),
    "ICLN": ("Clean Energy",             "energy"),
    "PAVE": ("US Infrastructure",        "real assets"),
    "JETS": ("Airlines",                 "cyclical"),
    # --- geography: the cheapest true diversification available ---
    "EWJ":  ("Japan",                    "intl dev"),
    "EZU":  ("Eurozone",                 "intl dev"),
    "EWU":  ("United Kingdom",           "intl dev"),
    "EWL":  ("Switzerland",              "intl dev"),
    "EWY":  ("South Korea",              "intl dev"),
    "EWT":  ("Taiwan",                   "intl dev"),
    "MCHI": ("China",                    "emerging"),
    "INDA": ("India",                    "emerging"),
    "EWZ":  ("Brazil",                   "emerging"),
    "EWW":  ("Mexico",                   "emerging"),
    "EWA":  ("Australia",                "intl dev"),
    # --- non-equity: the honest diversifiers ---
    "GLD":  ("Gold",                     "precious metals"),
    "SLV":  ("Silver",                   "precious metals"),
    "DBC":  ("Broad Commodities",        "commodities"),
    "TLT":  ("US 20y+ Treasuries",       "duration"),
    "IEF":  ("US 7-10y Treasuries",      "duration"),
    "TIP":  ("US TIPS",                  "duration"),
    "UUP":  ("US Dollar Index",          "currency"),
    # --- crypto: 24/7 series, realigned to the benchmark calendar below ---
    "BTC-USD":  ("Bitcoin",              "crypto"),
    "SOL-USD":  ("Solana",               "crypto"),
}

# Composite score weights. 12-1 momentum is the academically supported core;
# the rest are confirmation, not signal.
WEIGHTS = {
    "z_mom_12_1": 0.35,   # 12-month return skipping the last month
    "z_rs_3m":    0.25,   # relative strength vs benchmark, last 3 months
    "z_rs_slope": 0.15,   # is the RS line still rising
    "z_trend":    0.15,   # distance above the 200-day average
    "z_flow":     0.10,   # volume-weighted accumulation proxy
}

# Individual holdings. Charted but deliberately kept OUT of the scan ranking:
# single stocks are far more volatile than sector funds and would dominate the
# cross-sectional z-scores. Edit this list to match your actual positions.
PORTFOLIO = {
    "TSLA": ("Tesla",                    "my portfolio"),
    "PLTR": ("Palantir",                 "my portfolio"),
    "GOOG": ("Alphabet",                 "my portfolio"),
    "AMD":  ("AMD",                      "my portfolio"),
    "AMZN": ("Amazon",                   "my portfolio"),
    "BABA": ("Alibaba",                  "my portfolio"),
    "AVGO": ("Broadcom",                 "my portfolio"),
    "MRVL": ("Marvell",                  "my portfolio"),
    "SPCX": ("SPCX",                     "my portfolio"),
    "GC=F": ("Gold (spot)",              "my portfolio"),
}
# Tickers shown by the "My portfolio" button — includes ETFs already in
# UNIVERSE (SMH, COPX) so they are not duplicated above.
PORTFOLIO_VIEW = list(PORTFOLIO) + ["SMH", "COPX", "BTC-USD", "SOL-USD"]

# The classic sector chessboard: each GICS sector, with the narrower
# industry ETFs that sit inside it. First entry is the headline fund.
SECTOR_MAP = {
    "Technology":     ["XLK", "SMH", "IGV"],
    "Health Care":    ["XLV", "XBI", "IHI"],
    "Financials":     ["XLF", "KRE"],
    "Energy":         ["XLE", "XOP", "URA", "ICLN"],
    "Materials":      ["XLB", "XME", "COPX", "GDX"],
    "Industrials":    ["XLI", "ITA", "IYT", "JETS", "PAVE"],
    "Cons. Discret.": ["XLY"],
    "Cons. Staples":  ["XLP"],
    "Utilities":      ["XLU"],
    "Real Estate":    ["XLRE"],
    "Communications": ["XLC"],
}

ALL = {**UNIVERSE, **PORTFOLIO}

CRYPTO = {"BTC-USD", "SOL-USD"}

MIN_HOLD_DAYS = 183       # Circular 36 criterion 1
TURNOVER_LIMIT = 5.0      # Circular 36 criterion 2


# --------------------------------------------------------------------------
# DATA
# --------------------------------------------------------------------------

def _cache_path(ticker: str) -> str:
    return os.path.join(CACHE_DIR, f"{ticker.replace('=','_')}.csv")


def fetch_prices(tickers: list[str], demo: bool = False) -> dict[str, pd.DataFrame]:
    """Return {ticker: DataFrame[close, volume]} with a one-day-old disk cache."""
    if demo:
        return _synthetic_prices(tickers)

    os.makedirs(CACHE_DIR, exist_ok=True)
    import yfinance as yf

    out, to_fetch = {}, []
    today = dt.date.today()
    for t in tickers:
        p = _cache_path(t)
        if os.path.exists(p):
            age = today - dt.date.fromtimestamp(os.path.getmtime(p))
            if age.days < 1:
                df = pd.read_csv(p, index_col=0, parse_dates=True)
                if len(df) > 300:
                    out[t] = df
                    continue
        to_fetch.append(t)

    if to_fetch:
        print(f"downloading {len(to_fetch)} series one at a time "
              f"(slower, but cannot misalign)...", file=sys.stderr)
        start = today - dt.timedelta(days=365 * LOOKBACK_YEARS + 40)
        for t in to_fetch:
            try:
                # Deliberately NOT yf.download() with group_by/threads: bulk
                # multi-ticker frames can silently pair one ticker's prices
                # with another's when histories differ in length.
                h = yf.Ticker(t).history(start=start, auto_adjust=True,
                                         interval="1d")
                if h is None or h.empty:
                    print(f"  skip {t}: empty response", file=sys.stderr)
                    continue
                h.index = pd.to_datetime(h.index).tz_localize(None)
                df = pd.DataFrame({
                    "close": h["Close"].astype(float),
                    "volume": h["Volume"].astype(float).fillna(0),
                }).dropna(subset=["close"])
                df = df[df["close"] > 0]
                if len(df) < 300:
                    print(f"  skip {t}: only {len(df)} rows", file=sys.stderr)
                    continue
                df.to_csv(_cache_path(t))
                out[t] = df
                time.sleep(0.15)
            except Exception as e:  # noqa: BLE001
                print(f"  skip {t}: {e}", file=sys.stderr)
    return out


def _synthetic_prices(tickers: list[str], n: int = 800) -> dict[str, pd.DataFrame]:
    """Deterministic fake data so the logic can be tested without a network."""
    rng = np.random.default_rng(7)
    idx = pd.bdate_range(end=dt.date.today(), periods=n)
    n = len(idx)
    out = {}
    for i, t in enumerate(tickers):
        drift = rng.normal(0.0004, 0.0004)
        vol = rng.uniform(0.008, 0.022)
        shocks = rng.normal(drift, vol, n)
        # give a few names a late-stage acceleration so ranking has structure
        if i % 7 == 0:
            shocks[-120:] += 0.0016
        close = 100 * np.exp(np.cumsum(shocks))
        volume = rng.lognormal(15, 0.35, n) * (1 + 0.4 * (shocks > 0))
        out[t] = pd.DataFrame({"close": close, "volume": volume}, index=idx)
    return out


# --------------------------------------------------------------------------
# METRICS
# --------------------------------------------------------------------------

def _ret(s: pd.Series, months: int) -> float:
    """Return over `months` CALENDAR months, anchored by date, not row count.

    The earlier positional version assumed 252 rows == 1 year, which silently
    reported multi-year returns for any ticker whose feed had gaps.
    """
    if s.empty:
        return np.nan
    end = s.index[-1]
    target = end - pd.DateOffset(months=months)
    if s.index[0] > target - pd.Timedelta(days=10):
        return np.nan
    prior = s.loc[:target]
    if prior.empty:
        return np.nan
    # reject if the nearest observation is more than 10 days off the target
    if abs((prior.index[-1] - target).days) > 10:
        return np.nan
    return float(s.iloc[-1] / prior.iloc[-1] - 1)


def data_quality(px: pd.Series) -> dict:
    """Flag sparse or stale series before they poison the ranking."""
    span_days = (px.index[-1] - px.index[0]).days
    expected = span_days * (252 / 365.25)
    density = len(px) / expected if expected else 0.0
    gaps = px.index.to_series().diff().dt.days.dropna()
    staleness = (dt.date.today() - px.index[-1].date()).days
    return {
        "rows": len(px),
        "first": px.index[0].date(),
        "last": px.index[-1].date(),
        "density": density,          # 1.0 == a clean daily feed
        "max_gap": int(gaps.max()) if len(gaps) else 0,
        "stale_days": staleness,
        # staleness is reported but not part of "ok": the rank-drift pass
        # deliberately evaluates a truncated series ending 4 weeks ago.
        "ok": density > 0.90 and (gaps.max() if len(gaps) else 0) <= 12,
    }


def _zscore(s: pd.Series) -> pd.Series:
    sd = s.std(ddof=0)
    return (s - s.mean()) / sd if sd and not np.isnan(sd) else s * 0.0


def compute_metrics(prices: dict[str, pd.DataFrame],
                    bench: pd.Series,
                    fx: pd.Series | None) -> pd.DataFrame:
    """One row per instrument, USD and CHF views side by side."""
    rows = []
    for t, df in prices.items():
        if t not in UNIVERSE:
            continue
        label, bucket = UNIVERSE[t]
        px = df["close"].dropna()
        vol = df["volume"].reindex(px.index).fillna(0)
        if len(px) < 300:
            continue
        q = data_quality(px)
        r12_check = _ret(px, 12)
        lo, hi = (-0.95, 12.0) if t in CRYPTO else (-0.65, 1.50)
        if not pd.isna(r12_check) and not (lo < r12_check < hi):
            print(f"  [data] {t:<8} 12m return {r12_check*100:+.0f}% is outside "
                  f"the plausible band -> EXCLUDED, verify the feed",
                  file=sys.stderr)
            continue
        if not q["ok"]:
            print(f"  [data] {t:<8} density {q['density']:.2f} "
                  f"max gap {q['max_gap']}d last {q['last']} -> EXCLUDED",
                  file=sys.stderr)
            continue

        # CHF-translated price series (ETF is USD-denominated)
        if fx is not None:
            f = fx.reindex(px.index).ffill()
            px_chf = (px * f).dropna()
        else:
            px_chf = px

        b = bench.reindex(px.index).ffill()
        rs = px / b                       # relative strength line
        rs_ma = rs.rolling(50).mean()
        ma200 = px.rolling(200).mean()

        # accumulation proxy: share of the last 60 sessions' volume that
        # occurred on up-days, minus the 1-year baseline. Crude, but it is
        # the closest a retail data feed gets to "who is buying".
        chg = px.pct_change()
        up = (chg > 0).astype(float)
        recent = float((vol.tail(60) * up.tail(60)).sum() / max(vol.tail(60).sum(), 1))
        base = float((vol.tail(252) * up.tail(252)).sum() / max(vol.tail(252).sum(), 1))

        # base-and-breakout ("heartbeat") detector: a tight 6-month range,
        # then price pushing the top of it on above-average volume.
        hi, lo = px.tail(126).max(), px.tail(126).min()
        range_width = (hi / lo) - 1
        near_high = float(px.iloc[-1] / hi)
        vol_expansion = float(vol.tail(10).mean() / max(vol.tail(126).mean(), 1))
        breakout = (range_width < 0.22) and (near_high > 0.985) and (vol_expansion > 1.15)

        realised_vol = float(chg.tail(126).std() * np.sqrt(252))
        m12_1 = _ret(px, 12) - _ret(px, 1)

        rows.append({
            "ticker": t,
            "name": label,
            "bucket": bucket,
            "ret_1m": _ret(px, 1),
            "ret_3m": _ret(px, 3),
            "ret_6m": _ret(px, 6),
            "ret_12m": _ret(px, 12),
            "mom_12_1": m12_1,
            "ret_12m_chf": _ret(px_chf, 12),
            "ret_3m_chf": _ret(px_chf, 3),
            "rs_3m": _ret(rs, 3),
            "rs_slope": float(rs.iloc[-1] / rs_ma.iloc[-1] - 1) if not np.isnan(rs_ma.iloc[-1]) else np.nan,
            "vs_200dma": float(px.iloc[-1] / ma200.iloc[-1] - 1) if not np.isnan(ma200.iloc[-1]) else np.nan,
            "flow_proxy": recent - base,
            "vol_ann": realised_vol,
            "breakout": breakout,
            "range_6m": range_width,
        })

    df = pd.DataFrame(rows).set_index("ticker")
    if df.empty:
        return df

    df["z_mom_12_1"] = _zscore(df["mom_12_1"])
    df["z_rs_3m"] = _zscore(df["rs_3m"])
    df["z_rs_slope"] = _zscore(df["rs_slope"])
    df["z_trend"] = _zscore(df["vs_200dma"])
    df["z_flow"] = _zscore(df["flow_proxy"])
    df["score"] = sum(df[c] * w for c, w in WEIGHTS.items())

    # risk-adjusted variant: same signal, divided by realised volatility.
    # Prevents the ranking from just being "whatever moved most".
    df["score_riskadj"] = df["score"] / df["vol_ann"].clip(lower=0.05)

    df["rank"] = df["score"].rank(ascending=False).astype(int)
    df = df.sort_values("score", ascending=False)
    return df


def rank_drift(prices: dict[str, pd.DataFrame], bench: pd.Series,
               fx: pd.Series | None, weeks: int = 4) -> pd.Series:
    """Rank today vs rank `weeks` ago. Positive = capital rotating in."""
    lag = weeks * 5
    lagged = {t: df.iloc[:-lag] for t, df in prices.items() if len(df) > lag + 300}
    if not lagged:
        return pd.Series(dtype=float)
    old = compute_metrics(lagged, bench.iloc[:-lag],
                          fx.iloc[:-lag] if fx is not None else None)
    return old["rank"] if not old.empty else pd.Series(dtype=float)


# --------------------------------------------------------------------------
# OUTPUT
# --------------------------------------------------------------------------

def pct(x: float) -> str:
    return "  n/a " if pd.isna(x) else f"{x*100:+6.1f}%"


def print_scan(df: pd.DataFrame, top: int) -> None:
    cols = ["name", "bucket", "score", "ret_3m", "ret_12m", "ret_12m_chf",
            "rs_3m", "vs_200dma", "flow_proxy", "d_rank", "breakout"]
    head = f"{'':<9}{'sector / theme':<23}{'score':>7}{'3m':>8}{'12m':>8}{'12m CHF':>9}{'RS 3m':>8}{'>200d':>8}{'flow':>7}{'Δrank':>7}  base?"
    for title, sub in (("LEADERS — capital is here", df.head(top)),
                       ("LAGGARDS — capital is leaving", df.tail(min(top, 8)).iloc[::-1])):
        print(f"\n{title}\n{head}\n{'-'*len(head)}")
        for t, r in sub.iterrows():
            d = r.get("d_rank", np.nan)
            dstr = "     ." if pd.isna(d) else f"{int(d):+6d}"
            print(f"{t:<9}{r['name'][:22]:<23}{r['score']:>7.2f}"
                  f"{pct(r['ret_3m']):>8}{pct(r['ret_12m']):>8}{pct(r['ret_12m_chf']):>9}"
                  f"{pct(r['rs_3m']):>8}{pct(r['vs_200dma']):>8}"
                  f"{r['flow_proxy']*100:>6.1f}{dstr:>7}   {'YES' if r['breakout'] else '-'}")
    _ = cols


def print_buckets(df: pd.DataFrame) -> None:
    b = df.groupby("bucket")["score"].agg(["mean", "count"]).sort_values("mean", ascending=False)
    print("\nBUCKET AVERAGES — check you are not buying the same trade three times")
    print(f"{'bucket':<18}{'avg score':>10}{'n':>4}")
    print("-" * 32)
    for k, r in b.iterrows():
        print(f"{k:<18}{r['mean']:>10.2f}{int(r['count']):>4}")


# --------------------------------------------------------------------------
# PORTFOLIO GUARDRAILS
# --------------------------------------------------------------------------

def check_portfolio(scan: pd.DataFrame | None) -> None:
    if not os.path.exists("holdings.csv"):
        pd.DataFrame([{"ticker": "GLD", "shares": 100,
                       "cost_basis_chf": 20000, "buy_date": "2026-03-01"}]).to_csv(
            "holdings.csv", index=False)
        print("created a template holdings.csv — fill it in and rerun")
        return

    h = pd.read_csv("holdings.csv", parse_dates=["buy_date"])
    today = pd.Timestamp(dt.date.today())
    h["days_held"] = (today - h["buy_date"]).dt.days
    h["sellable"] = h["days_held"] >= MIN_HOLD_DAYS
    h["free_on"] = (h["buy_date"] + pd.Timedelta(days=MIN_HOLD_DAYS)).dt.date
    h["bucket"] = h["ticker"].map(lambda t: UNIVERSE.get(t, ("", "unmapped"))[1])
    total = h["cost_basis_chf"].sum()
    h["weight"] = h["cost_basis_chf"] / total

    print(f"\nPORTFOLIO — CHF {total:,.0f} at cost")
    print(f"{'ticker':<8}{'weight':>8}{'held':>7}{'sellable':>10}  free from")
    print("-" * 48)
    for _, r in h.sort_values("weight", ascending=False).iterrows():
        print(f"{r['ticker']:<8}{r['weight']*100:>7.1f}%{r['days_held']:>7}"
              f"{'yes' if r['sellable'] else 'NO':>10}  {r['free_on']}")

    print("\nCONCENTRATION BY BUCKET")
    for k, v in h.groupby("bucket")["weight"].sum().sort_values(ascending=False).items():
        flag = "  <-- over 35%" if v > 0.35 else ""
        print(f"  {k:<18}{v*100:>6.1f}%{flag}")

    # Circular 36 criterion 2: gross transaction volume vs opening portfolio
    if os.path.exists("trades.csv"):
        tr = pd.read_csv("trades.csv", parse_dates=["date"])
        ytd = tr[tr["date"].dt.year == today.year]
        gross = ytd["amount_chf"].abs().sum()
        ratio = gross / total if total else 0
        print(f"\nKS36 TURNOVER: CHF {gross:,.0f} traded YTD = {ratio:.2f}x portfolio "
              f"(limit {TURNOVER_LIMIT:.0f}x)")
        if ratio > TURNOVER_LIMIT * 0.6:
            print("  warning: approaching the professional-trader threshold")
    else:
        print("\n(no trades.csv — create one with columns date,ticker,side,amount_chf "
              "to track the KS36 5x turnover limit)")

    if scan is not None and not scan.empty:
        print("\nYOUR HOLDINGS IN THE CURRENT RANKING")
        for t in h["ticker"]:
            if t in scan.index:
                r = scan.loc[t]
                verdict = "leading" if r["rank"] <= 10 else ("lagging" if r["rank"] > len(scan) - 10 else "middle")
                print(f"  {t:<9}rank {int(r['rank']):>2}/{len(scan)}  score {r['score']:>5.2f}  {verdict}")
            else:
                print(f"  {t:<9}not in universe")


# --------------------------------------------------------------------------
# MAIN
# --------------------------------------------------------------------------


def check_one(ticker: str) -> None:
    """Fetch one ticker two ways and compare, to isolate feed problems."""
    import yfinance as yf
    print(f"--- {ticker}: single-ticker fetch (the method now used) ---")
    h = yf.Ticker(ticker).history(period="2y", auto_adjust=True, interval="1d")
    h.index = pd.to_datetime(h.index).tz_localize(None)
    px = h["Close"].dropna()
    print(px.tail(3).to_string())
    r = _ret(px, 12)
    target = px.index[-1] - pd.DateOffset(months=12)
    prior = px.loc[:target]
    print(f"price now      : {px.iloc[-1]:.2f}")
    if not prior.empty:
        print(f"price 12m ago  : {prior.iloc[-1]:.2f}  (on {prior.index[-1].date()})")
    print(f"12m return     : {pct(r)}")

    print(f"\n--- {ticker}: unadjusted close, for comparison ---")
    h2 = yf.Ticker(ticker).history(period="2y", auto_adjust=False, interval="1d")
    h2.index = pd.to_datetime(h2.index).tz_localize(None)
    px2 = h2["Close"].dropna()
    prior2 = px2.loc[:px2.index[-1] - pd.DateOffset(months=12)]
    print(f"price now      : {px2.iloc[-1]:.2f}")
    if not prior2.empty:
        print(f"price 12m ago  : {prior2.iloc[-1]:.2f}")
        print(f"12m return     : {pct(px2.iloc[-1]/prior2.iloc[-1]-1)}")

    print("\nCompare 'price now' against the live quote on your broker or "
          "finance.yahoo.com. If it matches but the 12m-ago price does not, "
          "the historical series is bad. If neither matches, the feed is wrong.")


# --------------------------------------------------------------------------
# HISTORICAL TRAILS
# --------------------------------------------------------------------------

def _rrg_series(px: pd.Series, bench: pd.Series) -> pd.DataFrame:
    """RRG axes: relative-strength ratio vs its own trend, and its momentum.

    Both axes are relative, so market-wide moves cancel out.
    """
    b = bench.reindex(px.index).ffill()
    ratio = (px / b).dropna()
    if len(ratio) < 200:
        return pd.DataFrame()
    norm = ratio / ratio.rolling(126).mean() * 100
    x = norm.rolling(5).mean() - 100
    y = ((norm / norm.shift(21) - 1) * 100).rolling(10).mean()
    return pd.DataFrame({"x": x, "y": y}).dropna()


def _classic_series(px: pd.Series, bench: pd.Series, smooth: int = 5) -> pd.DataFrame:
    """Original axes: 3-month relative strength vs benchmark, and percent
    above the 200-day average.

    Note the y-axis is absolute, so it carries market beta — a broad rally
    lifts every dot at once. Kept because it is directly readable.
    """
    b = bench.reindex(px.index).ffill()
    rs = (px / b).dropna()
    if len(rs) < 260:
        return pd.DataFrame()
    x = (rs / rs.shift(63) - 1) * 100
    ma200 = px.rolling(200).mean()
    y = (px / ma200 - 1) * 100
    df = pd.DataFrame({"x": x, "y": y}).dropna()
    if smooth > 1:
        df = df.rolling(smooth).mean().dropna()
    return df


def build_trails(prices: dict, bench: pd.Series, weeks: int,
                 axes: str = "classic", smooth: int = 5,
                 hold: list | None = None,
                 buckets: dict | None = None) -> dict:
    """Daily frames over the window, so playback glides instead of jumping."""
    end = min(s.index[-1] for s in prices.values())
    start = end - pd.Timedelta(weeks=weeks)
    series, all_idx = [], None
    for t, df in sorted(prices.items()):
        px = df["close"].dropna()
        rr = (_classic_series(px, bench, smooth) if axes == "classic"
              else _rrg_series(px, bench))
        if rr.empty:
            continue
        win = rr.loc[start:end]
        if len(win) < 20:
            continue
        all_idx = win.index if all_idx is None else all_idx.union(win.index)
        b = (buckets or {}).get(t, ALL[t][1])
        series.append({"t": t, "n": ALL[t][0], "b": b, "_d": win})
    if not series:
        return {"dates": [], "series": []}
    out = []
    for s in series:
        w = s.pop("_d").reindex(all_idx).ffill()
        s["f"] = [None if pd.isna(a) else [round(float(a), 2), round(float(c), 2)]
                  for a, c in zip(w.x, w.y)]
        out.append(s)
    lab = {"classic": ["3-month relative strength vs global equity (%)",
                       "percent above 200-day average"],
           "rrg": ["relative strength vs its own 6-month trend",
                   "momentum of relative strength"]}[axes]
    return {"dates": [str(d.date()) for d in all_idx], "series": out,
            "axes": axes, "labels": lab, "hold": hold or [],
            "port": [t for t in PORTFOLIO_VIEW if any(s["t"] == t for s in out)]}


TRAILS_HTML = """<!DOCTYPE html><html><head><meta charset="utf-8">
<title>Sector rotation trails</title><style>
body{font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',sans-serif;
margin:0;padding:24px;background:#fbfbfa;color:#1a1a19}
h1{font-size:18px;font-weight:500;margin:0 0 4px}
p.sub{font-size:13px;color:#6b6a66;margin:0 0 14px;max-width:780px;line-height:1.6}
#wrap{max-width:1040px}
.ctl{display:flex;align-items:center;gap:10px;margin:10px 0;flex-wrap:wrap}
button{font-size:13px;padding:5px 12px;border:1px solid #c3c2b7;background:#fff;
border-radius:6px;cursor:pointer}
button:hover{background:#f1efe8}
button.on{background:#1a1a19;color:#fff;border-color:#1a1a19}
input[type=range]{flex:1;min-width:200px}
#date{font-variant-numeric:tabular-nums;font-size:13px;min-width:92px;color:#52514e}
.legend{font-size:12px;color:#6b6a66;margin-top:10px;line-height:1.7}
.sw{display:inline-block;width:9px;height:9px;border-radius:2px;margin-right:4px}
#tip{position:fixed;pointer-events:none;background:#1a1a19;color:#fff;font-size:12px;
padding:5px 8px;border-radius:4px;opacity:0;transition:opacity .1s;white-space:nowrap}
select{font-size:13px;padding:5px 8px;border:1px solid #c3c2b7;border-radius:6px;background:#fff}
label.sp{font-size:12px;color:#52514e}
</style></head><body><div id="wrap">
<h1>Sector rotation trails</h1>
<p class="sub" id="sub"></p>
<div class="ctl">
<button id="play">Play</button>
<input type="range" id="slider" min="0" value="0" step="1">
<span id="date"></span>
</div>
<div class="ctl">
<label class="sp">show <select id="bucket"><option value="">every category</option></select></label>
<label class="sp">isolate <select id="focus"><option value="">nothing</option></select></label>
<input id="search" placeholder="type a ticker&hellip;" autocomplete="off"
 style="font-size:13px;padding:5px 9px;border:1px solid #c3c2b7;border-radius:6px;width:130px;flex:none">
<span id="msg" style="font-size:12px;color:#b4483a"></span>
<label class="sp">speed <input type="range" id="speed" min="1" max="24" value="8" style="width:90px;min-width:0;flex:none"></label>
<label class="sp">tail <input type="range" id="tail" min="5" max="120" value="45" style="width:90px;min-width:0;flex:none"></label>
<label class="sp">zoom <input type="range" id="zoom" min="20" max="100" value="90" style="width:90px;min-width:0;flex:none"></label>
<span id="zlab" style="font-size:12px;color:#898781"></span>
<button id="preset">My portfolio</button>
<button id="onlyhold">Only my holdings</button>
<button id="flash" class="on">Flash entries</button>
<button id="labels" class="on">Labels</button>
<button id="reset">Reset</button>
</div>
<svg id="plot" viewBox="0 0 960 615" style="width:100%;background:#fff;border:1px solid #e1e0d9;border-radius:8px"></svg>
<div class="legend">
<span class="sw" style="background:#1baf7a"></span>leading &nbsp;
<span class="sw" style="background:#eb6834"></span>rolling over &nbsp;
<span class="sw" style="background:#2a78d6"></span>early recovery &nbsp;
<span class="sw" style="background:#888780"></span>lagging<br>Faded, shrunken dots sit off-scale at the plot edge &mdash; widen the zoom to see them. A green pulse marks a dot crossing into the leading quadrant. Crossings are frequent and most do not persist &mdash; treat them as events to look at, not signals.
</div></div><div id="tip"></div>
<script>
const DATA = __DATA__;
document.getElementById('sub').textContent =
 'Horizontal: '+DATA.labels[0]+'. Vertical: '+DATA.labels[1]+
 '. Daily frames, smoothed; playback interpolates between them.';
const S=document.getElementById('plot'),NS='http://www.w3.org/2000/svg';
const X0=88,X1=930,Y0=30,Y1=530,N=DATA.dates.length;
let xs=[],ys=[];DATA.series.forEach(s=>s.f.forEach(f=>{if(f){xs.push(f[0]);ys.push(f[1]);}}));
const XS=xs.slice().sort((a,b)=>a-b), YS=ys.slice().sort((a,b)=>a-b);
const qt=(a,p)=>a[Math.max(0,Math.min(a.length-1,Math.round(p*(a.length-1))))];
let zoom=0.90,xmin,xmax,ymin,ymax;
function bounds(){
 const t=(1-zoom)/2;
 xmin=qt(XS,t);xmax=qt(XS,1-t);ymin=qt(YS,t);ymax=qt(YS,1-t);
 const px=(xmax-xmin)*.06,py=(ymax-ymin)*.08;
 xmin-=px;xmax+=px;ymin-=py;ymax+=py;
}
bounds();
const sx=v=>X0+(v-xmin)/(xmax-xmin)*(X1-X0),sy=v=>Y1-(v-ymin)/(ymax-ymin)*(Y1-Y0);
const cx_=v=>Math.max(X0+2,Math.min(X1-2,sx(v))),cy_=v=>Math.max(Y0+2,Math.min(Y1-2,sy(v)));
const off=(x,y)=>x<xmin||x>xmax||y<ymin||y>ymax;
const col=(x,y)=>x>0&&y>0?'#1baf7a':x<0&&y>0?'#eb6834':x>0?'#2a78d6':'#888780';
const FLASH=18;
DATA.series.forEach(s=>{s.cr=[];let prev=null;
 s.f.forEach((f,i)=>{if(!f)return;const lead=f[0]>0&&f[1]>0;
  if(lead&&prev===false)s.cr.push(i);prev=lead;});});
let bucket='',focus='',showLabels=true,tailLen=45,speed=8,pos=N-1,playing=false,flashOn=true;
const HOLD=new Set(DATA.hold||[]);let onlyHold=false;
const PORT=new Set(DATA.port||[]);let portOnly=false;
const bsel=document.getElementById('bucket'),fsel=document.getElementById('focus');
[...new Set(DATA.series.map(s=>s.b))].sort().forEach(b=>bsel.add(new Option(b,b)));
DATA.series.slice().sort((a,b)=>a.n.localeCompare(b.n)).forEach(s=>fsel.add(new Option(s.n,s.t)));
function el(n,a){const e=document.createElementNS(NS,n);for(const k in a)e.setAttribute(k,a[k]);return e;}
function txt(a,t){const e=el('text',a);e.textContent=t;return e;}
const tip=document.getElementById('tip');
function at(s,p){
 const i=Math.max(0,Math.min(N-1,Math.floor(p))),j=Math.min(N-1,i+1),fr=p-i;
 const a=s.f[i],b=s.f[j];
 if(!a)return b||null; if(!b)return a;
 return [a[0]+(b[0]-a[0])*fr, a[1]+(b[1]-a[1])*fr];
}
function draw(){
 const p=pos, i=Math.max(0,Math.min(N-1,Math.round(p)));
 S.innerHTML='';
 const gx=sx(0),gy=sy(0);
 S.appendChild(el('line',{x1:X0,y1:gy,x2:X1,y2:gy,stroke:'#b4b2a9','stroke-width':1}));
 S.appendChild(el('line',{x1:gx,y1:Y0,x2:gx,y2:Y1,stroke:'#b4b2a9','stroke-width':1}));
 S.appendChild(txt({x:gx+8,y:Y0+16,'font-size':12,fill:'#a3a29b'},'leading'));
 S.appendChild(txt({x:gx-8,y:Y0+16,'font-size':12,fill:'#a3a29b','text-anchor':'end'},'rolling over'));
 S.appendChild(txt({x:gx+8,y:Y1-8,'font-size':12,fill:'#a3a29b'},'early recovery'));
 S.appendChild(txt({x:gx-8,y:Y1-8,'font-size':12,fill:'#a3a29b','text-anchor':'end'},'lagging'));
 const step=v=>{const r=(v[1]-v[0])/6;const m=Math.pow(10,Math.floor(Math.log10(r)));
  return [1,2,2.5,5,10].map(k=>k*m).find(k=>k>=r)||m*10;};
 const xs2=step([xmin,xmax]),ys2=step([ymin,ymax]);
 for(let v=Math.ceil(xmin/xs2)*xs2;v<=xmax;v+=xs2){
  S.appendChild(el('line',{x1:sx(v),y1:Y0,x2:sx(v),y2:Y1,stroke:'#eeece5','stroke-width':.5}));
  S.appendChild(txt({x:sx(v),y:Y1+20,'font-size':11,fill:'#898781','text-anchor':'middle'},
   (v>0?'+':'')+Math.round(v)+'%'));}
 for(let v=Math.ceil(ymin/ys2)*ys2;v<=ymax;v+=ys2){
  S.appendChild(el('line',{x1:X0,y1:sy(v),x2:X1,y2:sy(v),stroke:'#eeece5','stroke-width':.5}));
  S.appendChild(txt({x:X0-9,y:sy(v)+4,'font-size':11,fill:'#898781','text-anchor':'end'},
   (v>0?'+':'')+Math.round(v)+'%'));}
 S.appendChild(txt({x:(X0+X1)/2,y:Y1+48,'font-size':13,fill:'#52514e','text-anchor':'middle'},
  DATA.labels[0]));
 S.appendChild(txt({x:18,y:(Y0+Y1)/2,'font-size':13,fill:'#52514e','text-anchor':'middle',
  transform:'rotate(-90 18 '+((Y0+Y1)/2)+')'},DATA.labels[1]));
 const shown=DATA.series.filter(s=>(!bucket||s.b===bucket)
  &&(!onlyHold||HOLD.has(s.t))&&(!portOnly||PORT.has(s.t)));
 const auto=showLabels;
 shown.forEach(s=>{
  const cur=at(s,p); if(!cur)return;
  const isF=focus&&s.t===focus, held=HOLD.has(s.t);
  const dim=focus&&!isF;
  const back=!portOnly&&HOLD.size&&!held&&!isF;
  const c=col(cur[0],cur[1]);
  if(!dim&&tailLen>2){
   const pts=[];
   for(let k=Math.max(0,i-tailLen);k<=i;k++){if(s.f[k])pts.push(cx_(s.f[k][0]).toFixed(1)+','+cy_(s.f[k][1]).toFixed(1));}
   pts.push(cx_(cur[0]).toFixed(1)+','+cy_(cur[1]).toFixed(1));
   if(pts.length>1)S.appendChild(el('polyline',{points:pts.join(' '),fill:'none',
    stroke:c,'stroke-width':isF?2.4:(held?1.9:1.0),opacity:isF?.85:(held?.6:(back?.14:.22)),
    'stroke-linejoin':'round','stroke-linecap':'round'}));
  }
  let cr=-1;for(const k of s.cr){if(k<=p)cr=k;else break;}
  const age=cr>=0?p-cr:1e9, flash=flashOn&&age<FLASH&&!dim;
  const r=isF?8:(held?7:4.5);
  if(flash){
   const fr=age/FLASH;
   S.appendChild(el('circle',{cx:cx_(cur[0]),cy:cy_(cur[1]),r:r+3+fr*22,
    fill:'none',stroke:'#1baf7a','stroke-width':2.2,opacity:(1-fr)*.75}));
   S.appendChild(el('circle',{cx:cx_(cur[0]),cy:cy_(cur[1]),r:r+3+fr*11,
    fill:'none',stroke:'#1baf7a','stroke-width':1.4,opacity:(1-fr)*.45}));
   const seg=[];
   for(let k=cr;k<=i;k++){if(s.f[k])seg.push(cx_(s.f[k][0]).toFixed(1)+','+cy_(s.f[k][1]).toFixed(1));}
   seg.push(cx_(cur[0]).toFixed(1)+','+cy_(cur[1]).toFixed(1));
   if(seg.length>1)S.appendChild(el('polyline',{points:seg.join(' '),fill:'none',
    stroke:'#1baf7a','stroke-width':3,opacity:(1-fr)*.85,
    'stroke-linejoin':'round','stroke-linecap':'round'}));
  }
  if(held&&!dim)S.appendChild(el('circle',{cx:cx_(cur[0]),cy:cy_(cur[1]),r:r+3.5,
   fill:'none',stroke:c,'stroke-width':1.4,opacity:.55}));
  const isOff=off(cur[0],cur[1]);
  const d=el('circle',{cx:cx_(cur[0]),cy:cy_(cur[1]),r:flash?r+1.5:(isOff?r*.7:r),fill:c,
   opacity:dim?.12:(flash?1:(isOff?.3:(back?.5:1)))});
  d.style.cursor='pointer';
  d.onmousemove=e=>{tip.style.opacity=1;tip.style.left=(e.clientX+12)+'px';
   tip.style.top=(e.clientY-8)+'px';
   tip.textContent=s.n+'   RS '+cur[0].toFixed(1)+'%   trend '+cur[1].toFixed(1)+'%';};
  d.onmouseleave=()=>tip.style.opacity=0;
  d.onclick=()=>{focus=(focus===s.t)?'':s.t;fsel.value=focus;draw();};
  S.appendChild(d);
  if((auto||isF||held||flash)&&!dim&&(!isOff||held||isF))S.appendChild(txt({x:cx_(cur[0])+r+7,y:cy_(cur[1])+4,
   'font-size':(isF||held)?12:10,
   fill:(isF||held)?'#1a1a19':'#8a8981','fill-opacity':back?.85:1},
   (isF||held)?s.n:s.t));
 });
 document.getElementById('date').textContent=DATA.dates[i];
 document.getElementById('zlab').textContent='x '+xmin.toFixed(0)+'% to '+xmax.toFixed(0)+'%';
 sl.value=i;
}
const sl=document.getElementById('slider');sl.max=N-1;sl.value=N-1;
sl.oninput=()=>{pos=+sl.value;draw();};
bsel.onchange=e=>{bucket=e.target.value;draw();};
fsel.onchange=e=>{focus=e.target.value;draw();};
document.getElementById('speed').oninput=e=>{speed=+e.target.value;};
document.getElementById('tail').oninput=e=>{tailLen=+e.target.value;draw();};
document.getElementById('zoom').oninput=e=>{zoom=+e.target.value/100;bounds();draw();};
const oh=document.getElementById('onlyhold');
if(!HOLD.size)oh.style.display='none';
oh.onclick=()=>{onlyHold=!onlyHold;oh.classList.toggle('on',onlyHold);draw();};
const sb=document.getElementById('search'),msg=document.getElementById('msg');
sb.addEventListener('input',()=>{
 const q=sb.value.trim().toUpperCase();msg.textContent='';
 if(!q){return;}
 const hit=DATA.series.find(s=>s.t===q)||
   DATA.series.find(s=>s.t.startsWith(q))||
   DATA.series.find(s=>s.n.toUpperCase().startsWith(q));
 if(hit){focus=hit.t;fsel.value=hit.t;portOnly=false;pr.classList.remove('on');
  bucket='';bsel.value='';draw();}
});
sb.addEventListener('keydown',e=>{
 if(e.key!=='Enter')return;
 const q=sb.value.trim().toUpperCase();
 if(q&&!DATA.series.some(s=>s.t===q||s.t.startsWith(q)||s.n.toUpperCase().startsWith(q)))
  msg.textContent='not in this chart — regenerate with  --add '+q;
 if(e.key==='Enter'&&!q){focus='';fsel.value='';draw();}
});
const pr=document.getElementById('preset');
if(!PORT.size)pr.style.display='none';
pr.onclick=()=>{portOnly=!portOnly;pr.classList.toggle('on',portOnly);
 if(portOnly){bucket='';bsel.value='';onlyHold=false;oh.classList.remove('on');}draw();};
const fb2=document.getElementById('flash');
fb2.onclick=()=>{flashOn=!flashOn;fb2.classList.toggle('on',flashOn);draw();};
const lb=document.getElementById('labels');
lb.onclick=()=>{showLabels=!showLabels;lb.classList.toggle('on',showLabels);draw();};
document.getElementById('reset').onclick=()=>{bucket='';focus='';onlyHold=false;portOnly=false;
 sb.value='';msg.textContent='';zoom=.90;document.getElementById('zoom').value=90;bounds();
 oh.classList.remove('on');pr.classList.remove('on');bsel.value='';fsel.value='';pos=N-1;draw();};
const pb=document.getElementById('play');
let last=0;
function tick(ts){
 if(!playing)return;
 if(last){pos+=(ts-last)/1000*speed;}
 last=ts;
 if(pos>=N-1){pos=N-1;playing=false;pb.textContent='Play';draw();return;}
 draw();requestAnimationFrame(tick);
}
pb.onclick=()=>{if(playing){playing=false;pb.textContent='Play';return;}
 if(pos>=N-1)pos=0; playing=true;last=0;pb.textContent='Pause';requestAnimationFrame(tick);};
draw();
</script></body></html>"""


def write_trails(prices: dict, bench: pd.Series, weeks: int,
                 axes: str = "classic", smooth: int = 5,
                 hold: list | None = None, out: str = "trails.html",
                 buckets: dict | None = None) -> None:
    import json
    data = build_trails(prices, bench, weeks, axes, smooth, hold, buckets)
    if not data["series"]:
        print("not enough history for trails")
        return
    with open(out, "w") as f:
        f.write(TRAILS_HTML.replace("__DATA__", json.dumps(data)))
    print(f"wrote {out} — {len(data['series'])} series, "
          f"{len(data['dates'])} daily frames "
          f"({data['dates'][0]} to {data['dates'][-1]}), axes: {axes}")
    print(f"open it with:  open {out}")


QUADS = ["leading", "weakening", "lagging", "improving"]


def _quad(x: float, y: float) -> str:
    if x > 0:
        return "leading" if y > 0 else "weakening"
    return "improving" if y > 0 else "lagging"


def _weekly_panel(prices: dict, bench: pd.Series) -> pd.DataFrame:
    """One row per (ticker, week): quadrant now, forward relative returns."""
    rows = []
    for t, df in prices.items():
        px = df["close"].dropna()
        rr = _rrg_series(px, bench)
        if rr.empty or len(rr) < 300:
            continue
        wk = rr.resample("W-FRI").last().dropna()
        b = bench.reindex(px.index).ffill()
        rel = (px / b).resample("W-FRI").last().dropna()
        rel = rel.reindex(wk.index).ffill()
        for h in (4, 13, 26):
            wk[f"fwd{h}"] = rel.shift(-h) / rel - 1
            wk[f"q{h}"] = None
        wk["quad"] = [_quad(a, c) for a, c in zip(wk.x, wk.y)]
        for h in (4, 13, 26):
            wk[f"q{h}"] = wk["quad"].shift(-h)
        wk["ticker"] = t
        rows.append(wk.reset_index().rename(columns={"index": "date",
                                                     wk.index.name or "index": "date"}))
    if not rows:
        return pd.DataFrame()
    out = pd.concat(rows, ignore_index=True)
    out.columns = [c if c != out.columns[0] else "date" for c in out.columns]
    return out


def _transition(panel: pd.DataFrame, h: int) -> pd.DataFrame:
    sub = panel.dropna(subset=[f"q{h}"])
    if sub.empty:
        return pd.DataFrame()
    tab = pd.crosstab(sub["quad"], sub[f"q{h}"], normalize="index") * 100
    return tab.reindex(index=QUADS, columns=QUADS).fillna(0)


def _fwd_stats(panel: pd.DataFrame, h: int) -> pd.DataFrame:
    sub = panel.dropna(subset=[f"fwd{h}"])
    g = sub.groupby("quad")[f"fwd{h}"]
    out = pd.DataFrame({
        "mean_%": (g.mean() * 100).round(2),
        "median_%": (g.median() * 100).round(2),
        "hit_%": (g.apply(lambda s: (s > 0).mean()) * 100).round(1),
        "n": g.size(),
    }).reindex(QUADS)
    out["eff_n"] = (out["n"] / h).round(0)
    return out


def _null_panel(prices: dict, bench: pd.Series, seed: int) -> pd.DataFrame:
    """Same pipeline on random walks with matched volatility and zero drift."""
    rng = np.random.default_rng(seed)
    fake = {}
    for t, df in prices.items():
        px = df["close"].dropna()
        vol = px.pct_change().std()
        steps = rng.normal(0, vol, len(px))
        fake[t] = pd.DataFrame(
            {"close": 100 * np.exp(np.cumsum(steps)), "volume": 1.0},
            index=px.index)
    bvol = bench.pct_change().std()
    fb = pd.Series(100 * np.exp(np.cumsum(rng.normal(0, bvol, len(bench)))),
                   index=bench.index)
    return _weekly_panel(fake, fb)


def run_persistence(prices: dict, bench: pd.Series, sims: int = 12) -> None:
    panel = _weekly_panel(prices, bench)
    if panel.empty:
        print("not enough history")
        return
    span = f"{panel['date'].min().date()} to {panel['date'].max().date()}"
    print(f"\n{len(panel):,} ticker-weeks across {panel.ticker.nunique()} funds, {span}")

    print("\n" + "=" * 66)
    print("1. QUADRANT PERSISTENCE — % still in the same quadrant later")
    print("=" * 66)
    print(f"{'quadrant':<13}{'4 wks':>9}{'13 wks':>9}{'26 wks':>9}"
          f"{'null 26w':>11}{'n':>9}")
    print("-" * 66)

    nulls = {q: [] for q in QUADS}
    for s in range(sims):
        np_panel = _null_panel(prices, bench, 1000 + s)
        if np_panel.empty:
            continue
        t26 = _transition(np_panel, 26)
        for q in QUADS:
            if q in t26.index:
                nulls[q].append(t26.loc[q, q])

    stay = {h: _transition(panel, h) for h in (4, 13, 26)}
    for q in QUADS:
        n = (panel["quad"] == q).sum()
        vals = [stay[h].loc[q, q] if q in stay[h].index else np.nan
                for h in (4, 13, 26)]
        nv = np.mean(nulls[q]) if nulls[q] else np.nan
        print(f"{q:<13}{vals[0]:>8.1f}%{vals[1]:>8.1f}%{vals[2]:>8.1f}%"
              f"{nv:>10.1f}%{n:>9,}")
    print("\nrandom-walk null from "
          f"{sims} simulations through the identical pipeline.")

    print("\n" + "=" * 66)
    print("2. WHERE DOTS GO NEXT — quadrant after 13 weeks (row %)")
    print("=" * 66)
    t = stay[13]
    print(f"{'from \\\\ to':<13}" + "".join(f"{c:>13}" for c in QUADS))
    print("-" * 66)
    for q in QUADS:
        if q in t.index:
            print(f"{q:<13}" + "".join(f"{t.loc[q, c]:>12.1f}%" for c in QUADS))

    print("\n" + "=" * 66)
    print("3. FORWARD RETURN vs BENCHMARK by starting quadrant")
    print("=" * 66)
    for h, lab in ((13, "13 weeks"), (26, "26 weeks — your minimum hold")):
        f = _fwd_stats(panel, h)
        print(f"\n{lab}")
        print(f"{'quadrant':<13}{'mean':>9}{'median':>9}{'hit rate':>11}"
              f"{'obs':>9}{'indep obs':>11}")
        print("-" * 62)
        for q in QUADS:
            if q in f.index and not pd.isna(f.loc[q, 'mean_%']):
                r = f.loc[q]
                print(f"{q:<13}{r['mean_%']:>8.2f}%{r['median_%']:>8.2f}%"
                      f"{r['hit_%']:>10.1f}%{int(r['n']):>9,}{int(r['eff_n']):>11,}")

    lead = _fwd_stats(panel, 26).loc["leading", "mean_%"]
    lag = _fwd_stats(panel, 26).loc["lagging", "mean_%"]
    print(f"\nleading minus lagging over 26 weeks: {lead - lag:+.2f}% "
          f"relative to benchmark, before costs.")
    print("Observations overlap heavily — trust the 'indep obs' column, "
          "not 'obs', when judging significance.")


INDEX_HTML = """<!DOCTYPE html><html><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Rotation charts</title><style>
body{font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',sans-serif;
background:#fbfbfa;color:#1a1a19;margin:0;padding:40px 24px;line-height:1.6}
.w{max-width:560px;margin:0 auto}
h1{font-size:22px;font-weight:500;margin:0 0 4px}
p.s{color:#6b6a66;font-size:14px;margin:0 0 28px}
a.card{display:block;text-decoration:none;color:inherit;border:1px solid #e1e0d9;
background:#fff;border-radius:10px;padding:16px 18px;margin-bottom:12px}
a.card:hover{border-color:#b4b2a9;background:#fdfdfc}
a.card b{font-size:16px;font-weight:600;display:block}
a.card span{font-size:13px;color:#6b6a66}
.f{margin-top:26px;font-size:12px;color:#898781;border-top:1px solid #e1e0d9;padding-top:14px}
</style></head><body><div class="w">
<h1>Rotation charts</h1>
<p class="s">Updated __STAMP__</p>
<a class="card" href="sectors.html"><b>Sectors</b>
<span>11 GICS sectors with their industry funds. Filter to one sector to see
its components.</span></a>
<a class="card" href="portfolio.html"><b>My portfolio</b>
<span>Your positions on the same axes, scaled to just those names.</span></a>
<a class="card" href="trails.html"><b>Full board</b>
<span>Sectors, countries, bonds, commodities and crypto together.</span></a>
<div class="f">Horizontal axis: 3-month relative strength versus global equity.
Vertical: percent above the 200-day average, which still carries market beta &mdash;
a broad rally lifts every dot at once. Price data from Yahoo via yfinance, which
fails silently when it breaks; cross-check anything surprising before acting on it.
Not investment advice.</div>
</div></body></html>"""


def publish(prices: dict, bench: pd.Series, weeks: int, axes: str,
            smooth: int, outdir: str = "site") -> None:
    """Regenerate all three charts plus a landing page into `outdir`."""
    import shutil
    os.makedirs(outdir, exist_ok=True)

    bmap = {t: sec for sec, ts in SECTOR_MAP.items() for t in ts}
    sect = {t: prices[t] for t in bmap if t in prices}
    heads = [ts[0] for ts in SECTOR_MAP.values() if ts[0] in prices]
    write_trails(sect, bench, weeks, axes, smooth, heads,
                 out="sectors.html", buckets=bmap)

    port = {t: prices[t] for t in PORTFOLIO_VIEW if t in prices}
    if port:
        write_trails(port, bench, weeks, axes, smooth, list(port),
                     out="portfolio.html")

    write_trails({t: d for t, d in prices.items() if t in ALL}, bench, weeks,
                 axes, smooth, [t for t in PORTFOLIO_VIEW if t in prices],
                 out="trails.html")

    stamp = dt.datetime.now().strftime("%d %B %Y")
    with open(os.path.join(outdir, "index.html"), "w") as f:
        f.write(INDEX_HTML.replace("__STAMP__", stamp))
    for name in ("sectors.html", "portfolio.html", "trails.html"):
        if os.path.exists(name):
            shutil.move(name, os.path.join(outdir, name))
    print(f"\nsite ready in {outdir}/ — index.html plus three charts")


def main() -> None:
    ap = argparse.ArgumentParser(description="sector relative-strength scanner")
    ap.add_argument("command", choices=["scan", "portfolio", "diagnose", "check", "trails", "mine", "sectors", "persistence", "publish"])
    ap.add_argument("--top", type=int, default=12)
    ap.add_argument("--ticker", default="EWL", help="for the check command")
    ap.add_argument("--weeks", type=int, default=26, help="trail length")
    ap.add_argument("--years", type=int, default=None, help="history depth")
    ap.add_argument("--sims", type=int, default=12, help="null simulations")
    ap.add_argument("--axes", choices=["classic", "rrg"], default="classic")
    ap.add_argument("--smooth", type=int, default=5, help="days of smoothing")
    ap.add_argument("--add", default="",
                    help="extra tickers to chart, e.g. --add NVDA,ASML,NESN.SW")
    ap.add_argument("--holdings", default="",
                    help="comma-separated tickers to highlight, e.g. XLK,GLD,SLV")
    ap.add_argument("--demo", action="store_true", help="synthetic data, no network")
    ap.add_argument("--no-drift", action="store_true", help="skip the 4-week rank drift pass")
    args = ap.parse_args()

    if args.command == "check":
        check_one(args.ticker)
        return

    global LOOKBACK_YEARS, CACHE_DIR
    if args.years:
        LOOKBACK_YEARS = args.years
        CACHE_DIR = f"cache_{args.years}y"
    elif args.command == "persistence":
        LOOKBACK_YEARS = 15
        CACHE_DIR = "cache_15y"

    for extra in [a.strip().upper() for a in args.add.split(",") if a.strip()]:
        if extra not in ALL:
            ALL[extra] = (extra, "added")

    tickers = list(ALL) + [BENCHMARK, FX_PAIR]
    prices = fetch_prices(tickers, demo=args.demo)
    if BENCHMARK not in prices:
        sys.exit(f"benchmark {BENCHMARK} unavailable")

    bench = prices[BENCHMARK]["close"]

    # Crypto trades every day; equities do not. Left unaligned, a 200-row
    # window would span 200 calendar days for crypto and 200 trading days
    # (about 9.5 months) for everything else, making the two incomparable.
    for t in list(prices):
        if t in CRYPTO:
            prices[t] = (prices[t].reindex(bench.index).ffill()
                         .dropna(subset=["close"]))
    fx = prices[FX_PAIR]["close"] if FX_PAIR in prices else None
    universe_prices = {t: d for t, d in prices.items() if t in UNIVERSE}
    chart_prices = {t: d for t, d in prices.items() if t in ALL}

    scan = compute_metrics(universe_prices, bench, fx)
    if scan.empty:
        sys.exit("no usable data")

    if not args.no_drift:
        old = rank_drift(universe_prices, bench, fx, weeks=4)
        scan["d_rank"] = (old.reindex(scan.index) - scan["rank"]) if not old.empty else np.nan
    else:
        scan["d_rank"] = np.nan

    if args.command == "persistence":
        run_persistence(universe_prices, bench, sims=args.sims)
        return

    if args.command == "publish":
        publish(prices, bench, args.weeks, args.axes, args.smooth)
        return

    if args.command == "sectors":
        bmap = {t: sec for sec, ts in SECTOR_MAP.items() for t in ts}
        want = [t for t in bmap if t in prices]
        missing = [t for t in bmap if t not in prices]
        if missing:
            print(f"no data for: {', '.join(missing)}", file=sys.stderr)
        heads = [ts[0] for ts in SECTOR_MAP.values() if ts[0] in prices]
        print(f"charting {len(want)} sector funds across "
              f"{len(SECTOR_MAP)} sectors")
        write_trails({t: prices[t] for t in want}, bench, args.weeks,
                     args.axes, args.smooth, heads, out="sectors.html",
                     buckets=bmap)
        return

    if args.command == "mine":
        want = [t for t in PORTFOLIO_VIEW if t in prices]
        missing = [t for t in PORTFOLIO_VIEW if t not in prices]
        if missing:
            print(f"no data returned for: {', '.join(missing)} — "
                  f"run  check --ticker <T>  to see why", file=sys.stderr)
        mine = {t: prices[t] for t in want}
        if not mine:
            sys.exit("none of the portfolio tickers returned data")
        print(f"charting {len(want)} positions: {', '.join(want)}")
        write_trails(mine, bench, args.weeks, args.axes, args.smooth,
                     want, out="portfolio.html")
        return

    if args.command == "trails":
        held = [h.strip().upper() for h in args.holdings.split(",") if h.strip()]
        unknown = [h for h in held if h not in ALL]
        if unknown:
            print(f"not in universe, ignored: {', '.join(unknown)}", file=sys.stderr)
        write_trails(chart_prices, bench, args.weeks, args.axes,
                     args.smooth, [h for h in held if h in ALL])
        return

    if args.command == "diagnose":
        print(f"{'ticker':<10}{'rows':>6}{'first':>12}{'last':>12}"
              f"{'density':>9}{'gap':>5}{'12m ago':>12}{'12m ret':>9}  ok")
        print("-" * 84)
        for t, d in sorted(universe_prices.items()):
            px = d["close"].dropna()
            q = data_quality(px)
            target = px.index[-1] - pd.DateOffset(months=12)
            prior = px.loc[:target]
            anchor = prior.index[-1].date() if not prior.empty else None
            r12 = _ret(px, 12)
            print(f"{t:<10}{q['rows']:>6}{str(q['first']):>12}{str(q['last']):>12}"
                  f"{q['density']:>9.2f}{q['max_gap']:>5}{str(anchor):>12}"
                  f"{pct(r12):>9}  {'ok' if q['ok'] and q['stale_days'] <= 7 else 'BAD'}")
        print("\ndensity 1.00 = clean daily feed. Anything under 0.90, or a 12m "
              "anchor far from one year ago, means the feed is gappy and that "
              "row's numbers cannot be trusted.")
        return

    if args.command == "scan":
        print_scan(scan, args.top)
        print_buckets(scan)
        movers = scan.dropna(subset=["d_rank"]).sort_values("d_rank", ascending=False).head(5)
        if not movers.empty:
            print("\nBIGGEST 4-WEEK RANK IMPROVEMENT — early rotation candidates")
            for t, r in movers.iterrows():
                print(f"  {t:<9}{r['name'][:26]:<28}{int(r['d_rank']):+3d} places "
                      f"-> rank {int(r['rank'])}")
        out = f"scan_{dt.date.today()}.csv"
        scan.to_csv(out)
        print(f"\nwritten to {out}")
        print("Reminder: a high score is a description of the past, not a forecast. "
              "Nothing here is investment advice.")
    else:
        check_portfolio(scan)


if __name__ == "__main__":
    main()
