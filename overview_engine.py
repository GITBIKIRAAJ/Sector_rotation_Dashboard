"""
overview_engine.py  (v3 — removed PE/FII/DII, merged price download, faster)
"""

import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings("ignore")

INDEX_TICKERS = {
    "NIFTY 50":   "^NSEI",
    "BANK NIFTY": "^NSEBANK",
    "SENSEX":     "^BSESN",
}


# ── 1. Index Snapshot ─────────────────────────────────────────────────────────

def fetch_index_snapshot() -> dict:
    result = {}
    tickers = list(INDEX_TICKERS.values())
    try:
        raw = yf.download(tickers, period="30d", auto_adjust=True,
                          progress=False, threads=True)
        if raw.empty:
            return result
        closes = raw["Close"] if isinstance(raw.columns, pd.MultiIndex) else raw[["Close"]]
        if not isinstance(raw.columns, pd.MultiIndex):
            closes.columns = tickers[:1]

        for name, tkr in INDEX_TICKERS.items():
            if tkr not in closes.columns:
                continue
            s = closes[tkr].dropna()
            if len(s) < 2:
                continue
            last  = float(s.iloc[-1])
            prev  = float(s.iloc[-2])
            chg   = last - prev
            chg_p = chg / prev * 100 if prev != 0 else 0
            ema20 = float(s.ewm(span=20, adjust=False).mean().iloc[-1])
            if last > ema20 * 1.005:
                tag, tc = "Bullish", "#16a34a"
            elif last < ema20 * 0.995:
                tag, tc = "Bearish", "#dc2626"
            else:
                tag, tc = "Range", "#ca8a04"
            result[name] = {
                "last_price": round(last, 2),
                "prev_close": round(prev, 2),
                "change_abs": round(chg, 2),
                "change_pct": round(chg_p, 2),
                "sparkline":  [round(float(x), 2) for x in s.tail(10).tolist()],
                "trend_tag":  tag,
                "tag_color":  tc,
                "ema20":      round(ema20, 2),
            }
    except Exception:
        pass
    return result


# ── 2. Shared price download (used by both breadth + movers) ──────────────────

def _fetch_shared_prices(universe_df: pd.DataFrame,
                          max_stocks: int = 200) -> pd.DataFrame:
    uni = universe_df.copy()
    if "nifty_index" in uni.columns:
        n500 = uni[uni["nifty_index"].isin(
            ["NIFTY 50","NIFTY 100","NIFTY 200","NIFTY 500"])]
        if len(n500) > 50:
            uni = n500
    uni = uni.head(max_stocks)
    tickers = uni["ticker"].dropna().tolist()
    if not tickers:
        return pd.DataFrame()
    start = (datetime.today() - timedelta(days=400)).strftime("%Y-%m-%d")
    try:
        raw = yf.download(tickers, start=start, auto_adjust=True,
                          progress=False, threads=True)
        if raw.empty:
            return pd.DataFrame()
        closes = raw["Close"] if isinstance(raw.columns, pd.MultiIndex) else raw[["Close"]]
        if not isinstance(raw.columns, pd.MultiIndex):
            closes.columns = tickers[:1]
        closes = closes.ffill()
        return closes
    except Exception:
        return pd.DataFrame()


# ── 3. Market Breadth ─────────────────────────────────────────────────────────

def compute_market_breadth(closes: pd.DataFrame) -> dict:
    if closes.empty:
        return {}
    total    = len(closes.columns)
    above    = {}
    advances = declines = unchanged = 0

    for label, w in {"9 SMA":9, "21 SMA":21, "50 SMA":50, "200 SMA":200}.items():
        if len(closes) < w:
            above[label] = {"count":0, "total":total, "pct":0.0}
            continue
        sma = closes.rolling(w, min_periods=w).mean()
        cnt = int((closes.iloc[-1] > sma.iloc[-1]).sum())
        above[label] = {
            "count": cnt, "total": total,
            "pct":   round(cnt / total * 100, 1) if total > 0 else 0.0,
        }

    rsi_count = 0
    dr = closes.pct_change().dropna()
    if len(dr) >= 14:
        gain = dr.clip(lower=0).rolling(14).mean()
        loss = (-dr.clip(upper=0)).rolling(14).mean()
        rsi_all   = (100 - 100 / (1 + gain / loss.replace(0, np.nan))).iloc[-1]
        rsi_count = int((rsi_all > 50).sum())
    above["RSI >50"] = {
        "count": rsi_count, "total": total,
        "pct":   round(rsi_count / total * 100, 1) if total > 0 else 0.0,
    }

    if len(closes) >= 2:
        dc        = closes.iloc[-1] - closes.iloc[-2]
        advances  = int((dc > 0).sum())
        declines  = int((dc < 0).sum())
        unchanged = total - advances - declines

    return {
        "advances":    advances,
        "declines":    declines,
        "unchanged":   unchanged,
        "total":       total,
        "bullish_pct": above.get("50 SMA", {}).get("pct", 0),
        "above_sma":   above,
    }


# ── 4. Top Movers (Gainers / Losers / Near 52W High) ─────────────────────────

def compute_top_movers(closes: pd.DataFrame,
                        universe_df: pd.DataFrame,
                        top_n: int = 10) -> dict:
    if closes.empty or len(closes) < 2:
        return {}

    uni  = universe_df.copy()
    if "nifty_index" in uni.columns:
        n500 = uni[uni["nifty_index"].isin(
            ["NIFTY 50","NIFTY 100","NIFTY 200","NIFTY 500"])]
        if len(n500) > 50:
            uni = n500
    uni = uni.head(len(closes.columns))
    meta = {row["ticker"]: row for _, row in uni.iterrows()}

    yest_chg = (
        (closes.iloc[-1] - closes.iloc[-2])
        / closes.iloc[-2].replace(0, np.nan) * 100
    ).dropna()

    def _enrich(series):
        rows = []
        for tkr, val in series.items():
            m = meta.get(tkr, {})
            rows.append({
                "stock":    tkr.replace(".NS", ""),
                "company":  m.get("company_name", tkr.replace(".NS", "")),
                "sector":   m.get("nse_sector", ""),
                "cap_tier": m.get("cap_tier", ""),
                "price":    round(float(closes[tkr].iloc[-1]), 2) if tkr in closes else 0,
                "chg_pct":  round(float(val), 2),
            })
        return rows

    # Near 52W High (within 3%)
    high_52w = closes.tail(252).max()
    last_p   = closes.iloc[-1]
    dist_52w = ((last_p - high_52w) / high_52w.replace(0, np.nan) * 100).dropna()
    near_52w = []
    for tkr, dist in dist_52w[dist_52w >= -3].nlargest(top_n).items():
        m = meta.get(tkr, {})
        near_52w.append({
            "stock":    tkr.replace(".NS", ""),
            "company":  m.get("company_name", tkr.replace(".NS", "")),
            "sector":   m.get("nse_sector", ""),
            "cap_tier": m.get("cap_tier", ""),
            "price":    round(float(closes[tkr].iloc[-1]), 2) if tkr in closes else 0,
            "high_52w": round(float(high_52w[tkr]), 2),
            "dist_pct": round(float(dist), 2),
        })

    return {
        "gainers":  _enrich(yest_chg.nlargest(top_n)),
        "losers":   _enrich(yest_chg.nsmallest(top_n)),
        "near_52w": near_52w,
    }


# ── 5. Sector Treemap ─────────────────────────────────────────────────────────

def build_sector_treemap_data(prices_df: pd.DataFrame, sectors) -> pd.DataFrame:
    rows = []
    for name in sectors:
        if name not in prices_df.columns:
            continue
        s = prices_df[name].dropna()
        if len(s) < 2:
            continue
        chg = (s.iloc[-1] - s.iloc[-2]) / s.iloc[-2] * 100 if s.iloc[-2] != 0 else 0
        rows.append({
            "sector":     name.replace("NIFTY ", ""),
            "change_pct": round(float(chg), 2),
        })
    df = pd.DataFrame(rows)
    if not df.empty:
        df["color"] = df["change_pct"].apply(
            lambda x: "#16a34a" if x > 0.5 else "#dc2626" if x < -0.5 else "#ca8a04"
        )
    return df.sort_values("change_pct", ascending=False) if not df.empty else df


# ── 6. Summary text ───────────────────────────────────────────────────────────

def generate_summary(snap: dict, breadth: dict) -> str:
    lines = []
    if snap:
        parts = []
        for name, d in snap.items():
            c    = d["change_pct"]
            sign = "up" if c >= 0 else "down"
            parts.append(name + " " + sign + " " + str(abs(round(c, 1))) + "%")
        lines.append("Indices: " + "  |  ".join(parts))
    if breadth:
        bp  = breadth.get("bullish_pct", 0)
        adv = breadth.get("advances", 0)
        dec = breadth.get("declines", 0)
        s   = "strong" if bp > 60 else "weak" if bp < 35 else "mixed"
        lines.append(
            "Breadth: " + s + " — "
            + str(round(bp)) + "% above 50-SMA  ("
            + str(adv) + " adv, " + str(dec) + " dec)"
        )
    return "  \n".join(lines) if lines else "Market data loading..."


# ── Master loader ─────────────────────────────────────────────────────────────

def load_overview_data(universe_df=None, prices_df=None,
                       sectors=None, progress_cb=None) -> dict:
    if progress_cb: progress_cb(0, 4, "Fetching index prices...")
    snap = fetch_index_snapshot()

    treemap = (build_sector_treemap_data(prices_df, sectors)
               if prices_df is not None and sectors else pd.DataFrame())

    breadth = {}
    movers  = {}
    if universe_df is not None:
        if progress_cb: progress_cb(1, 4, "Downloading stock prices (single batch)...")
        shared = _fetch_shared_prices(universe_df, max_stocks=200)

        if progress_cb: progress_cb(2, 4, "Computing breadth...")
        breadth = compute_market_breadth(shared)

        if progress_cb: progress_cb(3, 4, "Computing movers & 52W High...")
        movers  = compute_top_movers(shared, universe_df, top_n=10)

    return {
        "snap":         snap,
        "breadth":      breadth,
        "movers":       movers,
        "treemap":      treemap,
        "summary":      generate_summary(snap, breadth),
        "last_updated": datetime.now().strftime("%d %b %Y %H:%M IST"),
    }
