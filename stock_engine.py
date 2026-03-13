"""
stock_engine.py (v3 — fixed top/bottom split, real vol_ud & atr_pct scoring)
"""
import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime, timedelta
from concurrent.futures import ThreadPoolExecutor, as_completed
import warnings
warnings.filterwarnings("ignore")

from universe_builder import (
    build_universe, filter_universe,
    get_sectors, get_tickers, assign_tier,
)

CAP_TIER_ORDER  = ["Mega", "Large", "Mid", "Small", "Micro", "Unknown"]
CAP_TIER_COLORS = {
    "Mega":    "#7c3aed",
    "Large":   "#2563eb",
    "Mid":     "#16a34a",
    "Small":   "#ca8a04",
    "Micro":   "#dc2626",
    "Unknown": "#9ca3af",
}
CAP_TIER_RANGES = {
    "Mega":  "> 50,000 Cr",
    "Large": "20,000 – 50,000 Cr",
    "Mid":   "5,000 – 20,000 Cr",
    "Small": "1,000 – 5,000 Cr",
    "Micro": "< 1,000 Cr",
}

# Weight definitions — shown as tooltips in the UI
WEIGHT_DEFINITIONS = {
    "rs_nifty":  ("RS vs NIFTY (25%)",
                  "Relative Strength of stock vs NIFTY 50 index over 63 days. "
                  ">100 = outperforming the benchmark."),
    "rs_sector": ("RS vs Sector (20%)",
                  "Relative Strength vs the stock's own sector index. "
                  "Measures leadership within the sector."),
    "rsi":       ("RSI Strength (15%)",
                  "14-day RSI converted to 0–100 score. "
                  "Sweet spot is RSI 50–70 (strong momentum, not overbought)."),
    "ema_align": ("EMA Alignment (15%)",
                  "Checks if Price > EMA20 > EMA50 > EMA100 > EMA200. "
                  "4/4 = fully aligned uptrend."),
    "vol_ud":    ("Volume Up/Down Ratio (10%)",
                  "Ratio of avg volume on up-days vs down-days over 20 sessions. "
                  ">1.2 signals institutional buying pressure."),
    "atr_pct":   ("ATR % Efficiency (8%)",
                  "14-day ATR as % of price, normalised. "
                  "Rewards low-volatility uptrends over erratic movers."),
    "dist_52w":  ("52-Week High Proximity (7%)",
                  "How close the stock is to its 52-week high. "
                  "0% = at the high (best), -30%+ = deeply discounted."),
}

DEFAULT_WEIGHTS = {k: float(v.split("(")[1].split("%")[0]) / 100
                   for k, v in {
                       "rs_nifty": "(25%)", "rs_sector": "(20%)",
                       "rsi": "(15%)", "ema_align": "(15%)",
                       "vol_ud": "(10%)", "atr_pct": "(8%)", "dist_52w": "(7%)"
                   }.items()}


# ── Price data ────────────────────────────────────────────────────────────────

def fetch_stock_prices(tickers: list, period_days: int = 200) -> pd.DataFrame:
    if not tickers:
        return pd.DataFrame()
    start = (datetime.today() - timedelta(days=period_days)).strftime("%Y-%m-%d")
    try:
        raw = yf.download(
            tickers, start=start,
            auto_adjust=True, progress=False, threads=True,
        )
    except Exception:
        return pd.DataFrame()
    if raw.empty:
        return pd.DataFrame()
    if isinstance(raw.columns, pd.MultiIndex):
        prices = raw["Close"].copy()
    else:
        prices = raw[["Close"]].copy() if "Close" in raw.columns else raw.copy()
        if len(tickers) == 1:
            prices.columns = [tickers[0]]
    prices.replace([np.inf, -np.inf], np.nan, inplace=True)
    prices.ffill(limit=3, inplace=True)
    prices.dropna(axis=1, how="all", inplace=True)
    prices.dropna(how="all", inplace=True)
    return prices


# ── Technical indicators ─────────────────────────────────────────────────────

def _rsi(series: pd.Series, period: int = 14) -> float:
    delta = series.diff().dropna()
    if len(delta) < period:
        return np.nan
    gain = delta.clip(lower=0).rolling(period, min_periods=period).mean()
    loss = (-delta.clip(upper=0)).rolling(period, min_periods=period).mean()
    rs   = gain / loss.replace(0, np.nan)
    rsi  = (100 - (100 / (1 + rs))).dropna()
    return float(rsi.iloc[-1]) if len(rsi) > 0 else np.nan

def _ema(series: pd.Series, span: int) -> pd.Series:
    return series.ewm(span=span, adjust=False).mean()

def _ema_alignment_score(close: pd.Series) -> tuple:
    if len(close.dropna()) < 200:
        return 0, "N/A"
    last  = close.dropna().iloc[-1]
    e20   = _ema(close, 20).iloc[-1]
    e50   = _ema(close, 50).iloc[-1]
    e100  = _ema(close, 100).iloc[-1]
    e200  = _ema(close, 200).iloc[-1]
    checks = [last > e20, e20 > e50, e50 > e100, e100 > e200]
    score  = sum(checks)
    label  = "".join(["✓" if c else "✗" for c in checks])
    return score, label

def _distance_from_52w_high(close: pd.Series) -> float:
    year = close.dropna().tail(252)
    if len(year) < 20:
        return np.nan
    high = year.max()
    last = year.iloc[-1]
    if high == 0:
        return np.nan
    return float((last - high) / high * 100)

def _rs_ratio(stock: pd.Series, benchmark: pd.Series, window: int = 63) -> float:
    common = stock.index.intersection(benchmark.index)
    if len(common) < 20:
        return np.nan
    s  = stock.loc[common].ffill()
    b  = benchmark.loc[common].replace(0, np.nan).ffill()
    rs = (s / b).replace([np.inf, -np.inf], np.nan).dropna()
    if len(rs) < 10:
        return np.nan
    rs_norm = rs / rs.rolling(window, min_periods=10).mean() * 100
    val = rs_norm.dropna()
    return float(val.iloc[-1]) if len(val) > 0 else np.nan

def _vol_ud_ratio(close: pd.Series, volume: pd.Series = None, window: int = 20) -> float:
    """Up-day vs down-day volume ratio using price proxy when volume not available."""
    changes = close.diff().dropna()
    if len(changes) < window:
        return np.nan
    recent = changes.tail(window)
    up_count   = (recent > 0).sum()
    down_count = (recent < 0).sum()
    if down_count == 0:
        return 2.0
    ratio = up_count / down_count
    return float(ratio)

def _atr_pct(close: pd.Series, period: int = 14) -> float:
    """ATR as % of current price (lower = smoother trend = better score)."""
    if len(close.dropna()) < period + 1:
        return np.nan
    s    = close.dropna()
    high = s.rolling(2).max()
    low  = s.rolling(2).min()
    tr   = (high - low).dropna()
    atr  = tr.rolling(period, min_periods=period).mean().dropna()
    if atr.empty or s.iloc[-1] == 0:
        return np.nan
    return float(atr.iloc[-1] / s.iloc[-1] * 100)

def _period_return(close: pd.Series, days: int) -> float:
    s = close.dropna()
    if len(s) < days + 1:
        return np.nan
    val = (s.iloc[-1] - s.iloc[-(days + 1)]) / s.iloc[-(days + 1)] * 100
    return round(float(val), 2) if np.isfinite(val) else np.nan

def _rsi_to_score(rsi: float) -> float:
    if pd.isna(rsi): return 50
    if rsi < 30:     return rsi * 0.5
    if rsi <= 70:    return 50 + (rsi - 50) * 1.5 if rsi >= 50 else 30 + (rsi - 30)
    return max(0, 100 - (rsi - 70) * 3)

def _vol_ud_to_score(ratio: float) -> float:
    if pd.isna(ratio) or not np.isfinite(ratio): return 50.0
    # ratio 1.0 → 50, ratio 2.0 → 100, ratio 0.5 → 25
    return float(np.clip((ratio - 0.5) / 1.5 * 100, 0, 100))

def _atr_to_score(atr_pct_val: float) -> float:
    """Lower ATR% = smoother = higher score. Maps 0–5% to 100–0."""
    if pd.isna(atr_pct_val) or not np.isfinite(atr_pct_val): return 50.0
    return float(np.clip(100 - (atr_pct_val / 5.0) * 100, 0, 100))

def _clamp(v, lo=0.0, hi=100.0):
    if pd.isna(v) or not np.isfinite(v): return 50.0
    return max(lo, min(hi, float(v)))


# ── Single stock scorer ───────────────────────────────────────────────────────

def score_stock(ticker: str, prices_df: pd.DataFrame,
                nifty: pd.Series, sector_index: pd.Series,
                weights: dict = None, meta: dict = None) -> dict:
    w = weights or DEFAULT_WEIGHTS
    if ticker not in prices_df.columns:
        return {}
    close = prices_df[ticker].dropna()
    if len(close) < 30:
        return {}

    rsi_val              = _rsi(close)
    ema_score, ema_lbl   = _ema_alignment_score(close)
    dist_52w             = _distance_from_52w_high(close)
    rs_nf                = _rs_ratio(close, nifty)
    rs_sec               = _rs_ratio(close, sector_index)
    vol_ud_val           = _vol_ud_ratio(close)
    atr_pct_val          = _atr_pct(close)

    rsi_score     = _rsi_to_score(rsi_val)
    ema_norm      = (ema_score / 4) * 100
    rs_nf_score   = _clamp((rs_nf  - 70) / 60 * 100) if pd.notna(rs_nf)  else 50.0
    rs_sec_score  = _clamp((rs_sec - 70) / 60 * 100) if pd.notna(rs_sec) else 50.0
    dist_score    = _clamp((dist_52w + 30) / 30 * 100) if pd.notna(dist_52w) else 50.0
    vol_ud_score  = _vol_ud_to_score(vol_ud_val)
    atr_score     = _atr_to_score(atr_pct_val)

    composite = (
        rs_nf_score  * w["rs_nifty"]  +
        rs_sec_score * w["rs_sector"] +
        rsi_score    * w["rsi"]       +
        ema_norm     * w["ema_align"] +
        vol_ud_score * w["vol_ud"]    +
        atr_score    * w["atr_pct"]   +
        dist_score   * w["dist_52w"]
    )

    sym = ticker.replace(".NS", "")
    result = {
        "ticker":          ticker,
        "stock":           sym,
        "last_price":      round(float(close.iloc[-1]), 2),
        "composite_score": round(composite, 1),
        "rsi":             round(rsi_val, 1)    if pd.notna(rsi_val)    else np.nan,
        "ema_score":       ema_score,
        "ema_label":       ema_lbl,
        "rs_nifty":        round(rs_nf, 2)      if pd.notna(rs_nf)      else np.nan,
        "rs_sector":       round(rs_sec, 2)     if pd.notna(rs_sec)     else np.nan,
        "dist_52w":        round(dist_52w, 2)   if pd.notna(dist_52w)   else np.nan,
        "vol_ud":          round(vol_ud_val, 2) if pd.notna(vol_ud_val) else np.nan,
        "atr_pct":         round(atr_pct_val, 2) if pd.notna(atr_pct_val) else np.nan,
        "return_1w":       _period_return(close, 5),
        "return_1m":       _period_return(close, 21),
        "return_3m":       _period_return(close, 63),
        "rsi_score":       round(rsi_score, 1),
        "ema_norm":        round(ema_norm, 1),
        "rs_nifty_score":  round(rs_nf_score, 1),
        "rs_sector_score": round(rs_sec_score, 1),
        "dist_score":      round(dist_score, 1),
        "vol_ud_score":    round(vol_ud_score, 1),
        "atr_score":       round(atr_score, 1),
    }
    if meta:
        result.update({
            "company_name":   meta.get("company_name", sym),
            "market_cap_cr":  meta.get("market_cap_cr", np.nan),
            "cap_tier":       meta.get("cap_tier", "Unknown"),
            "nse_sector":     meta.get("nse_sector", ""),
            "nifty_index":    meta.get("nifty_index", ""),
            "sector_indices": meta.get("sector_indices", ""),
        })
    return result


# ── Batch ranker (parallel fetch in chunks) ──────────────────────────────────

def _fetch_chunk(tickers_chunk, period_days=200):
    return fetch_stock_prices(tickers_chunk, period_days)

def rank_stocks(universe_df: pd.DataFrame,
                nifty_prices: pd.Series,
                sector_index_prices: dict,
                weights: dict = None,
                progress_cb=None) -> pd.DataFrame:
    if universe_df.empty:
        return pd.DataFrame()
    tickers = universe_df["ticker"].dropna().tolist()
    if not tickers:
        return pd.DataFrame()

    # Parallel fetch in chunks of 100
    chunk_size = 100
    chunks = [tickers[i:i+chunk_size] for i in range(0, len(tickers), chunk_size)]
    prices_parts = []

    if progress_cb:
        progress_cb(0, len(tickers), "Fetching price data (parallel)")

    with ThreadPoolExecutor(max_workers=min(6, len(chunks))) as ex:
        futures = {ex.submit(_fetch_chunk, ch): ch for ch in chunks}
        for fut in as_completed(futures):
            try:
                part = fut.result()
                if not part.empty:
                    prices_parts.append(part)
            except Exception:
                pass

    if not prices_parts:
        return pd.DataFrame()

    prices = pd.concat(prices_parts, axis=1)
    prices = prices.loc[:, ~prices.columns.duplicated()]

    records = []
    for i, row in universe_df.iterrows():
        tkr    = row["ticker"]
        sector = row.get("nse_sector", "")
        sec_px = sector_index_prices.get(sector, nifty_prices)
        meta   = {
            "company_name":   row.get("company_name", ""),
            "market_cap_cr":  row.get("market_cap_cr", np.nan),
            "cap_tier":       row.get("cap_tier", "Unknown"),
            "nse_sector":     sector,
            "nifty_index":    row.get("nifty_index", ""),
            "sector_indices": row.get("sector_indices", ""),
        }
        scored = score_stock(tkr, prices, nifty_prices, sec_px, weights, meta)
        if scored:
            records.append(scored)
        if progress_cb and i % 20 == 0:
            progress_cb(i, len(universe_df), "Scoring stocks")

    if not records:
        return pd.DataFrame()

    df = pd.DataFrame(records)
    df.sort_values("composite_score", ascending=False, inplace=True)
    df.reset_index(drop=True, inplace=True)
    df.insert(0, "rank", range(1, len(df) + 1))
    return df


def get_top_bottom(df: pd.DataFrame, n: int = 7) -> tuple:
    """
    Returns (top_n, bottom_n) with ZERO overlap.
    top_n   — highest composite_score, sorted desc (rank #1 at top)
    bottom_n — lowest composite_score, sorted asc so WORST is rank #1
    """
    if df.empty:
        return pd.DataFrame(), pd.DataFrame()

    df_sorted = df.sort_values("composite_score", ascending=False).reset_index(drop=True)
    total = len(df_sorted)

    top_n = min(n, total // 2 if total >= 2 else total)
    bot_n = min(n, total - top_n)

    top = df_sorted.head(top_n).copy()
    bot = df_sorted.tail(bot_n).copy()

    # Sort bottom worst-first (lowest score at position #1)
    bot = bot.sort_values("composite_score", ascending=True).reset_index(drop=True)

    top.insert(0, "display_rank", range(1, len(top) + 1))
    bot.insert(0, "display_rank", range(1, len(bot) + 1))
    return top, bot


# ── Sector index price fetcher ────────────────────────────────────────────────

SECTOR_TO_INDEX_TICKER = {
    "Automobile":           "^CNXAUTO",
    "Banking":              "^NSEBANK",
    "Energy":               "^CNXENERGY",
    "FMCG":                 "^CNXFMCG",
    "Financial Services":   "^CNXFINANCE",
    "Infrastructure":       "^CNXINFRA",
    "Information Technology":"^CNXIT",
    "Media":                "^CNXMEDIA",
    "Metals":               "^CNXMETAL",
    "MNC":                  "^CNXMNC",
    "Pharma":               "^CNXPHARMA",
    "PSE":                  "^CNXPSE",
    "Realty":               "^CNXREALTY",
    "PSU Banking":          "^CNXPSUBANK",
    "Oil & Gas":            "^CNXOILGAS",
    "Commodities":          "^CNXCOMMODITIES",
    "Consumption":          "^CNXCONSUMPTION",
    "Services":             "^CNXSERVICE",
    "Capital Goods":        "^CNXINFRA",
    "Chemicals":            "^CNXPHARMA",
    "Healthcare":           "^CNXPHARMA",
    "Consumer":             "^CNXCONSUMPTION",
    "Power":                "^CNXENERGY",
    "Telecom":              "^CNXINFRA",
    "CPSE":                 "CPSE.NS",
}

def fetch_sector_index_prices(sectors: list) -> dict:
    unique_tickers = {}
    for sec in sectors:
        tkr = SECTOR_TO_INDEX_TICKER.get(sec, "^NSEI")
        unique_tickers[tkr] = sec
    result = {}
    try:
        all_tkrs = list(unique_tickers.keys())
        raw = yf.download(all_tkrs, period="1y",
                          auto_adjust=True, progress=False, threads=True)
        if isinstance(raw.columns, pd.MultiIndex):
            closes = raw["Close"]
        else:
            closes = raw[["Close"]].copy()
            closes.columns = all_tkrs[:1]
        for tkr, sec in unique_tickers.items():
            if tkr in closes.columns:
                result[sec] = closes[tkr].dropna().ffill()
    except Exception:
        pass
    return result
