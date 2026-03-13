"""
screener_engine.py
==================
All NSE stock screener logic — standalone module.
Screeners implemented:
  1. Turnover Screener     — your exact TradingView conditions from the screenshot
  2. 52-Week High Scanner  — stocks within X% of 52W high
  3. All-Time High Scanner — stocks at or near all-time high (3Y lookback)
  4. Volume Breakout       — volume spike > N× avg
  5. EMA Crossover         — fresh EMA10 > EMA20 crossover
  6. Momentum Breakout     — RSI > 55, price > EMA50, close > prev day high
  7. Bull Trend Setup      — full EMA stack + RS > 100

Uses yfinance bulk download — no external API needed.
"""

import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime, timedelta
from concurrent.futures import ThreadPoolExecutor, as_completed
import warnings
warnings.filterwarnings("ignore")

# ── Price + Volume fetcher ────────────────────────────────────────────────────

def _fetch_ohlcv(tickers: list, period_days: int = 365) -> tuple:
    """Returns (close_df, volume_df, high_df, low_df) all aligned on same index."""
    if not tickers:
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), pd.DataFrame()

    start = (datetime.today() - timedelta(days=period_days)).strftime("%Y-%m-%d")
    try:
        raw = yf.download(
            tickers, start=start,
            auto_adjust=True, progress=False, threads=True,
        )
    except Exception:
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), pd.DataFrame()

    if raw.empty:
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), pd.DataFrame()

    def _extract(field):
        if isinstance(raw.columns, pd.MultiIndex):
            if field in raw.columns.get_level_values(0):
                df = raw[field].copy()
            else:
                return pd.DataFrame()
        else:
            if field in raw.columns:
                df = raw[[field]].copy()
                if len(tickers) == 1:
                    df.columns = [tickers[0]]
            else:
                return pd.DataFrame()
        df.replace([np.inf, -np.inf], np.nan, inplace=True)
        df.ffill(limit=3, inplace=True)
        return df

    close  = _extract("Close")
    volume = _extract("Volume")
    high   = _extract("High")
    low    = _extract("Low")
    return close, volume, high, low


def _fetch_ohlcv_chunks(tickers: list, period_days: int = 365,
                         chunk_size: int = 150) -> tuple:
    """Parallel chunked OHLCV fetch — handles large universes cleanly."""
    chunks = [tickers[i:i+chunk_size] for i in range(0, len(tickers), chunk_size)]
    closes, volumes, highs, lows = [], [], [], []

    with ThreadPoolExecutor(max_workers=min(6, len(chunks))) as ex:
        futs = {ex.submit(_fetch_ohlcv, ch, period_days): ch for ch in chunks}
        for fut in as_completed(futs):
            try:
                c, v, h, l = fut.result()
                if not c.empty:
                    closes.append(c); volumes.append(v)
                    highs.append(h);  lows.append(l)
            except Exception:
                pass

    def _merge(parts):
        if not parts: return pd.DataFrame()
        df = pd.concat(parts, axis=1)
        return df.loc[:, ~df.columns.duplicated()]

    return _merge(closes), _merge(volumes), _merge(highs), _merge(lows)


# ── Technical helpers ─────────────────────────────────────────────────────────

def _ema(s: pd.Series, span: int) -> pd.Series:
    return s.ewm(span=span, adjust=False).mean()

def _sma(s: pd.Series, window: int) -> pd.Series:
    return s.rolling(window, min_periods=1).mean()

def _rsi(s: pd.Series, period: int = 14) -> float:
    d = s.diff().dropna()
    if len(d) < period: return np.nan
    g = d.clip(lower=0).rolling(period).mean()
    l = (-d.clip(upper=0)).rolling(period).mean()
    rs = g / l.replace(0, np.nan)
    rsi = (100 - 100/(1+rs)).dropna()
    return float(rsi.iloc[-1]) if len(rsi) > 0 else np.nan

def _vwap_approx(close: pd.Series, volume: pd.Series, window: int = 20) -> float:
    """Approximate VWAP over last `window` bars (no intraday data in daily)."""
    c = close.dropna().tail(window)
    v = volume.reindex(c.index).fillna(0)
    if v.sum() == 0: return float(c.iloc[-1])
    return float((c * v).sum() / v.sum())


# ── Base screener runner ──────────────────────────────────────────────────────

def _run_screener(universe_df: pd.DataFrame,
                  condition_fn,
                  period_days: int = 400,
                  progress_cb=None) -> pd.DataFrame:
    """
    Fetches OHLCV for all tickers in universe_df, then applies condition_fn
    per ticker. condition_fn(ticker, close, volume, high, low, meta) → dict or None.
    Returns DataFrame of passing stocks.
    """
    tickers = universe_df["ticker"].dropna().tolist()
    if not tickers:
        return pd.DataFrame()

    if progress_cb:
        progress_cb(0, len(tickers), "Fetching price & volume data")

    close, volume, high, low = _fetch_ohlcv_chunks(tickers, period_days)

    if close.empty:
        return pd.DataFrame()

    meta_map = {
        row["ticker"]: row.to_dict()
        for _, row in universe_df.iterrows()
    }

    results = []
    for i, tkr in enumerate(tickers):
        if tkr not in close.columns:
            continue
        c = close[tkr].dropna()
        v = volume[tkr].dropna() if tkr in volume.columns else pd.Series(dtype=float)
        h = high[tkr].dropna()   if tkr in high.columns   else pd.Series(dtype=float)
        l = low[tkr].dropna()    if tkr in low.columns    else pd.Series(dtype=float)

        if len(c) < 20:
            continue

        meta = meta_map.get(tkr, {})
        try:
            rec = condition_fn(tkr, c, v, h, l, meta)
            if rec:
                results.append(rec)
        except Exception:
            pass

        if progress_cb and i % 30 == 0:
            progress_cb(i, len(tickers), "Scanning stocks")

    if not results:
        return pd.DataFrame()

    df = pd.DataFrame(results)
    df.sort_values("score", ascending=False, inplace=True)
    df.reset_index(drop=True, inplace=True)
    df.insert(0, "rank", range(1, len(df)+1))
    return df


# ══════════════════════════════════════════════════════════════════════════════
# SCREENER 1 — Turnover Screener (exact conditions from your screenshot)
# ══════════════════════════════════════════════════════════════════════════════
# Conditions (ALL must pass):
#   1. Close * Volume >= 1,000,000,000  (turnover >= 100 Cr)
#   2. RSI(14) >= 55
#   3. Close * Volume > SMA(Close*Volume, 30) * 2  (turnover > 2× 30-day avg)
#   4. Volume > SMA(Volume, 20) * 2                (volume spike)
#   5. Close > EMA(Close, 50)
#   6. Close > VWAP (approximate)
#   7. Close > Previous day High
#   8. Market Cap <= 100,000 Cr  (mcap_max)
#   9. Market Cap >= 1,000 Cr    (mcap_min)
#  10. EMA(10) > EMA(20)
#  11. EMA(20) > EMA(200)

def screener_turnover(universe_df: pd.DataFrame,
                      progress_cb=None,
                      mcap_min_cr: float = 1000,
                      mcap_max_cr: float = 100000,
                      rsi_min: float = 55,
                      turnover_min: float = 1_000_000_000,
                      turnover_multiplier: float = 2.0,
                      vol_multiplier: float = 2.0) -> pd.DataFrame:
    """
    Your TradingView turnover screener — all 11 conditions.
    Returns passing stocks sorted by turnover desc.
    """

    # Filter universe by mcap first (fast, no fetch needed)
    uni = universe_df.copy()
    if "market_cap_cr" in uni.columns:
        uni = uni[
            (uni["market_cap_cr"].fillna(0) >= mcap_min_cr) &
            (uni["market_cap_cr"].fillna(0) <= mcap_max_cr)
        ].reset_index(drop=True)

    def _condition(tkr, c, v, h, l, meta):
        if len(c) < 30 or len(v) < 30:
            return None

        last_close  = float(c.iloc[-1])
        last_vol    = float(v.iloc[-1]) if len(v) > 0 else 0
        prev_high   = float(h.iloc[-2]) if len(h) >= 2 else np.nan

        # 1. Turnover check
        turnover    = last_close * last_vol
        if turnover < turnover_min:
            return None

        # 2. RSI >= rsi_min
        rsi_val = _rsi(c)
        if pd.isna(rsi_val) or rsi_val < rsi_min:
            return None

        # 3. Turnover > SMA(Turnover,30) * multiplier
        to_series   = c * v.reindex(c.index).fillna(0)
        to_sma30    = float(_sma(to_series, 30).iloc[-1])
        if to_sma30 <= 0 or turnover <= to_sma30 * turnover_multiplier:
            return None

        # 4. Volume > SMA(Volume, 20) * 2
        vol_sma20 = float(_sma(v, 20).iloc[-1])
        if vol_sma20 <= 0 or last_vol <= vol_sma20 * vol_multiplier:
            return None

        # 5. Close > EMA50
        ema50 = float(_ema(c, 50).iloc[-1])
        if last_close <= ema50:
            return None

        # 6. Close > approx VWAP
        vwap = _vwap_approx(c, v)
        if last_close <= vwap:
            return None

        # 7. Close > Previous day High
        if pd.notna(prev_high) and last_close <= prev_high:
            return None

        # 10. EMA10 > EMA20
        ema10 = float(_ema(c, 10).iloc[-1])
        ema20 = float(_ema(c, 20).iloc[-1])
        if ema10 <= ema20:
            return None

        # 11. EMA20 > EMA200
        if len(c) < 200:
            return None
        ema200 = float(_ema(c, 200).iloc[-1])
        if ema20 <= ema200:
            return None

        sym = tkr.replace(".NS","")
        vol_ratio = round(last_vol / vol_sma20, 1) if vol_sma20 > 0 else 0
        to_ratio  = round(turnover / to_sma30, 1) if to_sma30 > 0 else 0
        dist_52w  = _dist_from_52w(c)
        score     = round(rsi_val * 0.3 + vol_ratio * 10 + to_ratio * 10, 1)

        return {
            "ticker":        tkr,
            "stock":         sym,
            "company":       meta.get("company_name", sym),
            "sector":        meta.get("nse_sector",""),
            "cap_tier":      meta.get("cap_tier",""),
            "mcap_cr":       meta.get("market_cap_cr", np.nan),
            "last_price":    round(last_close, 2),
            "turnover_cr":   round(turnover / 1e7, 2),
            "turnover_ratio":to_ratio,
            "vol_ratio":     vol_ratio,
            "rsi":           round(rsi_val, 1),
            "ema10":         round(ema10, 2),
            "ema20":         round(ema20, 2),
            "ema50":         round(ema50, 2),
            "ema200":        round(ema200, 2),
            "vwap":          round(vwap, 2),
            "dist_52w_pct":  dist_52w,
            "score":         score,
        }

    return _run_screener(uni, _condition, period_days=400, progress_cb=progress_cb)


# ══════════════════════════════════════════════════════════════════════════════
# SCREENER 2 — 52-Week High Scanner
# ══════════════════════════════════════════════════════════════════════════════

def _dist_from_52w(close: pd.Series) -> float:
    yr = close.dropna().tail(252)
    if len(yr) < 20: return np.nan
    high = yr.max()
    last = yr.iloc[-1]
    if high == 0: return np.nan
    return round((last - high) / high * 100, 2)

def screener_52w_high(universe_df: pd.DataFrame,
                      within_pct: float = 3.0,
                      rsi_min: float = 50,
                      progress_cb=None) -> pd.DataFrame:
    """
    Stocks within `within_pct`% of their 52-week high.
    E.g. within_pct=3 → stock is max 3% below the 52W high.
    """
    def _condition(tkr, c, v, h, l, meta):
        dist = _dist_from_52w(c)
        if pd.isna(dist) or dist < -within_pct:
            return None
        rsi_val  = _rsi(c)
        yr       = c.dropna().tail(252)
        high_52w = float(yr.max())
        low_52w  = float(yr.min())
        last     = float(c.iloc[-1])

        # Only take RSI-filtered stocks (avoid overbought exhaustion)
        if pd.notna(rsi_val) and rsi_val < rsi_min:
            return None

        vol_sma20 = float(_sma(v, 20).iloc[-1]) if len(v) >= 20 else 1
        vol_ratio = round(float(v.iloc[-1]) / vol_sma20, 2) if vol_sma20 > 0 else 1

        # Score: closer to high = higher score
        score = round((1 - abs(dist) / within_pct) * 100, 1) if within_pct > 0 else 100

        sym = tkr.replace(".NS","")
        return {
            "ticker":       tkr,
            "stock":        sym,
            "company":      meta.get("company_name", sym),
            "sector":       meta.get("nse_sector",""),
            "cap_tier":     meta.get("cap_tier",""),
            "mcap_cr":      meta.get("market_cap_cr", np.nan),
            "last_price":   round(last, 2),
            "high_52w":     round(high_52w, 2),
            "low_52w":      round(low_52w, 2),
            "dist_from_high_pct": dist,
            "rsi":          round(rsi_val, 1) if pd.notna(rsi_val) else np.nan,
            "vol_ratio":    vol_ratio,
            "score":        score,
        }

    return _run_screener(universe_df, _condition, period_days=400, progress_cb=progress_cb)


# ══════════════════════════════════════════════════════════════════════════════
# SCREENER 3 — All-Time High Scanner (3-Year lookback)
# ══════════════════════════════════════════════════════════════════════════════

def screener_ath(universe_df: pd.DataFrame,
                 within_pct: float = 5.0,
                 progress_cb=None) -> pd.DataFrame:
    """
    Stocks within `within_pct`% of their All-Time High (3-year lookback).
    Needs period_days=1100 to cover ~3 years.
    """
    def _condition(tkr, c, v, h, l, meta):
        s = c.dropna()
        if len(s) < 50: return None
        ath    = float(s.max())
        last   = float(s.iloc[-1])
        if ath == 0: return None
        dist   = (last - ath) / ath * 100
        if dist < -within_pct:
            return None

        rsi_val   = _rsi(s)
        vol_sma20 = float(_sma(v, 20).iloc[-1]) if len(v) >= 20 else 1
        vol_ratio = round(float(v.iloc[-1]) / vol_sma20, 2) if vol_sma20 > 0 else 1

        # ATH date (when was it last at ATH)
        ath_idx   = s.idxmax()
        days_from_ath = (s.index[-1] - ath_idx).days if hasattr(ath_idx, 'days') else 0

        score = round((1 - abs(dist)/within_pct)*80 + (vol_ratio-1)*10, 1)

        sym = tkr.replace(".NS","")
        return {
            "ticker":          tkr,
            "stock":           sym,
            "company":         meta.get("company_name", sym),
            "sector":          meta.get("nse_sector",""),
            "cap_tier":        meta.get("cap_tier",""),
            "mcap_cr":         meta.get("market_cap_cr", np.nan),
            "last_price":      round(last, 2),
            "all_time_high":   round(ath, 2),
            "dist_from_ath_pct": round(dist, 2),
            "rsi":             round(rsi_val,1) if pd.notna(rsi_val) else np.nan,
            "vol_ratio":       vol_ratio,
            "score":           score,
        }

    return _run_screener(universe_df, _condition, period_days=1100, progress_cb=progress_cb)


# ══════════════════════════════════════════════════════════════════════════════
# SCREENER 4 — Volume Breakout
# ══════════════════════════════════════════════════════════════════════════════

def screener_volume_breakout(universe_df: pd.DataFrame,
                              vol_multiplier: float = 2.0,
                              rsi_min: float = 50,
                              progress_cb=None) -> pd.DataFrame:
    """Volume today > N× 20-day avg AND price above EMA50."""
    def _condition(tkr, c, v, h, l, meta):
        if len(v) < 21: return None
        last_vol  = float(v.iloc[-1])
        vol_sma20 = float(_sma(v, 20).iloc[-2])  # use yesterday avg to avoid self-comparison
        if vol_sma20 <= 0 or last_vol < vol_sma20 * vol_multiplier:
            return None
        last_close = float(c.iloc[-1])
        ema50      = float(_ema(c, 50).iloc[-1])
        if last_close < ema50:
            return None
        rsi_val = _rsi(c)
        if pd.notna(rsi_val) and rsi_val < rsi_min:
            return None
        vol_ratio = round(last_vol / vol_sma20, 1)
        dist      = _dist_from_52w(c)
        score     = round(vol_ratio * 20 + (rsi_val or 50) * 0.5, 1)
        sym = tkr.replace(".NS","")
        return {
            "ticker": tkr, "stock": sym,
            "company": meta.get("company_name", sym),
            "sector": meta.get("nse_sector",""),
            "cap_tier": meta.get("cap_tier",""),
            "mcap_cr": meta.get("market_cap_cr", np.nan),
            "last_price": round(last_close, 2),
            "vol_ratio": vol_ratio,
            "rsi": round(rsi_val,1) if pd.notna(rsi_val) else np.nan,
            "ema50": round(ema50, 2),
            "dist_52w_pct": dist,
            "score": score,
        }
    return _run_screener(universe_df, _condition, period_days=300, progress_cb=progress_cb)


# ══════════════════════════════════════════════════════════════════════════════
# SCREENER 5 — EMA Crossover (EMA10 just crossed above EMA20)
# ══════════════════════════════════════════════════════════════════════════════

def screener_ema_crossover(universe_df: pd.DataFrame,
                            lookback_bars: int = 3,
                            progress_cb=None) -> pd.DataFrame:
    """EMA10 crossed above EMA20 within the last `lookback_bars` candles."""
    def _condition(tkr, c, v, h, l, meta):
        if len(c) < 25: return None
        e10 = _ema(c, 10)
        e20 = _ema(c, 20)
        diff = e10 - e20
        # Check if crossover happened within last `lookback_bars`
        crossed = False
        for i in range(1, lookback_bars + 1):
            if (float(diff.iloc[-i]) > 0) and (float(diff.iloc[-(i+1)]) <= 0):
                crossed = True; break
        if not crossed: return None

        last   = float(c.iloc[-1])
        rsi_val = _rsi(c)
        ema50   = float(_ema(c, 50).iloc[-1])
        dist    = _dist_from_52w(c)
        score   = round((rsi_val or 50) * 0.5 + 50, 1)
        sym = tkr.replace(".NS","")
        return {
            "ticker": tkr, "stock": sym,
            "company": meta.get("company_name", sym),
            "sector": meta.get("nse_sector",""),
            "cap_tier": meta.get("cap_tier",""),
            "mcap_cr": meta.get("market_cap_cr", np.nan),
            "last_price": round(last, 2),
            "ema10": round(float(e10.iloc[-1]), 2),
            "ema20": round(float(e20.iloc[-1]), 2),
            "ema50": round(ema50, 2),
            "rsi": round(rsi_val,1) if pd.notna(rsi_val) else np.nan,
            "dist_52w_pct": dist,
            "score": score,
        }
    return _run_screener(universe_df, _condition, period_days=200, progress_cb=progress_cb)


# ══════════════════════════════════════════════════════════════════════════════
# SCREENER 6 — Momentum Breakout
# ══════════════════════════════════════════════════════════════════════════════

def screener_momentum_breakout(universe_df: pd.DataFrame,
                                rsi_min: float = 55,
                                progress_cb=None) -> pd.DataFrame:
    """RSI > 55, Close > EMA50, Close > Previous Day High."""
    def _condition(tkr, c, v, h, l, meta):
        if len(c) < 51 or len(h) < 2: return None
        last   = float(c.iloc[-1])
        ema50  = float(_ema(c, 50).iloc[-1])
        prev_h = float(h.iloc[-2])
        rsi_val = _rsi(c)
        if pd.isna(rsi_val) or rsi_val < rsi_min: return None
        if last <= ema50: return None
        if last <= prev_h: return None
        ema10  = float(_ema(c, 10).iloc[-1])
        ema20  = float(_ema(c, 20).iloc[-1])
        dist   = _dist_from_52w(c)
        vol_sma20 = float(_sma(v, 20).iloc[-1]) if len(v) >= 20 else 1
        vol_ratio = round(float(v.iloc[-1]) / vol_sma20, 1) if vol_sma20 > 0 else 1
        score  = round(rsi_val * 0.5 + vol_ratio * 5 + (100 - abs(dist or 0)) * 0.2, 1)
        sym = tkr.replace(".NS","")
        return {
            "ticker": tkr, "stock": sym,
            "company": meta.get("company_name", sym),
            "sector": meta.get("nse_sector",""),
            "cap_tier": meta.get("cap_tier",""),
            "mcap_cr": meta.get("market_cap_cr", np.nan),
            "last_price": round(last, 2),
            "prev_high":  round(prev_h, 2),
            "rsi": round(rsi_val, 1),
            "ema10": round(ema10, 2),
            "ema20": round(ema20, 2),
            "ema50": round(ema50, 2),
            "vol_ratio":  vol_ratio,
            "dist_52w_pct": dist,
            "score": score,
        }
    return _run_screener(universe_df, _condition, period_days=300, progress_cb=progress_cb)


# ══════════════════════════════════════════════════════════════════════════════
# SCREENER 7 — Bull Trend Setup (Full EMA stack + RS > 100)
# ══════════════════════════════════════════════════════════════════════════════

def screener_bull_trend(universe_df: pd.DataFrame,
                         nifty_prices: pd.Series = None,
                         progress_cb=None) -> pd.DataFrame:
    """
    Full bull stack: Price > EMA20 > EMA50 > EMA200, RSI 55-75,
    EMA20 slope positive, optional RS vs NIFTY > 100.
    """
    def _condition(tkr, c, v, h, l, meta):
        if len(c) < 200: return None
        last   = float(c.iloc[-1])
        e20    = _ema(c, 20)
        e50    = _ema(c, 50)
        e200   = _ema(c, 200)
        e20v   = float(e20.iloc[-1])
        e50v   = float(e50.iloc[-1])
        e200v  = float(e200.iloc[-1])
        # Full stack
        if not (last > e20v > e50v > e200v): return None
        rsi_val = _rsi(c)
        if pd.isna(rsi_val) or not (55 <= rsi_val <= 78): return None
        # EMA20 slope up
        e20_slope = e20v - float(e20.iloc[-6]) if len(e20) >= 6 else 0
        if e20_slope <= 0: return None
        dist   = _dist_from_52w(c)
        vol_sma20 = float(_sma(v, 20).iloc[-1]) if len(v) >= 20 else 1
        vol_ratio = round(float(v.iloc[-1]) / vol_sma20, 1) if vol_sma20 > 0 else 1
        ema_stack = 4  # all 4 aligned

        # RS vs NIFTY (optional)
        rs_score = 50.0
        if nifty_prices is not None:
            common = c.index.intersection(nifty_prices.index)
            if len(common) >= 20:
                rs = (c.loc[common] / nifty_prices.loc[common].replace(0, np.nan)).dropna()
                rm = rs.rolling(252, min_periods=20).mean().replace(0, np.nan)
                rn = (rs / rm * 100).dropna()
                if len(rn) > 0:
                    rs_score = float(rn.iloc[-1])

        score = round(rsi_val*0.3 + ema_stack*10 + (rs_score-90)*0.5 + vol_ratio*5, 1)
        sym = tkr.replace(".NS","")
        return {
            "ticker": tkr, "stock": sym,
            "company": meta.get("company_name", sym),
            "sector": meta.get("nse_sector",""),
            "cap_tier": meta.get("cap_tier",""),
            "mcap_cr": meta.get("market_cap_cr", np.nan),
            "last_price": round(last, 2),
            "ema20": round(e20v, 2),
            "ema50": round(e50v, 2),
            "ema200": round(e200v, 2),
            "ema_slope_20": round(e20_slope, 2),
            "rsi": round(rsi_val, 1),
            "rs_nifty": round(rs_score, 1),
            "vol_ratio": vol_ratio,
            "dist_52w_pct": dist,
            "score": score,
        }

    return _run_screener(universe_df, _condition, period_days=400, progress_cb=progress_cb)


# ── Screener registry (for UI) ────────────────────────────────────────────────

SCREENER_REGISTRY = {
    "📊 Turnover Screener": {
        "key":  "turnover",
        "fn":   screener_turnover,
        "desc": (
            "Your TradingView screener — all 11 conditions: "
            "Turnover ≥ 100 Cr, RSI ≥ 55, Turnover > 2× avg, Volume > 2× avg, "
            "Close > EMA50 > VWAP, Close > Prev High, EMA10 > EMA20 > EMA200, "
            "MCap 1,000–1,00,000 Cr"
        ),
        "params": {
            "mcap_min_cr":          {"label":"Min MCap (Cr)",          "type":"number","default":1000,  "step":500},
            "mcap_max_cr":          {"label":"Max MCap (Cr)",          "type":"number","default":100000,"step":5000},
            "rsi_min":              {"label":"Min RSI",                "type":"slider","default":55,   "min":30,"max":80},
            "turnover_min":         {"label":"Min Turnover (₹)",       "type":"number","default":1_000_000_000,"step":100_000_000},
            "turnover_multiplier":  {"label":"Turnover vs Avg (×)",    "type":"slider","default":2.0,  "min":1.0,"max":5.0},
            "vol_multiplier":       {"label":"Volume vs Avg (×)",      "type":"slider","default":2.0,  "min":1.0,"max":5.0},
        },
    },
    "🏔️ 52-Week High": {
        "key":  "52w_high",
        "fn":   screener_52w_high,
        "desc": "Stocks within X% of their 52-week high — momentum leaders near breakout.",
        "params": {
            "within_pct": {"label":"Within % of 52W High","type":"slider","default":3.0,"min":0.5,"max":20.0},
            "rsi_min":    {"label":"Min RSI",              "type":"slider","default":50.0,"min":30,"max":75},
        },
    },
    "🚀 All-Time High": {
        "key":  "ath",
        "fn":   screener_ath,
        "desc": "Stocks within X% of their All-Time High (3-year lookback). True momentum leaders.",
        "params": {
            "within_pct": {"label":"Within % of ATH","type":"slider","default":5.0,"min":0.5,"max":25.0},
        },
    },
    "📈 Volume Breakout": {
        "key":  "vol_breakout",
        "fn":   screener_volume_breakout,
        "desc": "Volume today > N× 20-day avg AND price above EMA50. Signals institutional interest.",
        "params": {
            "vol_multiplier": {"label":"Volume vs Avg (×)","type":"slider","default":2.0,"min":1.5,"max":8.0},
            "rsi_min":        {"label":"Min RSI",           "type":"slider","default":50.0,"min":30,"max":75},
        },
    },
    "⚡ EMA Crossover": {
        "key":  "ema_cross",
        "fn":   screener_ema_crossover,
        "desc": "EMA10 just crossed above EMA20 within last N candles — fresh momentum signal.",
        "params": {
            "lookback_bars": {"label":"Within last N bars","type":"slider","default":3,"min":1,"max":10},
        },
    },
    "💥 Momentum Breakout": {
        "key":  "momentum",
        "fn":   screener_momentum_breakout,
        "desc": "RSI > 55, Close > EMA50, and today's close above previous day's high.",
        "params": {
            "rsi_min": {"label":"Min RSI","type":"slider","default":55,"min":40,"max":75},
        },
    },
    "🐂 Bull Trend Setup": {
        "key":  "bull_trend",
        "fn":   screener_bull_trend,
        "desc": "Full bull stack: Price > EMA20 > EMA50 > EMA200, RSI 55–78, EMA20 slope up.",
        "params": {},
    },
}
