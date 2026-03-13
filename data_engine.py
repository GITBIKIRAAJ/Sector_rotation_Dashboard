"""
data_engine.py  (v2 — parallel fetch + RRG days-in-quadrant)
"""
import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from concurrent.futures import ThreadPoolExecutor
import warnings
warnings.filterwarnings("ignore")

SECTORS = {
    "NIFTY AUTO":         "^CNXAUTO",
    "NIFTY BANK":         "^NSEBANK",
    "NIFTY ENERGY":       "^CNXENERGY",
    "NIFTY FMCG":         "^CNXFMCG",
    "NIFTY FINANCE":      "^CNXFINANCE",
    "NIFTY INFRA":        "^CNXINFRA",
    "NIFTY IT":           "^CNXIT",
    "NIFTY MEDIA":        "^CNXMEDIA",
    "NIFTY METAL":        "^CNXMETAL",
    "NIFTY MNC":          "^CNXMNC",
    "NIFTY PHARMA":       "^CNXPHARMA",
    "NIFTY PSE":          "^CNXPSE",
    "NIFTY REALTY":       "^CNXREALTY",
    "NIFTY CPSE":         "CPSE.NS",
    "NIFTY PSU BANK":     "^CNXPSUBANK",
    "NIFTY COMMODITIES":  "^CNXCOMMODITIES",
    "NIFTY CONSUMPTION":  "^CNXCONSUMPTION",
    "NIFTY SERVICES":     "^CNXSERVICE",
    "NIFTY INDIA MFG":    "^CNXMFG",
    "NIFTY OIL & GAS":    "^CNXOILGAS",
}
BENCHMARK  = {"NIFTY 50": "^NSEI"}
TIMEFRAMES = {"1W": 7, "1M": 30, "3M": 90, "6M": 180, "YTD": None, "1Y": 365}

# ── Helpers ───────────────────────────────────────────────────────────────────

def _safe_int_series(s: pd.Series) -> pd.Series:
    def to_int(x):
        if pd.isna(x): return pd.NA
        try:
            f = float(x)
            return int(round(f)) if np.isfinite(f) else pd.NA
        except Exception:
            return pd.NA
    return s.apply(to_int).astype("Int64")

def _clean(series: pd.Series) -> pd.Series:
    return series.replace([np.inf, -np.inf], np.nan).ffill().bfill()

# ── Fetch raw price data (single bulk call) ───────────────────────────────────

def fetch_prices(period_days: int = 365) -> pd.DataFrame:
    """Single bulk download — all sectors + benchmark in one call."""
    start   = (datetime.today() - timedelta(days=period_days)).strftime("%Y-%m-%d")
    tickers = list(SECTORS.values()) + list(BENCHMARK.values())

    raw = yf.download(
        tickers, start=start,
        auto_adjust=True, progress=False, threads=True,
    )
    if raw.empty:
        raise ValueError("No data returned from Yahoo Finance.")

    prices = raw["Close"].copy() if isinstance(raw.columns, pd.MultiIndex) else raw[["Close"]].copy()
    reverse_map = {v: k for k, v in {**SECTORS, **BENCHMARK}.items()}
    prices.rename(columns=reverse_map, inplace=True)
    prices.dropna(axis=1, how="all", inplace=True)
    prices.dropna(how="all", inplace=True)
    prices.replace([np.inf, -np.inf], np.nan, inplace=True)
    prices.ffill(limit=3, inplace=True)
    prices.dropna(how="all", inplace=True)
    if prices.empty:
        raise ValueError("Prices DataFrame empty after cleaning.")
    return prices

# ── Core Calculations ─────────────────────────────────────────────────────────

def pct_return(prices: pd.DataFrame, days: int = None, ytd: bool = False) -> pd.Series:
    if ytd:
        year_start = datetime(datetime.today().year, 1, 1)
        subset = prices[prices.index >= year_start].dropna(how="all")
        if subset.empty:
            return pd.Series(dtype=float)
        start_prices = subset.bfill().iloc[0]
        end_prices   = subset.ffill().iloc[-1]
    else:
        if len(prices) < 2:
            return pd.Series(dtype=float)
        lookback     = min(days, len(prices) - 1)
        start_prices = prices.ffill().iloc[-(lookback + 1)]
        end_prices   = prices.ffill().iloc[-1]
    start_safe = start_prices.replace(0, np.nan)
    ret = ((end_prices - start_safe) / start_safe * 100)
    ret.replace([np.inf, -np.inf], np.nan, inplace=True)
    return ret.round(2)

def build_returns_table(prices: pd.DataFrame) -> pd.DataFrame:
    records = []
    for label, days in TIMEFRAMES.items():
        ret      = pct_return(prices, days=days, ytd=(label == "YTD"))
        ret.name = label
        records.append(ret)
    return pd.concat(records, axis=1)

def relative_returns(returns_df: pd.DataFrame) -> pd.DataFrame:
    rel            = returns_df.copy()
    benchmark_name = list(BENCHMARK.keys())[0]
    if benchmark_name not in rel.index:
        return rel
    for col in rel.columns:
        bm_val = rel.loc[benchmark_name, col]
        if pd.isna(bm_val):
            rel[col] = np.nan
        else:
            rel[col] = rel[col] - bm_val
    rel.replace([np.inf, -np.inf], np.nan, inplace=True)
    return rel.round(2)

def _assign_quadrant(rs_ratio: float, rs_mom: float) -> str:
    if pd.isna(rs_ratio) or pd.isna(rs_mom):
        return "Lagging"
    if rs_ratio >= 100 and rs_mom >= 0:
        return "Leading"
    if rs_ratio < 100  and rs_mom >= 0:
        return "Improving"
    if rs_ratio >= 100 and rs_mom < 0:
        return "Weakening"
    return "Lagging"

def compute_rs_ratio_momentum(prices: pd.DataFrame, window: int = 10) -> pd.DataFrame:
    benchmark_name = list(BENCHMARK.keys())[0]
    if benchmark_name not in prices.columns:
        return pd.DataFrame()
    bm      = _clean(prices[benchmark_name])
    results = {}
    for sector in SECTORS.keys():
        if sector not in prices.columns:
            continue
        sec    = _clean(prices[sector])
        common = bm.index.intersection(sec.index)
        if len(common) < 30:
            continue
        bm_c    = bm.loc[common]
        sec_c   = sec.loc[common]
        bm_safe = bm_c.replace(0, np.nan).ffill()
        rs      = (sec_c / bm_safe).replace([np.inf, -np.inf], np.nan).ffill()
        rs_mean = rs.rolling(252, min_periods=20).mean().replace(0, np.nan)
        rs_norm = (rs / rs_mean * 100).replace([np.inf, -np.inf], np.nan)
        rs_smooth = rs_norm.ewm(span=window, min_periods=5).mean()
        rs_mom    = rs_smooth.pct_change(window).mul(100).replace([np.inf, -np.inf], np.nan)
        ratio_val = float(rs_smooth.dropna().iloc[-1]) if len(rs_smooth.dropna()) > 0 else np.nan
        mom_val   = float(rs_mom.dropna().iloc[-1])    if len(rs_mom.dropna()) > 0    else np.nan
        results[sector] = {
            "RS_Ratio":    round(ratio_val, 2) if np.isfinite(ratio_val) else np.nan,
            "RS_Momentum": round(mom_val, 2)   if np.isfinite(mom_val)   else np.nan,
        }
    if not results:
        return pd.DataFrame()
    rrg = pd.DataFrame(results).T.copy()
    rrg["Quadrant"] = rrg.apply(
        lambda r: _assign_quadrant(r["RS_Ratio"], r["RS_Momentum"]), axis=1
    )
    return rrg


def compute_rrg_days_in_quadrant(prices: pd.DataFrame, window: int = 10) -> pd.DataFrame:
    """
    For each sector, count how many consecutive trading days it has been
    in its CURRENT quadrant by scanning daily RRG quadrant history.
    Returns DataFrame with columns: sector, Quadrant, Days_In_Quadrant
    """
    benchmark_name = list(BENCHMARK.keys())[0]
    if benchmark_name not in prices.columns:
        return pd.DataFrame()

    bm = _clean(prices[benchmark_name])
    records = {}

    for sector in SECTORS.keys():
        if sector not in prices.columns:
            continue
        sec    = _clean(prices[sector])
        common = bm.index.intersection(sec.index)
        if len(common) < 40:
            continue
        bm_c    = bm.loc[common]
        sec_c   = sec.loc[common]
        bm_safe = bm_c.replace(0, np.nan).ffill()
        rs      = (sec_c / bm_safe).replace([np.inf, -np.inf], np.nan).ffill()
        rs_mean = rs.rolling(252, min_periods=20).mean().replace(0, np.nan)
        rs_norm = (rs / rs_mean * 100).replace([np.inf, -np.inf], np.nan)
        rs_smooth = rs_norm.ewm(span=window, min_periods=5).mean()
        rs_mom    = rs_smooth.pct_change(window).mul(100).replace([np.inf, -np.inf], np.nan)

        # Build daily quadrant series
        ratio_s = rs_smooth.dropna()
        mom_s   = rs_mom.reindex(ratio_s.index)
        if len(ratio_s) < 5:
            continue

        daily_quads = [
            _assign_quadrant(ratio_s.iloc[i], mom_s.iloc[i])
            for i in range(len(ratio_s))
        ]
        if not daily_quads:
            continue

        current_quad = daily_quads[-1]
        count        = 0
        for q in reversed(daily_quads):
            if q == current_quad:
                count += 1
            else:
                break

        records[sector] = {"Quadrant": current_quad, "Days_In_Quadrant": count}

    if not records:
        return pd.DataFrame()
    df = pd.DataFrame(records).T
    df.index.name = "sector"
    df["Days_In_Quadrant"] = df["Days_In_Quadrant"].astype(int)
    return df


def compute_normalized_series(prices: pd.DataFrame, days: int = 90) -> pd.DataFrame:
    subset = prices.iloc[-days:].copy() if len(prices) >= days else prices.copy()
    subset.replace([np.inf, -np.inf], np.nan, inplace=True)
    first_valid = subset.ffill().bfill().iloc[0].replace(0, np.nan)
    normalized  = subset.div(first_valid).mul(100)
    normalized.replace([np.inf, -np.inf], np.nan, inplace=True)
    return normalized.round(2)

def compute_rolling_rs(prices: pd.DataFrame, days: int = 90) -> pd.DataFrame:
    benchmark_name = list(BENCHMARK.keys())[0]
    if benchmark_name not in prices.columns:
        return pd.DataFrame()
    subset = prices.iloc[-days:].copy() if len(prices) >= days else prices.copy()
    subset.replace([np.inf, -np.inf], np.nan, inplace=True)
    bm      = _clean(subset[benchmark_name]).replace(0, np.nan)
    rs_data = pd.DataFrame(index=subset.index)
    for sector in SECTORS.keys():
        if sector not in subset.columns:
            continue
        sec     = _clean(subset[sector])
        rs      = (sec / bm).replace([np.inf, -np.inf], np.nan)
        rs_first = rs.dropna().iloc[0] if rs.dropna().shape[0] > 0 else np.nan
        if pd.isna(rs_first) or rs_first == 0:
            continue
        rs_data[sector] = (rs / rs_first * 100).replace([np.inf, -np.inf], np.nan)
    return rs_data.round(2)

def compute_drawdown(prices: pd.DataFrame) -> pd.Series:
    year_start  = datetime(datetime.today().year, 1, 1)
    ytd_prices  = prices[prices.index >= year_start].copy()
    ytd_prices.replace([np.inf, -np.inf], np.nan, inplace=True)
    drawdowns = {}
    for col in ytd_prices.columns:
        series = ytd_prices[col].dropna()
        if len(series) < 2:
            drawdowns[col] = np.nan
            continue
        roll_max = series.cummax().replace(0, np.nan)
        dd       = ((series - roll_max) / roll_max * 100).replace([np.inf, -np.inf], np.nan)
        min_dd   = dd.min()
        drawdowns[col] = round(float(min_dd), 2) if pd.notna(min_dd) and np.isfinite(min_dd) else np.nan
    return pd.Series(drawdowns)

def compute_volatility(prices: pd.DataFrame, window: int = 20) -> pd.Series:
    daily_ret = prices.pct_change().replace([np.inf, -np.inf], np.nan).dropna(how="all")
    vol = daily_ret.rolling(window, min_periods=5).std().iloc[-1] * np.sqrt(252) * 100
    vol.replace([np.inf, -np.inf], np.nan, inplace=True)
    return vol.round(2)

def compute_rank_shift(prices: pd.DataFrame) -> pd.DataFrame:
    benchmark_name = list(BENCHMARK.keys())[0]
    if benchmark_name not in prices.columns:
        return pd.DataFrame(columns=["Rank_Now", "Rank_4W_Ago", "Shift"])

    def rs_rank_at(offset_days: int) -> pd.Series:
        if len(prices) <= offset_days:
            return pd.Series(dtype=float)
        sub = (prices.iloc[:len(prices) - offset_days] if offset_days > 0 else prices).copy()
        sub.replace([np.inf, -np.inf], np.nan, inplace=True)
        bm = _clean(sub[benchmark_name])
        bm_last_s = bm.dropna()
        if bm_last_s.empty:
            return pd.Series(dtype=float)
        bm_last = float(bm_last_s.iloc[-1])
        if not np.isfinite(bm_last) or bm_last == 0:
            return pd.Series(dtype=float)
        rs_vals = {}
        for s in SECTORS.keys():
            if s not in sub.columns:
                continue
            sec     = _clean(sub[s])
            sec_last_s = sec.dropna()
            if sec_last_s.empty:
                continue
            sec_last = float(sec_last_s.iloc[-1])
            if not np.isfinite(sec_last):
                continue
            rs = sec_last / bm_last
            if np.isfinite(rs):
                rs_vals[s] = rs
        if not rs_vals:
            return pd.Series(dtype=float)
        return pd.Series(rs_vals).rank(ascending=False, method="min")

    now_rank  = rs_rank_at(0)
    prev_rank = rs_rank_at(20)
    df        = pd.DataFrame({"Rank_Now": now_rank, "Rank_4W_Ago": prev_rank})
    df["Rank_Now"]    = _safe_int_series(df["Rank_Now"])
    df["Rank_4W_Ago"] = _safe_int_series(df["Rank_4W_Ago"])
    both = df["Rank_Now"].notna() & df["Rank_4W_Ago"].notna()
    df["Shift"] = pd.array([pd.NA] * len(df), dtype="Int64")
    if both.any():
        shift_vals = (
            df.loc[both, "Rank_4W_Ago"].astype(float)
            - df.loc[both, "Rank_Now"].astype(float)
        )
        df.loc[both, "Shift"] = _safe_int_series(shift_vals)
    df.sort_values("Rank_Now", inplace=True, na_position="last")
    return df

# ── Master loader (parallel computation) ─────────────────────────────────────

def load_all_data():
    """Fetch once, compute all metrics — returns dict of DataFrames."""
    prices = fetch_prices(period_days=365)

    # Run independent computations in parallel
    with ThreadPoolExecutor(max_workers=6) as ex:
        f_returns   = ex.submit(build_returns_table, prices)
        f_rrg       = ex.submit(compute_rs_ratio_momentum, prices)
        f_norm90    = ex.submit(compute_normalized_series, prices, 90)
        f_norm30    = ex.submit(compute_normalized_series, prices, 30)
        f_norm7     = ex.submit(compute_normalized_series, prices, 7)
        f_rs90      = ex.submit(compute_rolling_rs, prices, 90)
        f_drawdown  = ex.submit(compute_drawdown, prices)
        f_vol       = ex.submit(compute_volatility, prices)
        f_rankshift = ex.submit(compute_rank_shift, prices)
        f_rrg_days  = ex.submit(compute_rrg_days_in_quadrant, prices)

        returns    = f_returns.result()
        rrg        = f_rrg.result()
        norm_90    = f_norm90.result()
        norm_30    = f_norm30.result()
        norm_7     = f_norm7.result()
        rs_90      = f_rs90.result()
        drawdown   = f_drawdown.result()
        volatility = f_vol.result()
        rank_shift = f_rankshift.result()
        rrg_days   = f_rrg_days.result()

    rel_returns = relative_returns(returns)

    return {
        "prices":          prices,
        "returns":         returns,
        "rel_returns":     rel_returns,
        "rrg":             rrg,
        "rrg_days":        rrg_days,
        "norm_90":         norm_90,
        "norm_30":         norm_30,
        "norm_7":          norm_7,
        "rs_90":           rs_90,
        "drawdown":        drawdown,
        "volatility":      volatility,
        "rank_shift":      rank_shift,
        "sector_names":    list(SECTORS.keys()),
        "benchmark_name":  list(BENCHMARK.keys())[0],
        "last_updated":    datetime.now().strftime("%d %b %Y %H:%M IST"),
    }
