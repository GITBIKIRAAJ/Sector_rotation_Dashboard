"""
universe_builder.py
====================
Builds and caches the master NIFTY 2000 universe with:
- Market cap (Cr)
- Cap tier (Micro / Small / Mid / Large / Mega)
- NSE sector (industry classification)
- NIFTY index membership (500 / 1000 / 2000)

Data sources (tried in order, first success wins):
1. NSE bhavcopy ZIP
2. NSE index constituent CSVs
3. yfinance fast_info
4. Hardcoded seed universe (~500 key stocks)

Refresh cadence: once per calendar day (auto-checked)
"""

import os, io, time, zipfile as _zipfile, logging, requests
import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime, timedelta
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
log = logging.getLogger(__name__)

BASE_DIR   = Path(__file__).parent
DATA_DIR   = BASE_DIR / "data"
UNIVERSE_F = DATA_DIR / "master_universe.csv"
DATA_DIR.mkdir(exist_ok=True)

CAP_TIERS = [
    ("Mega",  50_000, float("inf")),
    ("Large", 20_000, 50_000),
    ("Mid",    5_000, 20_000),
    ("Small",  1_000,  5_000),
    ("Micro",      0,  1_000),
]

def assign_tier(mcap_cr: float) -> str:
    if pd.isna(mcap_cr) or mcap_cr <= 0:
        return "Unknown"
    for tier, lo, hi in CAP_TIERS:
        if lo <= mcap_cr < hi:
            return tier
    return "Mega"

NSE_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/122.0.0.0 Safari/537.36"
    ),
    "Accept":          "text/html,application/xhtml+xml,*/*;q=0.8",
    "Accept-Language": "en-US,en;q=0.9",
    "Referer":         "https://www.nseindia.com/",
}

def _nse_session() -> requests.Session:
    s = requests.Session()
    try:
        s.get("https://www.nseindia.com", headers=NSE_HEADERS, timeout=10)
        time.sleep(0.5)
    except Exception:
        pass
    return s

BHAV_URL = (
    "https://nsearchives.nseindia.com/content/cm/"
    "BhavCopy_NSE_CM_0_0_0_{date}_F_0000.csv.zip"
)

def _fetch_bhavcopy(session):
    for delta in range(7):
        dt = datetime.today() - timedelta(days=delta)
        if dt.weekday() >= 5:
            continue
        url = BHAV_URL.format(date=dt.strftime("%Y%m%d"))
        try:
            r = session.get(url, headers=NSE_HEADERS, timeout=20)
            if r.status_code == 200 and len(r.content) > 2000:
                z  = _zipfile.ZipFile(io.BytesIO(r.content))
                df = pd.read_csv(z.open(z.namelist()[0]))
                df.columns = df.columns.str.strip().str.lower().str.replace(" ", "_")
                log.info("Bhavcopy fetched for %s — %d rows", dt.date(), len(df))
                return df, dt.date()
        except Exception as e:
            log.debug("Bhavcopy %s: %s", dt.date(), e)
    return pd.DataFrame(), None

def _parse_bhavcopy(raw):
    col_map = {
        "tckrsymbl": "symbol", "isin": "isin", "srs": "series",
        "lastpric": "close", "tottrdqnt": "volume", "totaltrdval": "turnover",
        "symbol": "symbol", "close": "close", "totaltrdqty": "volume",
    }
    df = raw.rename(columns={k: v for k, v in col_map.items() if k in raw.columns})
    if "series" in df.columns:
        df = df[df["series"].str.strip().str.upper() == "EQ"]
    keep = [c for c in ["symbol","isin","close","volume","turnover"] if c in df.columns]
    df   = df[keep].copy()
    df["symbol"]       = df["symbol"].str.strip().str.upper()
    df["market_cap_cr"] = np.nan
    return df.drop_duplicates("symbol").reset_index(drop=True)

INDEX_CSV_URLS = {
    "NIFTY 50":           "https://www.niftyindices.com/IndexConstituent/ind_nifty50list.csv",
    "NIFTY 100":          "https://www.niftyindices.com/IndexConstituent/ind_nifty100list.csv",
    "NIFTY 200":          "https://www.niftyindices.com/IndexConstituent/ind_nifty200list.csv",
    "NIFTY 500":          "https://www.niftyindices.com/IndexConstituent/ind_nifty500list.csv",
    "NIFTY MIDCAP 150":   "https://www.niftyindices.com/IndexConstituent/ind_niftymidcap150list.csv",
    "NIFTY SMALLCAP 250": "https://www.niftyindices.com/IndexConstituent/ind_niftysmallcap250list.csv",
    "NIFTY MICROCAP 250": "https://www.niftyindices.com/IndexConstituent/ind_niftymicrocap250list.csv",
    "NIFTY AUTO":         "https://www.niftyindices.com/IndexConstituent/ind_niftyautolist.csv",
    "NIFTY BANK":         "https://www.niftyindices.com/IndexConstituent/ind_niftybanklist.csv",
    "NIFTY ENERGY":       "https://www.niftyindices.com/IndexConstituent/ind_niftyenergylist.csv",
    "NIFTY FMCG":         "https://www.niftyindices.com/IndexConstituent/ind_niftyfmcglist.csv",
    "NIFTY FINANCE":      "https://www.niftyindices.com/IndexConstituent/ind_niftyfinancelist.csv",
    "NIFTY IT":           "https://www.niftyindices.com/IndexConstituent/ind_niftyitlist.csv",
    "NIFTY MEDIA":        "https://www.niftyindices.com/IndexConstituent/ind_niftymedialist.csv",
    "NIFTY METAL":        "https://www.niftyindices.com/IndexConstituent/ind_niftymetallist.csv",
    "NIFTY PHARMA":       "https://www.niftyindices.com/IndexConstituent/ind_niftypharmalist.csv",
    "NIFTY PSE":          "https://www.niftyindices.com/IndexConstituent/ind_niftypselist.csv",
    "NIFTY REALTY":       "https://www.niftyindices.com/IndexConstituent/ind_niftyrealtylist.csv",
    "NIFTY PSU BANK":     "https://www.niftyindices.com/IndexConstituent/ind_niftypsubanklist.csv",
    "NIFTY INFRA":        "https://www.niftyindices.com/IndexConstituent/ind_niftyinfralist.csv",
    "NIFTY COMMODITIES":  "https://www.niftyindices.com/IndexConstituent/ind_niftycommoditieslist.csv",
    "NIFTY CONSUMPTION":  "https://www.niftyindices.com/IndexConstituent/ind_niftyconsumptionlist.csv",
    "NIFTY SERVICES":     "https://www.niftyindices.com/IndexConstituent/ind_niftyservicesectorlist.csv",
    "NIFTY OIL & GAS":    "https://www.niftyindices.com/IndexConstituent/ind_niftyoilgaslist.csv",
    "NIFTY MNC":          "https://www.niftyindices.com/IndexConstituent/ind_niftymnclist.csv",
    "NIFTY CPSE":         "https://www.niftyindices.com/IndexConstituent/ind_niftycpselist.csv",
}

BROAD_INDICES  = ["NIFTY 50","NIFTY 100","NIFTY 200","NIFTY 500",
                  "NIFTY MIDCAP 150","NIFTY SMALLCAP 250","NIFTY MICROCAP 250"]
SECTOR_INDICES = [k for k in INDEX_CSV_URLS if k not in BROAD_INDICES]

def _fetch_index_constituents(session):
    result = {}
    for idx, url in INDEX_CSV_URLS.items():
        try:
            r = session.get(url, headers=NSE_HEADERS, timeout=15)
            if r.status_code == 200 and len(r.content) > 100:
                df      = pd.read_csv(io.StringIO(r.text))
                sym_col = next((c for c in df.columns if "symbol" in c.lower()), None)
                if sym_col:
                    symbols    = df[sym_col].dropna().str.strip().str.upper().tolist()
                    result[idx] = [s for s in symbols if s]
                    log.info("  %s: %d stocks", idx, len(symbols))
            time.sleep(0.2)
        except Exception as e:
            log.debug("  %s failed: %s", idx, e)
    return result

def _fetch_mcap_yfinance(symbols, batch_size=50, progress_cb=None):
    mcap    = {}
    tickers = [s if s.endswith(".NS") else s + ".NS" for s in symbols]
    for i in range(0, len(tickers), batch_size):
        batch = tickers[i:i+batch_size]
        if progress_cb:
            progress_cb(i, len(tickers))
        for tkr in batch:
            sym = tkr.replace(".NS","")
            try:
                fi = yf.Ticker(tkr).fast_info
                mc = getattr(fi, "market_cap", None)
                if mc and mc > 0:
                    mcap[sym] = round(mc / 1e7, 2)
            except Exception:
                pass
        time.sleep(0.3)
    return mcap

# ── Hardcoded seed universe ───────────────────────────────────────────────────
SEED_UNIVERSE = {
    "RELIANCE":   (1900000,"Oil & Gas","Mega"),       "TCS":        (1300000,"Information Technology","Mega"),
    "HDFCBANK":   (1200000,"Banking","Mega"),          "BHARTIARTL": (900000,"Telecom","Mega"),
    "ICICIBANK":  (870000,"Banking","Mega"),            "INFOSYS":    (620000,"Information Technology","Mega"),
    "SBIN":       (750000,"Banking","Mega"),            "HINDUNILVR": (560000,"FMCG","Mega"),
    "ITC":        (530000,"FMCG","Mega"),               "LT":         (490000,"Capital Goods","Mega"),
    "KOTAKBANK":  (370000,"Banking","Mega"),            "AXISBANK":   (350000,"Banking","Mega"),
    "BAJFINANCE": (420000,"Financial Services","Mega"), "MARUTI":     (380000,"Automobile","Mega"),
    "SUNPHARMA":  (380000,"Pharma","Mega"),             "NTPC":       (330000,"Power","Mega"),
    "POWERGRID":  (290000,"Power","Mega"),              "WIPRO":      (250000,"Information Technology","Mega"),
    "HCLTECH":    (390000,"Information Technology","Mega"), "ONGC":   (320000,"Oil & Gas","Mega"),
    "TATAMOTORS": (280000,"Automobile","Mega"),         "ADANIPORTS": (260000,"Infrastructure","Mega"),
    "TATASTEEL":  (200000,"Metals","Mega"),             "M&M":        (270000,"Automobile","Mega"),
    "NESTLEIND":  (220000,"FMCG","Mega"),               "TECHM":      (130000,"Information Technology","Large"),
    "DRREDDY":    (120000,"Pharma","Large"),            "BAJAJ-AUTO": (200000,"Automobile","Mega"),
    "HEROMOTOCO": (90000,"Automobile","Large"),         "EICHERMOT":  (110000,"Automobile","Large"),
    "TVSMOTOR":   (100000,"Automobile","Large"),        "ASHOKLEY":   (45000,"Automobile","Large"),
    "CIPLA":      (120000,"Pharma","Large"),            "DIVISLAB":   (75000,"Pharma","Large"),
    "AUROPHARMA": (55000,"Pharma","Large"),             "TORNTPHARM": (80000,"Pharma","Large"),
    "ALKEM":      (35000,"Pharma","Large"),             "INDUSINDBK": (100000,"Banking","Large"),
    "FEDERALBNK": (35000,"Banking","Mid"),              "IDFCFIRSTB": (38000,"Banking","Large"),
    "BANDHANBNK": (25000,"Banking","Mid"),              "AUBANK":     (43000,"Banking","Large"),
    "BRITANNIA":  (115000,"FMCG","Large"),              "DABUR":      (85000,"FMCG","Large"),
    "MARICO":     (65000,"FMCG","Large"),               "GODREJCP":   (72000,"FMCG","Large"),
    "TATACONSUM": (80000,"FMCG","Large"),               "COLPAL":     (55000,"FMCG","Large"),
    "EMAMILTD":   (28000,"FMCG","Mid"),                "HINDPETRO":  (45000,"Oil & Gas","Large"),
    "BPCL":       (120000,"Oil & Gas","Large"),         "IOC":        (180000,"Oil & Gas","Mega"),
    "GAIL":       (120000,"Oil & Gas","Large"),         "JSWSTEEL":   (220000,"Metals","Mega"),
    "HINDALCO":   (145000,"Metals","Large"),            "VEDL":       (130000,"Metals","Large"),
    "COALINDIA":  (280000,"Metals","Mega"),             "NMDC":       (60000,"Metals","Large"),
    "SAIL":       (35000,"Metals","Large"),             "HINDCOPPER": (22000,"Metals","Mid"),
    "NATIONALUM": (18000,"Metals","Mid"),               "APLAPOLLO":  (40000,"Metals","Large"),
    "DLF":        (160000,"Realty","Large"),            "GODREJPROP": (75000,"Realty","Large"),
    "OBEROIRLTY": (60000,"Realty","Large"),             "PHOENIXLTD": (55000,"Realty","Large"),
    "PRESTIGE":   (65000,"Realty","Large"),             "BRIGADE":    (25000,"Realty","Mid"),
    "SOBHA":      (20000,"Realty","Mid"),               "SIEMENS":    (160000,"Capital Goods","Large"),
    "ABB":        (130000,"Capital Goods","Large"),     "HAVELLS":    (100000,"Capital Goods","Large"),
    "CUMMINSIND": (60000,"Capital Goods","Large"),      "BAJAJFINSV": (245000,"Financial Services","Mega"),
    "SBILIFE":    (150000,"Financial Services","Large"),"HDFCLIFE":   (130000,"Financial Services","Large"),
    "ICICIGI":    (100000,"Financial Services","Large"),"MUTHOOTFIN": (50000,"Financial Services","Large"),
    "CHOLAFIN":   (80000,"Financial Services","Large"), "M&MFIN":     (32000,"Financial Services","Mid"),
    "LICHSGFIN":  (22000,"Financial Services","Mid"),   "LTIM":       (130000,"Information Technology","Large"),
    "MPHASIS":    (45000,"Information Technology","Large"), "COFORGE": (38000,"Information Technology","Large"),
    "PERSISTENT": (50000,"Information Technology","Large"), "LTTS":   (42000,"Information Technology","Large"),
    "INFY":       (620000,"Information Technology","Mega"), "TATAPOWER":(90000,"Power","Large"),
    "ADANIGREEN": (190000,"Power","Mega"),              "ADANIENT":   (240000,"Infrastructure","Mega"),
    "IPCALAB":    (18000,"Pharma","Mid"),               "BIOCON":     (22000,"Pharma","Mid"),
    "GLENMARK":   (18000,"Pharma","Mid"),               "LALPATHLAB": (20000,"Healthcare","Mid"),
    "METROPOLIS": (8000,"Healthcare","Mid"),            "YESBANK":    (20000,"Banking","Mid"),
    "RBLBANK":    (12000,"Banking","Mid"),              "KTKBANK":    (14000,"Banking","Mid"),
    "DCBBANK":    (5000,"Banking","Mid"),               "DLF":        (160000,"Realty","Large"),
    "BALKRISIND": (55000,"Automobile","Large"),         "MRF":        (44000,"Automobile","Large"),
    "APOLLOTYRE": (18000,"Automobile","Mid"),           "CEATLTD":    (10000,"Automobile","Mid"),
    "EXIDEIND":   (16000,"Automobile","Mid"),           "AMARAJABAT": (8000,"Automobile","Mid"),
    "DEEPAKNTR":  (30000,"Chemicals","Mid"),            "PIDILITIND": (130000,"Chemicals","Large"),
    "ASIANPAINT": (290000,"Chemicals","Mega"),          "BERGERPAINTS":(65000,"Chemicals","Large"),
    "KANSAINER":  (25000,"Chemicals","Mid"),            "NAVINFLUOR": (18000,"Chemicals","Mid"),
    "FINEORG":    (8000,"Chemicals","Mid"),             "AARTIIND":   (12000,"Chemicals","Mid"),
    "SRF":        (70000,"Chemicals","Large"),          "TATACHEM":   (30000,"Chemicals","Mid"),
    "JUBLFOOD":   (55000,"Consumer","Large"),           "ZOMATO":     (200000,"Consumer","Mega"),
    "NYKAA":      (38000,"Consumer","Large"),           "IRFC":       (180000,"Financial Services","Mega"),
    "RVNL":       (60000,"Infrastructure","Large"),     "IRCTC":      (70000,"Consumer","Large"),
    "CONCOR":     (50000,"Infrastructure","Large"),     "CDSL":       (22000,"Financial Services","Mid"),
    "MCX":        (18000,"Financial Services","Mid"),   "BSE":        (40000,"Financial Services","Large"),
    "DIXON":      (38000,"Capital Goods","Large"),      "POLYCAB":    (75000,"Capital Goods","Large"),
    "KECINT":     (18000,"Capital Goods","Mid"),        "NHPC":       (85000,"Power","Large"),
    "SJVN":       (35000,"Power","Large"),              "NAUKRI":     (95000,"Information Technology","Large"),
    "TATAELXSI":  (42000,"Information Technology","Large"), "KPITTECH":(30000,"Information Technology","Mid"),
    "INDIGOPNTS": (4000,"Chemicals","Small"),           "ALKYLAMINE": (4500,"Chemicals","Small"),
    "MANAPPURAM": (18000,"Financial Services","Mid"),   "SUNDARMFIN": (45000,"Financial Services","Large"),
    "MAXHEALTH":  (28000,"Healthcare","Mid"),           "FORTIS":     (28000,"Healthcare","Mid"),
    "NBCC":       (22464,"CPSE","Large"),               "RAILTEL":    (12000,"Infrastructure","Mid"),
    "IRCON":      (20000,"Infrastructure","Mid"),       "SONACOMS":   (25000,"Automobile","Large"),
}

def _build_from_seed():
    records = []
    for sym, (mcap, sector, tier) in SEED_UNIVERSE.items():
        records.append({
            "symbol": sym, "company_name": sym,
            "nse_sector": sector, "market_cap_cr": mcap,
            "cap_tier": tier, "nifty_index": _infer_nifty_index(mcap),
            "ticker": sym + ".NS", "isin": "", "sector_indices": "",
        })
    return pd.DataFrame(records)

def _infer_nifty_index(mcap_cr):
    if pd.isna(mcap_cr): return "NIFTY 2000"
    if mcap_cr >= 50000: return "NIFTY 50"
    if mcap_cr >= 10000: return "NIFTY 500"
    if mcap_cr >= 3000:  return "NIFTY 1000"
    return "NIFTY 2000"

def _assign_index_membership(symbol, index_members):
    priority = ["NIFTY 50","NIFTY 100","NIFTY 200","NIFTY 500",
                "NIFTY MIDCAP 150","NIFTY SMALLCAP 250","NIFTY MICROCAP 250"]
    for idx in priority:
        if symbol in index_members.get(idx, []):
            return idx
    return "NIFTY 2000"

def _assign_sector_indices(symbol, index_members):
    return [idx for idx in SECTOR_INDICES if symbol in index_members.get(idx, [])]

def _enrich_mcap(df, progress_cb=None):
    need = df[df["market_cap_cr"].isna()]["symbol"].tolist()
    if not need:
        return df
    log.info("Fetching market cap for %d stocks via yfinance...", len(need))
    for i in range(0, len(need), 100):
        syms = need[i:i+100]
        if progress_cb:
            progress_cb(i, len(need), "Fetching market caps")
        for sym in syms:
            try:
                fi = yf.Ticker(sym + ".NS").fast_info
                mc = getattr(fi, "market_cap", None)
                if mc and mc > 0:
                    df.loc[df["symbol"] == sym, "market_cap_cr"] = round(mc/1e7, 2)
            except Exception:
                pass
        time.sleep(0.5)
    df["cap_tier"] = df["market_cap_cr"].apply(assign_tier)
    return df

def build_universe(force_refresh=False, progress_cb=None):
    """Main entry — returns master universe DataFrame, cached daily."""
    if UNIVERSE_F.exists() and not force_refresh:
        mtime     = datetime.fromtimestamp(UNIVERSE_F.stat().st_mtime)
        age_hours = (datetime.now() - mtime).total_seconds() / 3600
        if age_hours < 20:
            df = pd.read_csv(UNIVERSE_F)
            log.info("Loaded cached universe: %d stocks (%.1fh old)", len(df), age_hours)
            return df

    log.info("Building fresh universe...")
    session = _nse_session()

    if progress_cb: progress_cb(1, 5, "Fetching NSE index lists")
    index_members = {}
    try:
        index_members = _fetch_index_constituents(session)
    except Exception as e:
        log.warning("Index constituent fetch failed: %s", e)

    broad_syms = set()
    for idx in BROAD_INDICES:
        broad_syms.update(index_members.get(idx, []))
    broad_syms.update(SEED_UNIVERSE.keys())
    all_symbols = sorted(broad_syms)
    log.info("Total unique symbols: %d", len(all_symbols))

    if progress_cb: progress_cb(2, 5, "Fetching NSE bhavcopy")
    bhav_df = pd.DataFrame()
    try:
        raw_bhav, _ = _fetch_bhavcopy(session)
        if not raw_bhav.empty:
            bhav_df = _parse_bhavcopy(raw_bhav)
    except Exception as e:
        log.warning("Bhavcopy failed: %s", e)

    if progress_cb: progress_cb(3, 5, "Building master universe table")
    records = []
    for sym in all_symbols:
        rec = {"symbol": sym, "ticker": sym + ".NS", "isin": ""}
        if not bhav_df.empty and sym in bhav_df["symbol"].values:
            row        = bhav_df[bhav_df["symbol"] == sym].iloc[0]
            rec["isin"] = row.get("isin","")
        if sym in SEED_UNIVERSE:
            seed               = SEED_UNIVERSE[sym]
            rec["market_cap_cr"] = seed[0]
            rec["nse_sector"]    = seed[1]
            rec["cap_tier"]      = seed[2]
        else:
            rec["market_cap_cr"] = np.nan
            rec["nse_sector"]    = "Others"
            rec["cap_tier"]      = "Unknown"
        rec["nifty_index"]    = _assign_index_membership(sym, index_members)
        sec_idxs              = _assign_sector_indices(sym, index_members)
        rec["sector_indices"] = "|".join(sec_idxs) if sec_idxs else ""
        rec["company_name"]   = sym
        records.append(rec)

    df = pd.DataFrame(records)

    SECTOR_INDEX_TO_SECTOR = {
        "NIFTY AUTO":"Automobile","NIFTY BANK":"Banking","NIFTY ENERGY":"Energy",
        "NIFTY FMCG":"FMCG","NIFTY FINANCE":"Financial Services","NIFTY INFRA":"Infrastructure",
        "NIFTY IT":"Information Technology","NIFTY MEDIA":"Media","NIFTY METAL":"Metals",
        "NIFTY MNC":"MNC","NIFTY PHARMA":"Pharma","NIFTY PSE":"PSE","NIFTY REALTY":"Realty",
        "NIFTY CPSE":"CPSE","NIFTY PSU BANK":"PSU Banking","NIFTY COMMODITIES":"Commodities",
        "NIFTY CONSUMPTION":"Consumption","NIFTY SERVICES":"Services","NIFTY OIL & GAS":"Oil & Gas",
    }

    def _get_sector_from_indices(sec_idx_str):
        if not sec_idx_str: return None
        for idx in sec_idx_str.split("|"):
            sec = SECTOR_INDEX_TO_SECTOR.get(idx.strip())
            if sec: return sec
        return None

    mask = df["nse_sector"] == "Others"
    df.loc[mask, "nse_sector"] = (
        df.loc[mask, "sector_indices"].apply(_get_sector_from_indices).fillna("Others")
    )

    if progress_cb: progress_cb(4, 5, "Fetching market caps from yfinance")
    missing_count = df["market_cap_cr"].isna().sum()
    if 0 < missing_count < 2000:
        df = _enrich_mcap(df, progress_cb)
    elif missing_count >= 2000:
        top_syms = all_symbols[:500]
        df = _enrich_mcap(df[df["symbol"].isin(top_syms)].copy(), progress_cb)

    df["cap_tier"]    = df["market_cap_cr"].apply(assign_tier)
    df["nifty_index"] = df.apply(
        lambda r: r["nifty_index"] if r["nifty_index"] != "NIFTY 2000"
                  else _infer_nifty_index(r.get("market_cap_cr", np.nan)),
        axis=1
    )

    if progress_cb: progress_cb(5, 5, "Saving universe")
    df.sort_values("market_cap_cr", ascending=False, inplace=True)
    df.reset_index(drop=True, inplace=True)
    df["rank_by_mcap"] = range(1, len(df) + 1)
    df.to_csv(UNIVERSE_F, index=False)
    log.info("Universe saved: %d stocks -> %s", len(df), UNIVERSE_F)
    return df

def filter_universe(df, cap_tiers=None, mcap_min_cr=0, mcap_max_cr=float("inf"),
                    sectors=None, nifty_index=None, min_mcap_rank=None, max_mcap_rank=None):
    out = df.copy()
    if cap_tiers:
        out = out[out["cap_tier"].isin(cap_tiers)]
    if mcap_min_cr > 0:
        out = out[out["market_cap_cr"].fillna(0) >= mcap_min_cr]
    if mcap_max_cr < float("inf"):
        out = out[out["market_cap_cr"].fillna(0) <= mcap_max_cr]
    if sectors:
        out = out[out["nse_sector"].isin(sectors)]
    if nifty_index:
        priority = ["NIFTY 50","NIFTY 100","NIFTY 200","NIFTY 500",
                    "NIFTY MIDCAP 150","NIFTY SMALLCAP 250",
                    "NIFTY MICROCAP 250","NIFTY 1000","NIFTY 2000"]
        idx_pos  = priority.index(nifty_index) if nifty_index in priority else len(priority)
        allowed  = set(priority[:idx_pos+1])
        out      = out[out["nifty_index"].isin(allowed)]
    if min_mcap_rank:
        out = out[out["rank_by_mcap"] >= min_mcap_rank]
    if max_mcap_rank:
        out = out[out["rank_by_mcap"] <= max_mcap_rank]
    return out.reset_index(drop=True)

def get_sectors(df):
    return sorted(df["nse_sector"].dropna().unique().tolist())

def get_tickers(df):
    return df["ticker"].dropna().tolist()

if __name__ == "__main__":
    import sys
    df = build_universe(force_refresh="--force" in sys.argv)
    print(f"\nUniverse built: {len(df)} stocks")
    print(df[["symbol","nse_sector","cap_tier","market_cap_cr","nifty_index"]].head(20).to_string())
