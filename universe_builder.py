"""
universe_builder.py  (v4 — NIFTY 1000 support, correct tickers, custom cap tiers)
===================================================================================
Builds and caches master universe with 1000+ stocks.
Cap tiers: Mega >1,00,000 | Large 30k-1L | Mid 5k-30k | Small 1k-5k | Micro <1k
"""

import io, time, logging, requests
import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
log = logging.getLogger(__name__)

BASE_DIR   = Path(__file__).parent
DATA_DIR   = BASE_DIR / "data"
UNIVERSE_F = DATA_DIR / "master_universe.csv"
DATA_DIR.mkdir(exist_ok=True)

# ── Cap tiers (INR Crore) ─────────────────────────────────────────────────────
CAP_TIERS = [
    ("Mega",  100_000, float("inf")),   # > 1,00,000 Cr
    ("Large",  30_000, 100_000),         # 30,000 to 1,00,000 Cr
    ("Mid",     5_000,  30_000),         # 5,000 to 30,000 Cr
    ("Small",   1_000,   5_000),         # 1,000 to 5,000 Cr
    ("Micro",       0,   1_000),         # below 1,000 Cr
]

def assign_tier(mcap_cr):
    if pd.isna(mcap_cr) or mcap_cr <= 0:
        return "Unknown"
    for tier, lo, hi in CAP_TIERS:
        if lo <= mcap_cr < hi:
            return tier
    return "Mega"

def _infer_nifty_index(mcap_cr):
    if pd.isna(mcap_cr):   return "NIFTY 2000"
    if mcap_cr >= 50_000:  return "NIFTY 50"
    if mcap_cr >= 15_000:  return "NIFTY 100"
    if mcap_cr >= 6_000:   return "NIFTY 200"
    if mcap_cr >= 2_000:   return "NIFTY 500"
    if mcap_cr >= 600:     return "NIFTY 1000"
    return "NIFTY 2000"

# ── NSE session ───────────────────────────────────────────────────────────────
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

INDEX_CSV_URLS = {
    "NIFTY 50":           "https://www.niftyindices.com/IndexConstituent/ind_nifty50list.csv",
    "NIFTY 100":          "https://www.niftyindices.com/IndexConstituent/ind_nifty100list.csv",
    "NIFTY 200":          "https://www.niftyindices.com/IndexConstituent/ind_nifty200list.csv",
    "NIFTY 500":          "https://www.niftyindices.com/IndexConstituent/ind_nifty500list.csv",
    "NIFTY 1000":         "https://www.niftyindices.com/IndexConstituent/ind_nifty1000list.csv",
    "NIFTY MIDCAP 150":   "https://www.niftyindices.com/IndexConstituent/ind_niftymidcap150list.csv",
    "NIFTY SMALLCAP 250": "https://www.niftyindices.com/IndexConstituent/ind_niftysmallcap250list.csv",
    "NIFTY MICROCAP 250": "https://www.niftyindices.com/IndexConstituent/ind_niftymicrocap250list.csv",
    "NIFTY AUTO":         "https://www.niftyindices.com/IndexConstituent/ind_niftyautolist.csv",
    "NIFTY BANK":         "https://www.niftyindices.com/IndexConstituent/ind_niftybanklist.csv",
    "NIFTY ENERGY":       "https://www.niftyindices.com/IndexConstituent/ind_niftyenergylist.csv",
    "NIFTY FMCG":         "https://www.niftyindices.com/IndexConstituent/ind_niftyfmcglist.csv",
    "NIFTY FINANCE":      "https://www.niftyindices.com/IndexConstituent/ind_niftyfinancelist.csv",
    "NIFTY INFRA":        "https://www.niftyindices.com/IndexConstituent/ind_niftyinfralist.csv",
    "NIFTY IT":           "https://www.niftyindices.com/IndexConstituent/ind_niftyitlist.csv",
    "NIFTY MEDIA":        "https://www.niftyindices.com/IndexConstituent/ind_niftymedialist.csv",
    "NIFTY METAL":        "https://www.niftyindices.com/IndexConstituent/ind_niftymetallist.csv",
    "NIFTY PHARMA":       "https://www.niftyindices.com/IndexConstituent/ind_niftypharmalist.csv",
    "NIFTY PSE":          "https://www.niftyindices.com/IndexConstituent/ind_niftypselist.csv",
    "NIFTY REALTY":       "https://www.niftyindices.com/IndexConstituent/ind_niftyrealtylist.csv",
    "NIFTY PSU BANK":     "https://www.niftyindices.com/IndexConstituent/ind_niftypsubanklist.csv",
    "NIFTY COMMODITIES":  "https://www.niftyindices.com/IndexConstituent/ind_niftycommoditieslist.csv",
    "NIFTY CONSUMPTION":  "https://www.niftyindices.com/IndexConstituent/ind_niftyconsumptionlist.csv",
    "NIFTY SERVICES":     "https://www.niftyindices.com/IndexConstituent/ind_niftyservicesectorlist.csv",
    "NIFTY OIL & GAS":    "https://www.niftyindices.com/IndexConstituent/ind_niftyoilgaslist.csv",
    "NIFTY MNC":          "https://www.niftyindices.com/IndexConstituent/ind_niftymnclist.csv",
    "NIFTY CPSE":         "https://www.niftyindices.com/IndexConstituent/ind_niftycpselist.csv",
}

BROAD_INDICES = [
    "NIFTY 50", "NIFTY 100", "NIFTY 200", "NIFTY 500", "NIFTY 1000",
    "NIFTY MIDCAP 150", "NIFTY SMALLCAP 250", "NIFTY MICROCAP 250",
]
SECTOR_INDICES = [k for k in INDEX_CSV_URLS if k not in BROAD_INDICES]

BROAD_PRIORITY = [
    "NIFTY 50", "NIFTY 100", "NIFTY 200", "NIFTY 500", "NIFTY 1000",
    "NIFTY MIDCAP 150", "NIFTY SMALLCAP 250", "NIFTY MICROCAP 250", "NIFTY 2000",
]

SECTOR_INDEX_TO_SECTOR = {
    "NIFTY AUTO":        "Automobile",
    "NIFTY BANK":        "Banking",
    "NIFTY ENERGY":      "Energy",
    "NIFTY FMCG":        "FMCG",
    "NIFTY FINANCE":     "Financial Services",
    "NIFTY INFRA":       "Infrastructure",
    "NIFTY IT":          "Information Technology",
    "NIFTY MEDIA":       "Media",
    "NIFTY METAL":       "Metals",
    "NIFTY MNC":         "MNC",
    "NIFTY PHARMA":      "Pharma",
    "NIFTY PSE":         "PSE",
    "NIFTY REALTY":      "Realty",
    "NIFTY CPSE":        "CPSE",
    "NIFTY PSU BANK":    "PSU Banking",
    "NIFTY COMMODITIES": "Commodities",
    "NIFTY CONSUMPTION": "Consumption",
    "NIFTY SERVICES":    "Services",
    "NIFTY OIL & GAS":   "Oil & Gas",
}

# ── SEED UNIVERSE — ~340 verified .NS tickers ────────────────────────────────
SEED_UNIVERSE = {
    "RELIANCE":    (1900000,"Oil & Gas","NIFTY 50"),
    "TCS":         (1300000,"Information Technology","NIFTY 50"),
    "HDFCBANK":    (1200000,"Banking","NIFTY 50"),
    "BHARTIARTL":  (900000,"Telecom","NIFTY 50"),
    "ICICIBANK":   (870000,"Banking","NIFTY 50"),
    "SBIN":        (750000,"Banking","NIFTY 50"),
    "INFOSYS":     (620000,"Information Technology","NIFTY 50"),
    "HINDUNILVR":  (560000,"FMCG","NIFTY 50"),
    "ITC":         (530000,"FMCG","NIFTY 50"),
    "LT":          (490000,"Capital Goods","NIFTY 50"),
    "BAJFINANCE":  (420000,"Financial Services","NIFTY 50"),
    "KOTAKBANK":   (370000,"Banking","NIFTY 50"),
    "AXISBANK":    (350000,"Banking","NIFTY 50"),
    "DMART":       (350000,"Consumer","NIFTY 50"),
    "BAJAJ-AUTO":  (360000,"Automobile","NIFTY 50"),
    "MARUTI":      (380000,"Automobile","NIFTY 50"),
    "SUNPHARMA":   (380000,"Pharma","NIFTY 50"),
    "HCLTECH":     (390000,"Information Technology","NIFTY 50"),
    "NTPC":        (330000,"Power","NIFTY 50"),
    "ADANIGREEN":  (310000,"Power","NIFTY 50"),
    "POWERGRID":   (290000,"Power","NIFTY 50"),
    "COALINDIA":   (280000,"Metals","NIFTY 50"),
    "TATAMOTORS":  (280000,"Automobile","NIFTY 50"),
    "M&M":         (270000,"Automobile","NIFTY 50"),
    "ADANIENT":    (260000,"Infrastructure","NIFTY 50"),
    "ADANIPORTS":  (260000,"Infrastructure","NIFTY 50"),
    "WIPRO":       (250000,"Information Technology","NIFTY 50"),
    "ULTRACEMCO":  (250000,"Capital Goods","NIFTY 50"),
    "BAJAJFINSV":  (245000,"Financial Services","NIFTY 50"),
    "NESTLEIND":   (220000,"FMCG","NIFTY 50"),
    "JSWSTEEL":    (220000,"Metals","NIFTY 50"),
    "ASIANPAINT":  (200000,"Chemicals","NIFTY 50"),
    "TATASTEEL":   (200000,"Metals","NIFTY 50"),
    "ZOMATO":      (200000,"Consumer","NIFTY 50"),
    "HAL":         (180000,"Capital Goods","NIFTY 50"),
    "ADANIPOWER":  (180000,"Power","NIFTY 50"),
    "IRFC":        (180000,"Financial Services","NIFTY 50"),
    "IOC":         (180000,"Oil & Gas","NIFTY 50"),
    "HINDALCO":    (145000,"Metals","NIFTY 50"),
    "TECHM":       (130000,"Information Technology","NIFTY 50"),
    "BEL":         (130000,"Capital Goods","NIFTY 50"),
    "LICI":        (485000,"Financial Services","NIFTY 100"),
    "SIEMENS":     (160000,"Capital Goods","NIFTY 100"),
    "DLF":         (160000,"Realty","NIFTY 100"),
    "SBILIFE":     (150000,"Financial Services","NIFTY 100"),
    "HDFCLIFE":    (130000,"Financial Services","NIFTY 100"),
    "PIDILITIND":  (130000,"Chemicals","NIFTY 100"),
    "ABB":         (130000,"Capital Goods","NIFTY 100"),
    "LTIM":        (130000,"Information Technology","NIFTY 100"),
    "TATAPOWER":   (120000,"Power","NIFTY 100"),
    "BPCL":        (120000,"Oil & Gas","NIFTY 100"),
    "GAIL":        (120000,"Oil & Gas","NIFTY 100"),
    "DRREDDY":     (120000,"Pharma","NIFTY 100"),
    "CIPLA":       (120000,"Pharma","NIFTY 100"),
    "RECLTD":      (120000,"Financial Services","NIFTY 100"),
    "PFC":         (120000,"Financial Services","NIFTY 100"),
    "AMBUJACEM":   (120000,"Capital Goods","NIFTY 100"),
    "BRITANNIA":   (115000,"FMCG","NIFTY 100"),
    "EICHERMOT":   (110000,"Automobile","NIFTY 100"),
    "HAVELLS":     (100000,"Capital Goods","NIFTY 100"),
    "TVSMOTOR":    (100000,"Automobile","NIFTY 100"),
    "INDUSINDBK":  (100000,"Banking","NIFTY 100"),
    "ICICIPRULI":  (80000,"Financial Services","NIFTY 100"),
    "SHRIRAMFIN":  (90000,"Financial Services","NIFTY 100"),
    "BAJAJHFL":    (80000,"Financial Services","NIFTY 100"),
    "APOLLOHOSP":  (80000,"Healthcare","NIFTY 100"),
    "TATACONSUM":  (80000,"FMCG","NIFTY 100"),
    "SHREECEM":    (80000,"Capital Goods","NIFTY 100"),
    "TORNTPHARM":  (80000,"Pharma","NIFTY 100"),
    "PNB":         (80000,"Banking","NIFTY 100"),
    "CHOLAFIN":    (80000,"Financial Services","NIFTY 100"),
    "BANKBARODA":  (90000,"Banking","NIFTY 100"),
    "HEROMOTOCO":  (90000,"Automobile","NIFTY 100"),
    "BOSCHLTD":    (90000,"Automobile","NIFTY 100"),
    "SWIGGY":      (95000,"Consumer","NIFTY 100"),
    "NAUKRI":      (95000,"Information Technology","NIFTY 100"),
    "LUPIN":       (90000,"Pharma","NIFTY 100"),
    "VBL":         (90000,"Consumer","NIFTY 100"),
    "NHPC":        (85000,"Power","NIFTY 100"),
    "DABUR":       (85000,"FMCG","NIFTY 100"),
    "MACROTECH":   (90000,"Realty","NIFTY 100"),
    "TRENT":       (90000,"Consumer","NIFTY 100"),
    "HDFCAMC":     (70000,"Financial Services","NIFTY 100"),
    "POLYCAB":     (75000,"Capital Goods","NIFTY 100"),
    "DIXON":       (80000,"Capital Goods","NIFTY 100"),
    "SUZLON":      (55000,"Power","NIFTY 100"),
    "MARICO":      (65000,"FMCG","NIFTY 100"),
    "GODREJPROP":  (75000,"Realty","NIFTY 200"),
    "DIVISLAB":    (75000,"Pharma","NIFTY 200"),
    "SRF":         (70000,"Chemicals","NIFTY 200"),
    "ZYDUSLIFE":   (70000,"Pharma","NIFTY 200"),
    "OFSS":        (70000,"Information Technology","NIFTY 200"),
    "CANBK":       (70000,"Banking","NIFTY 200"),
    "BHEL":        (70000,"Capital Goods","NIFTY 200"),
    "KPITTECH":    (60000,"Information Technology","NIFTY 200"),
    "NMDC":        (60000,"Metals","NIFTY 200"),
    "RVNL":        (60000,"Infrastructure","NIFTY 200"),
    "OBEROIRLTY":  (60000,"Realty","NIFTY 200"),
    "JSWENERGY":   (60000,"Power","NIFTY 200"),
    "GODREJCP":    (72000,"FMCG","NIFTY 200"),
    "PRESTIGE":    (65000,"Realty","NIFTY 200"),
    "BERGEPAINT":  (65000,"Chemicals","NIFTY 200"),
    "INDIANB":     (60000,"Banking","NIFTY 200"),
    "COLPAL":      (55000,"FMCG","NIFTY 200"),
    "BALKRISIND":  (55000,"Automobile","NIFTY 200"),
    "MOTHERSON":   (55000,"Automobile","NIFTY 200"),
    "APLAPOLLO":   (40000,"Metals","NIFTY 200"),
    "CONCOR":      (50000,"Infrastructure","NIFTY 200"),
    "MUTHOOTFIN":  (50000,"Financial Services","NIFTY 200"),
    "UNIONBI":     (50000,"Banking","NIFTY 200"),
    "TATAELXSI":   (42000,"Information Technology","NIFTY 200"),
    "PERSISTENT":  (50000,"Information Technology","NIFTY 200"),
    "LTTS":        (42000,"Information Technology","NIFTY 200"),
    "MPHASIS":     (45000,"Information Technology","NIFTY 200"),
    "HEXAWARE":    (40000,"Information Technology","NIFTY 200"),
    "HINDPETRO":   (45000,"Oil & Gas","NIFTY 200"),
    "PIIND":       (45000,"Chemicals","NIFTY 200"),
    "TORNTPOWER":  (45000,"Power","NIFTY 200"),
    "SUNDARMFIN":  (45000,"Financial Services","NIFTY 200"),
    "MCDOWELLN":   (45000,"Consumer","NIFTY 200"),
    "TATACOMM":    (45000,"Telecom","NIFTY 200"),
    "ASTRAL":      (40000,"Capital Goods","NIFTY 200"),
    "MOTILALOFS":  (40000,"Financial Services","NIFTY 200"),
    "COFORGE":     (38000,"Information Technology","NIFTY 200"),
    "NYKAA":       (38000,"Consumer","NIFTY 200"),
    "IDFCFIRSTB":  (38000,"Banking","NIFTY 200"),
    "AUBANK":      (43000,"Banking","NIFTY 200"),
    "ABBOTINDIA":  (35000,"Pharma","NIFTY 200"),
    "ALKEM":       (35000,"Pharma","NIFTY 200"),
    "SAIL":        (35000,"Metals","NIFTY 200"),
    "TATAINVEST":  (35000,"Financial Services","NIFTY 200"),
    "SJVN":        (35000,"Power","NIFTY 200"),
    "PAGEIND":     (35000,"Consumer","NIFTY 200"),
    "GMRINFRA":    (35000,"Infrastructure","NIFTY 200"),
    "ACCLTD":      (35000,"Capital Goods","NIFTY 200"),
    "FEDERALBNK":  (35000,"Banking","NIFTY 200"),
    "MRF":         (44000,"Automobile","NIFTY 200"),
    "BHARATFORG":  (35000,"Automobile","NIFTY 200"),
    "ASHOKLEY":    (45000,"Automobile","NIFTY 200"),
    "BSE":         (40000,"Financial Services","NIFTY 200"),
    "NLCINDIA":    (40000,"Power","NIFTY 200"),
    "JSWINFRA":    (40000,"Infrastructure","NIFTY 200"),
    "M&MFIN":      (32000,"Financial Services","NIFTY 200"),
    "CUMMINSIND":  (60000,"Capital Goods","NIFTY 200"),
    "PHOENIXLTD":  (55000,"Realty","NIFTY 500"),
    "BRIGADE":     (25000,"Realty","NIFTY 500"),
    "SOBHA":       (20000,"Realty","NIFTY 500"),
    "MAHLIFE":     (12000,"Realty","NIFTY 500"),
    "IBREALEST":   (8000,"Realty","NIFTY 500"),
    "BANDHANBNK":  (25000,"Banking","NIFTY 500"),
    "YESBANK":     (20000,"Banking","NIFTY 500"),
    "RBLBANK":     (12000,"Banking","NIFTY 500"),
    "KTKBANK":     (14000,"Banking","NIFTY 500"),
    "UCOBANK":     (30000,"Banking","NIFTY 500"),
    "MAHABANK":    (18000,"Banking","NIFTY 500"),
    "CENTRALBK":   (20000,"Banking","NIFTY 500"),
    "HINDCOPPER":  (22000,"Metals","NIFTY 500"),
    "NATIONALUM":  (18000,"Metals","NIFTY 500"),
    "WELCORP":     (12000,"Metals","NIFTY 500"),
    "RATNAMANI":   (12000,"Metals","NIFTY 500"),
    "JINDALSAW":   (10000,"Metals","NIFTY 500"),
    "JKCEMENT":    (18000,"Capital Goods","NIFTY 500"),
    "DALBHARAT":   (25000,"Capital Goods","NIFTY 500"),
    "RAMCOCEM":    (12000,"Capital Goods","NIFTY 500"),
    "BIRLACORPN":  (10000,"Capital Goods","NIFTY 500"),
    "THERMAX":     (25000,"Capital Goods","NIFTY 500"),
    "CARBORUNIV":  (12000,"Capital Goods","NIFTY 500"),
    "SUPREME":     (30000,"Capital Goods","NIFTY 500"),
    "FINOLEXCAB":  (18000,"Capital Goods","NIFTY 500"),
    "KPIL":        (12000,"Capital Goods","NIFTY 500"),
    "KEC":         (18000,"Capital Goods","NIFTY 500"),
    "VGUARD":      (8000,"Capital Goods","NIFTY 500"),
    "AMBER":       (10000,"Capital Goods","NIFTY 500"),
    "KAYNES":      (18000,"Capital Goods","NIFTY 500"),
    "AIAENG":      (25000,"Capital Goods","NIFTY 500"),
    "GRINDWELL":   (18000,"Capital Goods","NIFTY 500"),
    "FINOLEXIND":  (9000,"Capital Goods","NIFTY 500"),
    "DEEPAKNTR":   (30000,"Chemicals","NIFTY 500"),
    "TATACHEM":    (30000,"Chemicals","NIFTY 500"),
    "AARTIIND":    (12000,"Chemicals","NIFTY 500"),
    "NAVINFLUOR":  (18000,"Chemicals","NIFTY 500"),
    "KANSAINER":   (25000,"Chemicals","NIFTY 500"),
    "FINEORG":     (8000,"Chemicals","NIFTY 500"),
    "BASF":        (12000,"Chemicals","NIFTY 500"),
    "BAYERCROP":   (18000,"Chemicals","NIFTY 500"),
    "IPCALAB":     (18000,"Pharma","NIFTY 500"),
    "BIOCON":      (22000,"Pharma","NIFTY 500"),
    "GLENMARK":    (18000,"Pharma","NIFTY 500"),
    "WOCKPHARMA":  (12000,"Pharma","NIFTY 500"),
    "GRANULES":    (8000,"Pharma","NIFTY 500"),
    "NATCOPHARM":  (8000,"Pharma","NIFTY 500"),
    "JBCHEPHARM":  (12000,"Pharma","NIFTY 500"),
    "PGHL":        (15000,"Pharma","NIFTY 500"),
    "PFIZER":      (20000,"Pharma","NIFTY 500"),
    "SANOFI":      (18000,"Pharma","NIFTY 500"),
    "STAR":        (8000,"Pharma","NIFTY 500"),
    "ZYDUSWELL":   (18000,"FMCG","NIFTY 500"),
    "LALPATHLAB":  (20000,"Healthcare","NIFTY 500"),
    "METROPOLIS":  (8000,"Healthcare","NIFTY 500"),
    "MAXHEALTH":   (28000,"Healthcare","NIFTY 500"),
    "FORTIS":      (28000,"Healthcare","NIFTY 500"),
    "ASTER":       (18000,"Healthcare","NIFTY 500"),
    "KIMS":        (12000,"Healthcare","NIFTY 500"),
    "NARAYANAH":   (18000,"Healthcare","NIFTY 500"),
    "APOLLOTYRE":  (18000,"Automobile","NIFTY 500"),
    "CEATLTD":     (10000,"Automobile","NIFTY 500"),
    "EXIDEIND":    (16000,"Automobile","NIFTY 500"),
    "AMARA":       (18000,"Automobile","NIFTY 500"),
    "SUNDRMFAST":  (18000,"Automobile","NIFTY 500"),
    "SCHAEFFLER":  (25000,"Automobile","NIFTY 500"),
    "TIINDIA":     (30000,"Automobile","NIFTY 500"),
    "MAHINDCIE":   (12000,"Automobile","NIFTY 500"),
    "CDSL":        (22000,"Financial Services","NIFTY 500"),
    "CAMS":        (18000,"Financial Services","NIFTY 500"),
    "MCX":         (18000,"Financial Services","NIFTY 500"),
    "ANGELONE":    (18000,"Financial Services","NIFTY 500"),
    "NIPPONLAMF":  (30000,"Financial Services","NIFTY 500"),
    "MFSL":        (25000,"Financial Services","NIFTY 500"),
    "ABSLAMC":     (18000,"Financial Services","NIFTY 500"),
    "IIFL":        (18000,"Financial Services","NIFTY 500"),
    "CANFINHOME":  (12000,"Financial Services","NIFTY 500"),
    "PNBHOUSING":  (18000,"Financial Services","NIFTY 500"),
    "POLICYBZR":   (35000,"Financial Services","NIFTY 500"),
    "PAYTM":       (40000,"Financial Services","NIFTY 500"),
    "CRISIL":      (30000,"Financial Services","NIFTY 500"),
    "LICHSGFIN":   (22000,"Financial Services","NIFTY 500"),
    "MANAPPURAM":  (18000,"Financial Services","NIFTY 500"),
    "NUVAMA":      (18000,"Financial Services","NIFTY 500"),
    "IRCON":       (20000,"Infrastructure","NIFTY 500"),
    "RAILTEL":     (12000,"Infrastructure","NIFTY 500"),
    "IRB":         (18000,"Infrastructure","NIFTY 500"),
    "ASHOKA":      (8000,"Infrastructure","NIFTY 500"),
    "JPOWER":      (18000,"Power","NIFTY 500"),
    "CESC":        (18000,"Power","NIFTY 500"),
    "INDIAMART":   (18000,"Information Technology","NIFTY 500"),
    "NEWGEN":      (8000,"Information Technology","NIFTY 500"),
    "CYIENT":      (14000,"Information Technology","NIFTY 500"),
    "JUSTDIAL":    (8000,"Information Technology","NIFTY 500"),
    "PVRINOX":     (12000,"Media","NIFTY 500"),
    "SUNTV":       (18000,"Media","NIFTY 500"),
    "ZEEL":        (12000,"Media","NIFTY 500"),
    "NETWORK18":   (8000,"Media","NIFTY 500"),
    "HFCL":        (8000,"Telecom","NIFTY 500"),
    "STLTECH":     (10000,"Telecom","NIFTY 500"),
    "IDEA":        (25000,"Telecom","NIFTY 500"),
    "ABFRL":       (18000,"Consumer","NIFTY 500"),
    "RAYMOND":     (12000,"Consumer","NIFTY 500"),
    "VOLTAS":      (40000,"Consumer","NIFTY 500"),
    "BLUESTARCO":  (18000,"Consumer","NIFTY 500"),
    "CROMPTON":    (18000,"Consumer","NIFTY 500"),
    "WHIRLPOOL":   (10000,"Consumer","NIFTY 500"),
    "EMAMILTD":    (28000,"FMCG","NIFTY 500"),
    "PATANJALI":   (25000,"FMCG","NIFTY 500"),
    "GODREJIND":   (25000,"FMCG","NIFTY 500"),
    "RADICO":      (14000,"Consumer","NIFTY 500"),
    "UNITDSPR":    (12000,"Consumer","NIFTY 500"),
    "UBL":         (30000,"Consumer","NIFTY 500"),
    "JUBLFOOD":    (55000,"Consumer","NIFTY 500"),
    "WESTLIFE":    (8000,"Consumer","NIFTY 500"),
    "DEVYANI":     (12000,"Consumer","NIFTY 500"),
    "IRCTC":       (70000,"Consumer","NIFTY 500"),
    "CENTURYPLY":  (8000,"Consumer","NIFTY 500"),
    "JYOTHYLAB":   (8000,"FMCG","NIFTY 500"),
    "VSTIND":      (12000,"Consumer","NIFTY 500"),
    "GODFRYPHLP":  (10000,"Consumer","NIFTY 500"),
    "AWL":         (18000,"FMCG","NIFTY 500"),
    "GALAXYSURF":  (7000,"Chemicals","NIFTY 1000"),
    "SUDARSCHEM":  (5000,"Chemicals","NIFTY 1000"),
    "VINATIORGA":  (7000,"Chemicals","NIFTY 1000"),
    "ALKYLAMINE":  (4500,"Chemicals","NIFTY 1000"),
    "NOCIL":       (4000,"Chemicals","NIFTY 1000"),
    "CLEAN":       (4000,"Chemicals","NIFTY 1000"),
    "INDIGOPNTS":  (4000,"Chemicals","NIFTY 1000"),
    "GHCL":        (5000,"Chemicals","NIFTY 1000"),
    "RALLIS":      (6000,"Chemicals","NIFTY 1000"),
    "SOLARA":      (3000,"Pharma","NIFTY 1000"),
    "NEULANDLAB":  (7000,"Pharma","NIFTY 1000"),
    "SUVEN":       (8000,"Pharma","NIFTY 1000"),
    "CAPLIPOINT":  (5000,"Pharma","NIFTY 1000"),
    "SEQUENT":     (4000,"Pharma","NIFTY 1000"),
    "THYROCARE":   (2800,"Healthcare","NIFTY 1000"),
    "RAINBOW":     (8000,"Healthcare","NIFTY 1000"),
    "KRSNAA":      (3500,"Healthcare","NIFTY 1000"),
    "YATHARTH":    (5000,"Healthcare","NIFTY 1000"),
    "DCBBANK":     (5000,"Banking","NIFTY 1000"),
    "JSWHL":       (8000,"Metals","NIFTY 1000"),
    "GRAVITA":     (7000,"Metals","NIFTY 1000"),
    "TATAMETALI":  (5000,"Metals","NIFTY 1000"),
    "SYRMA":       (5000,"Capital Goods","NIFTY 1000"),
    "TITAGARH":    (8000,"Capital Goods","NIFTY 1000"),
    "HBLPOWER":    (7000,"Capital Goods","NIFTY 1000"),
    "ELECON":      (6000,"Capital Goods","NIFTY 1000"),
    "ISGEC":       (7000,"Capital Goods","NIFTY 1000"),
    "INOXWIND":    (7000,"Capital Goods","NIFTY 1000"),
    "PRINCEPIPE":  (7000,"Capital Goods","NIFTY 1000"),
    "NILKAMAL":    (5000,"Capital Goods","NIFTY 1000"),
    "UNIPARTS":    (4000,"Capital Goods","NIFTY 1000"),
    "NUVOCO":      (7000,"Capital Goods","NIFTY 1000"),
    "KNRCON":      (8000,"Infrastructure","NIFTY 1000"),
    "NCC":         (8000,"Infrastructure","NIFTY 1000"),
    "HGINFRA":     (7000,"Infrastructure","NIFTY 1000"),
    "AHLUWALIA":   (5000,"Infrastructure","NIFTY 1000"),
    "HCC":         (5000,"Infrastructure","NIFTY 1000"),
    "CAPACITE":    (4000,"Infrastructure","NIFTY 1000"),
    "PNCINFRA":    (7000,"Infrastructure","NIFTY 1000"),
    "NESCO":       (5000,"Infrastructure","NIFTY 1000"),
    "RATEGAIN":    (7000,"Information Technology","NIFTY 1000"),
    "INTELLECT":   (7000,"Information Technology","NIFTY 1000"),
    "MASTEK":      (8000,"Information Technology","NIFTY 1000"),
    "TANLA":       (7000,"Information Technology","NIFTY 1000"),
    "LATENTVIEW":  (6000,"Information Technology","NIFTY 1000"),
    "BSOFT":       (5000,"Information Technology","NIFTY 1000"),
    "KPITECH":     (12000,"Information Technology","NIFTY 1000"),
    "IXIGO":       (6000,"Consumer","NIFTY 1000"),
    "SAREGAMA":    (7000,"Consumer","NIFTY 1000"),
    "NAZARA":      (5000,"Consumer","NIFTY 1000"),
    "GREENPANEL":  (5000,"Consumer","NIFTY 1000"),
    "PLYBOARDS":   (5000,"Consumer","NIFTY 1000"),
    "SYMPHONY":    (5000,"Consumer","NIFTY 1000"),
    "ORIENTELEC":  (4000,"Consumer","NIFTY 1000"),
    "VMART":       (5000,"Consumer","NIFTY 1000"),
    "BAJAJCON":    (5000,"Consumer","NIFTY 1000"),
    "JKPAPER":     (6000,"Consumer","NIFTY 1000"),
    "HATSUN":      (7000,"FMCG","NIFTY 1000"),
    "KRBL":        (6000,"FMCG","NIFTY 1000"),
    "LTFOODS":     (6000,"FMCG","NIFTY 1000"),
    "AVANTIFEED":  (5000,"FMCG","NIFTY 1000"),
    "TEJASNET":    (7000,"Telecom","NIFTY 1000"),
    "VINDHYATEL":  (4000,"Telecom","NIFTY 1000"),
    "TTML":        (8000,"Telecom","NIFTY 1000"),
    "TIPSINDLTD":  (5000,"Media","NIFTY 1000"),
    "ICRA":        (8000,"Financial Services","NIFTY 1000"),
    "CARERATING":  (5000,"Financial Services","NIFTY 1000"),
    "5PAISA":      (2500,"Financial Services","NIFTY 1000"),
    "SUPRAJIT":    (6000,"Automobile","NIFTY 1000"),
    "CRAFTSMAN":   (8000,"Automobile","NIFTY 1000"),
    "LUMAXTECH":   (5000,"Automobile","NIFTY 1000"),
    "JAMNAAUTO":   (4000,"Automobile","NIFTY 1000"),
    "SAGCEM":      (6000,"Capital Goods","NIFTY 1000"),
    "STARCEMENT":  (5000,"Capital Goods","NIFTY 1000"),
}

def _nse_session():
    s = requests.Session()
    try:
        s.get("https://www.nseindia.com", headers=NSE_HEADERS, timeout=10)
        time.sleep(0.5)
    except Exception:
        pass
    return s

def _fetch_index_constituents(session):
    result = {}
    for idx, url in INDEX_CSV_URLS.items():
        for attempt in range(3):
            try:
                r = session.get(url, headers=NSE_HEADERS, timeout=20)
                if r.status_code == 200 and len(r.content) > 100:
                    df = pd.read_csv(io.StringIO(r.text))
                    sym_col = next((c for c in df.columns if "symbol" in c.lower()), None)
                    if sym_col:
                        syms = df[sym_col].dropna().str.strip().str.upper().tolist()
                        result[idx] = [s for s in syms if s]
                        log.info("  %s: %d stocks", idx, len(syms))
                    break
                time.sleep(0.5 * (attempt + 1))
            except Exception as e:
                log.debug("  %s attempt %d failed: %s", idx, attempt + 1, e)
                time.sleep(1)
        time.sleep(0.2)
    return result

def _assign_index_membership(symbol, index_members):
    for idx in BROAD_PRIORITY:
        if symbol in index_members.get(idx, []):
            return idx
    return "NIFTY 2000"

def _assign_sector_from_indices(symbol, index_members):
    for idx in SECTOR_INDICES:
        if symbol in index_members.get(idx, []):
            sec = SECTOR_INDEX_TO_SECTOR.get(idx)
            if sec:
                return sec
    return None

def _enrich_mcap(df, progress_cb=None):
    need = df[df["market_cap_cr"].isna()]["symbol"].tolist()
    if not need:
        return df
    log.info("Fetching market cap for %d stocks via yfinance...", len(need))
    mcap_data = {}
    for i in range(0, len(need), 100):
        batch = need[i:i + 100]
        if progress_cb:
            progress_cb(i, len(need), "Fetching market caps")
        for sym in batch:
            try:
                fi = yf.Ticker(sym + ".NS").fast_info
                mc = getattr(fi, "market_cap", None)
                if mc and mc > 0:
                    mcap_data[sym] = round(mc / 1e7, 2)
            except Exception:
                pass
        time.sleep(0.3)
    for sym, mc in mcap_data.items():
        df.loc[df["symbol"] == sym, "market_cap_cr"] = mc
    df["cap_tier"] = df["market_cap_cr"].apply(assign_tier)
    return df

def build_universe(force_refresh=False, progress_cb=None):
    if UNIVERSE_F.exists() and not force_refresh:
        mtime     = datetime.fromtimestamp(UNIVERSE_F.stat().st_mtime)
        age_hours = (datetime.now() - mtime).total_seconds() / 3600
        if age_hours < 20:
            df = pd.read_csv(UNIVERSE_F)
            log.info("Loaded cached universe: %d stocks (%.1fh old)", len(df), age_hours)
            return df

    log.info("Building fresh universe...")
    if progress_cb: progress_cb(1, 5, "Connecting to NSE...")
    session = _nse_session()

    if progress_cb: progress_cb(2, 5, "Fetching NSE index lists...")
    index_members = {}
    try:
        index_members = _fetch_index_constituents(session)
        log.info("Got %d index lists from NSE", len(index_members))
    except Exception as e:
        log.warning("NSE index fetch failed: %s", e)

    live_syms   = set()
    for idx in BROAD_INDICES:
        live_syms.update(index_members.get(idx, []))
    seed_syms   = set(SEED_UNIVERSE.keys())
    all_symbols = sorted(live_syms | seed_syms)
    log.info("Total symbols: %d  (live: %d  seed: %d)",
             len(all_symbols), len(live_syms), len(seed_syms))

    if progress_cb: progress_cb(3, 5, "Building master table...")
    records = []
    for sym in all_symbols:
        rec = {"symbol": sym, "ticker": sym + ".NS"}
        live_idx = _assign_index_membership(sym, index_members) if index_members else None

        if sym in SEED_UNIVERSE:
            mcap, sector, seed_idx = SEED_UNIVERSE[sym]
            rec["market_cap_cr"] = mcap
            rec["nse_sector"]    = sector
            rec["cap_tier"]      = assign_tier(mcap)
            rec["nifty_index"]   = live_idx if (live_idx and live_idx != "NIFTY 2000") else seed_idx
        else:
            rec["market_cap_cr"] = np.nan
            rec["nse_sector"]    = "Others"
            rec["cap_tier"]      = "Unknown"
            rec["nifty_index"]   = live_idx if live_idx else "NIFTY 1000"

        if index_members:
            live_sector = _assign_sector_from_indices(sym, index_members)
            if live_sector:
                rec["nse_sector"] = live_sector

        sec_idxs              = [i for i in SECTOR_INDICES if sym in index_members.get(i, [])]
        rec["sector_indices"] = "|".join(sec_idxs)
        rec["company_name"]   = sym
        records.append(rec)

    df = pd.DataFrame(records)

    if progress_cb: progress_cb(4, 5, "Enriching market caps...")
    missing = df["market_cap_cr"].isna().sum()
    if 0 < missing < 1500:
        df = _enrich_mcap(df, progress_cb)

    df["nifty_index"] = df.apply(
        lambda r: r["nifty_index"]
        if r["nifty_index"] not in (None, "", "NIFTY 2000")
        else _infer_nifty_index(r.get("market_cap_cr", np.nan)),
        axis=1
    )
    df["cap_tier"] = df["market_cap_cr"].apply(assign_tier)

    if progress_cb: progress_cb(5, 5, "Saving universe...")
    df.sort_values("market_cap_cr", ascending=False, inplace=True)
    df.reset_index(drop=True, inplace=True)
    df["rank_by_mcap"] = range(1, len(df) + 1)
    df.to_csv(UNIVERSE_F, index=False)
    log.info("Universe saved: %d stocks -> %s", len(df), UNIVERSE_F)
    return df

def filter_universe(df, cap_tiers=None, mcap_min_cr=0, mcap_max_cr=float("inf"),
                    sectors=None, nifty_index=None,
                    min_mcap_rank=None, max_mcap_rank=None):
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
                    "NIFTY 1000","NIFTY MICROCAP 250","NIFTY 2000"]
        pos     = priority.index(nifty_index) if nifty_index in priority else len(priority)
        allowed = set(priority[:pos + 1])
        out = out[out["nifty_index"].isin(allowed)]
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
    print("\nIndex distribution:")
    print(df["nifty_index"].value_counts())
    print("\nCap tier distribution:")
    print(df["cap_tier"].value_counts())
    print("\nSector distribution (top 15):")
    print(df["nse_sector"].value_counts().head(15))
