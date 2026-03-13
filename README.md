# 📊 NSE Sector Rotation Dashboard

A full-featured **NSE India stock market dashboard** built with Python & Streamlit.
Covers sector rotation analysis, stock ranking, and a multi-condition screener —
all powered by live data from Yahoo Finance and NSE.

---

## 🖥️ Live Demo

> Deployed on Streamlit Community Cloud — accessible from any browser, no install needed.

---

## ✨ Features

### 📊 Page 1 — Sector Dashboard
- **20 NSE sector indices** tracked vs NIFTY 50 benchmark
- Performance table across 1W / 1M / 3M / 6M / YTD / 1Y timeframes
- **Relative Rotation Graph (RRG)** — Leading / Improving / Weakening / Lagging quadrants
- Days-in-quadrant tracker per sector
- Heatmap, normalized trend lines, rolling RS lines
- Volatility & drawdown charts
- Rank shift chart (4-week sector momentum change)
- Rotation alerts with quadrant + rank change signals

### 📈 Page 2 — Stock Ranker
- **NIFTY 2000 universe** — filter by index, cap tier, sector, MCap range
- Scores every stock on **7 technical factors** (fully customisable weights):
  - RS vs NIFTY 50
  - RS vs Sector Index
  - RSI(14)
  - EMA Alignment (Price > EMA20 > EMA50 > EMA200)
  - Volume Up/Down ratio
  - ATR % (volatility)
  - Distance from 52-Week High
- Top & Bottom N cards with cap badge, sector, score breakdown
- Full ranked table with export to CSV

### 🔍 Page 3 — Stock Screener
7 ready-to-run screeners with adjustable parameters:

| Screener | Key Conditions |
|---|---|
| 📊 Turnover Screener | Turnover ≥ 100 Cr, RSI ≥ 55, Vol > 2× avg, Close > EMA50 & VWAP & Prev High, EMA stack |
| 🏔️ 52-Week High | Within X% of 52W high |
| 🚀 All-Time High | Within X% of 3-year ATH |
| 📈 Volume Breakout | Volume > N× 20-day avg + Close > EMA50 |
| ⚡ EMA Crossover | EMA10 freshly crossed above EMA20 |
| 💥 Momentum Breakout | RSI > 55 + Close > EMA50 + Close > Prev Day High |
| 🐂 Bull Trend Setup | Full EMA stack + RSI 55–78 + EMA20 slope up |

---

## 📁 Project Structure

```
nse_dashboard/
│
├── app.py                  ← Main Streamlit app (3-page router)
├── data_engine.py          ← Sector dashboard data engine (NSE indices)
├── charts.py               ← Sector dashboard Plotly charts
├── stock_engine.py         ← Stock scoring & ranking engine
├── stock_charts.py         ← Stock ranker Plotly charts (table, radar)
├── universe_builder.py     ← NSE 2000 universe builder & cache
├── page_screener.py        ← Screener page UI (Streamlit)
├── screener_engine.py      ← All 7 screener logic functions
├── requirements.txt        ← Python dependencies
│
└── data/                   ← Auto-created on first run
    └── master_universe.csv ← Cached NSE universe (refreshed daily)
```

---

## 🚀 Local Setup

### 1. Clone / Download the project
```bash
git clone https://github.com/YOUR_USERNAME/nse-dashboard.git
cd nse-dashboard
```

### 2. Create a virtual environment
```bash
python -m venv .venv

# Windows
.venv\Scripts\activate

# macOS / Linux
source .venv/bin/activate
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

### 4. Run the app
```bash
streamlit run app.py
```

App opens at → **http://localhost:8501**

> **First run:** Universe builder fetches ~2000 NSE stocks — takes 2–3 minutes.
> Subsequent runs load from cache (`data/master_universe.csv`) — instant.

---

## ☁️ Deploy on Streamlit Community Cloud (Free)

### Step 1 — Push to GitHub

Make sure your project is in a **GitHub repository**:

```bash
git init
git add .
git commit -m "Initial commit"
git branch -M main
git remote add origin https://github.com/YOUR_USERNAME/nse-dashboard.git
git push -u origin main
```

> ⚠️ Add `data/` to `.gitignore` — the universe CSV is auto-generated on deploy.

Create a `.gitignore` file:
```
.venv/
__pycache__/
*.pyc
data/
.env
```

### Step 2 — Sign up on Streamlit Community Cloud

Go to → **[share.streamlit.io](https://share.streamlit.io)**
Sign in with your GitHub account (free, no credit card needed).

### Step 3 — Deploy

1. Click **"New app"**
2. Select your GitHub repo
3. Branch: `main`
4. Main file path: `app.py`
5. Click **"Deploy!"**

Streamlit auto-installs everything from `requirements.txt` and launches the app.
Your public URL will be: `https://YOUR_USERNAME-nse-dashboard-app-XXXXX.streamlit.app`

### Step 4 — After Deployment

| Task | How |
|---|---|
| Update the app | `git push` → app auto-redeploys |
| Reboot the app | Streamlit dashboard → ⋮ menu → Reboot |
| View logs | Streamlit dashboard → "Manage app" → Logs |
| Make app private | Requires Snowflake/paid plan |

---

## ⚙️ Configuration

### Adjust cache duration
In `app.py`, change the TTL as needed:
```python
@st.cache_data(ttl=900)    # 15 min — sector data
@st.cache_data(ttl=3600)   # 1 hour — universe (NSE 2000 stocks)
```

### Force universe refresh
In the app sidebar click **🔄 Refresh Data**, or delete:
```
data/master_universe.csv
```
and restart the app.

### Change default score weights (Stock Ranker)
Edit `DEFAULT_WEIGHTS` in `stock_engine.py`:
```python
DEFAULT_WEIGHTS = {
    "rs_nifty":  0.25,   # Relative Strength vs NIFTY 50
    "rs_sector": 0.20,   # Relative Strength vs Sector
    "rsi":       0.15,   # RSI(14)
    "ema_align": 0.15,   # EMA alignment score
    "vol_ud":    0.10,   # Volume Up/Down ratio
    "atr_pct":   0.08,   # ATR % volatility
    "dist_52w":  0.07,   # Distance from 52-week high
}
```

---

## 📦 Dependencies

| Package | Version | Purpose |
|---|---|---|
| streamlit | ≥ 1.32 | Web app framework |
| yfinance | ≥ 0.2.38 | Price & volume data |
| pandas | ≥ 2.0 | Data processing |
| numpy | ≥ 1.26 | Numerical computation |
| plotly | ≥ 5.20 | Interactive charts |
| requests | ≥ 2.31 | NSE data fetching |
| beautifulsoup4 | ≥ 4.12 | HTML parsing |
| lxml | ≥ 4.9 | XML/HTML parser |

---

## ⚠️ Known Limitations

- **Data source:** Yahoo Finance — may have 15–20 min delay during market hours
- **Universe builder:** First run takes 2–3 min to fetch market caps for ~2000 stocks
- **Streamlit Cloud:** App sleeps after 7 days of inactivity (free tier) — wake it by visiting the URL
- **ATH Screener:** Fetches 3 years of data — slower than other screeners (~5 min for large universe)
- **VWAP:** Approximated using daily close × volume (true intraday VWAP not available from Yahoo Finance daily data)

---

## 📄 License

MIT License — free to use, modify, and distribute.

---

## 🙌 Built With

- [Streamlit](https://streamlit.io) — App framework
- [Yahoo Finance (yfinance)](https://pypi.org/project/yfinance/) — Market data
- [Plotly](https://plotly.com/python/) — Interactive visualisations
- [NSE India](https://www.nseindia.com) — Index constituent data
