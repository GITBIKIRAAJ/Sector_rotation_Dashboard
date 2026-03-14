"""
app.py  (v4 — Sector Dashboard + Stock Ranker + Screener)
Run: streamlit run app.py
"""
import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime
from page_overview import render_overview_page


st.set_page_config(
    page_title="NSE Sector Rotation Dashboard",
    page_icon="📊", layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
  .block-container { padding-top: 1rem; }
  .metric-card {
    background: #f8fafc; border: 1px solid #e2e8f0;
    border-radius: 10px; padding: 14px 18px; margin-bottom: 8px;
  }
  .stock-card-top {
    background: linear-gradient(135deg,#f0fdf4,#dcfce7);
    border: 1px solid #86efac; border-radius: 10px;
    padding: 12px 16px; margin-bottom: 8px;
  }
  .stock-card-bot {
    background: linear-gradient(135deg,#fef2f2,#fee2e2);
    border: 1px solid #fca5a5; border-radius: 10px;
    padding: 12px 16px; margin-bottom: 8px;
  }
  .cap-badge {
    display:inline-block; padding:2px 8px; border-radius:12px;
    font-size:11px; font-weight:600; color:#fff; margin-left:6px;
  }
  .weight-def { font-size:12px; color:#6b7280; line-height:1.5; }
</style>
""", unsafe_allow_html=True)

from data_engine import load_all_data, SECTORS, BENCHMARK, TIMEFRAMES
import charts as ch
from page_screener import render_screener_page
from stock_engine import (
    build_universe, filter_universe, get_sectors, get_tickers,
    rank_stocks, get_top_bottom, fetch_sector_index_prices,
    CAP_TIER_ORDER, CAP_TIER_COLORS, CAP_TIER_RANGES, DEFAULT_WEIGHTS, WEIGHT_DEFINITIONS,
)

# ── Cached loaders ────────────────────────────────────────────────────────────

@st.cache_data(ttl=900, show_spinner=False)
def get_data():
    return load_all_data()

@st.cache_data(ttl=3600, show_spinner=False)
def get_universe():
    return build_universe()

# ── Sidebar ───────────────────────────────────────────────────────────────────

with st.sidebar:
    st.markdown("## ⚙️ Settings")

    page = st.radio(
        "📄 Page",
        ["🏠 Overview", "📊 Sector Dashboard", "📈 Stock Ranker", "🔍 Screener"],
        label_visibility="collapsed"
    )
    st.markdown("---")

    if page == "📊 Sector Dashboard":
        tf_options  = ["1W","1M","3M","6M","YTD","1Y"]
        selected_tf = st.selectbox("Primary Timeframe", tf_options, index=1)
        st.markdown("---")
        st.markdown("### Sector Filter")
        sector_list = list(SECTORS.keys())
        select_all  = st.checkbox("Select All Sectors", value=True)
        if select_all:
            selected_sectors = sector_list
        else:
            selected_sectors = st.multiselect("Choose Sectors", sector_list, default=sector_list[:10])
        st.markdown("---")
        st.markdown("### Chart Period")
        chart_period = st.radio("Trend Chart Lookback", ["1 Week","1 Month","3 Months"], index=2)
        period_map   = {"1 Week":"norm_7","1 Month":"norm_30","3 Months":"norm_90"}
        st.markdown("---")
        st.markdown("### RRG Settings")
        rrg_window = st.slider("RS Smoothing Window (days)", 5, 30, 10)

    elif page == "📈 Stock Ranker":
        st.markdown("### Universe Filter")
        universe_df_full = get_universe()
        all_sectors_uni  = sorted(universe_df_full["nse_sector"].dropna().unique().tolist())

        nifty_idx   = st.selectbox("NIFTY Index", ["All","NIFTY 50","NIFTY 100","NIFTY 200",
                                                    "NIFTY 500","NIFTY 1000","NIFTY 2000"])
        cap_options = st.multiselect("Cap Tier", CAP_TIER_ORDER[:-1], default=CAP_TIER_ORDER[:-1])
        mcap_min    = st.number_input("Min cap (Cr)", value=1000, step=500)
        mcap_max    = st.number_input("Max cap (Cr)", value=50000, step=1000)
        st.markdown("---")
        st.markdown("### Sector")
        show_all_sec = st.radio("Show", ["All sectors","Select specific"], index=0)
        if show_all_sec == "Select specific":
            chosen_sectors = st.multiselect("Sectors", all_sectors_uni, default=all_sectors_uni[:3])
        else:
            chosen_sectors = None
        st.markdown("---")

        st.markdown("### Score weights")
        st.caption("Adjust to your style — auto-normalised")

        raw_weights = {}
        for key, (label, definition) in WEIGHT_DEFINITIONS.items():
            default_pct = int(DEFAULT_WEIGHTS[key] * 100)
            col1, col2 = st.columns([8, 1])
            with col1:
                raw_weights[key] = st.slider(
                    label.split("(")[0].strip(), 0, 50, default_pct, key=f"w_{key}"
                )
            with col2:
                st.markdown(
                    f'<span title="{definition}" style="cursor:help;font-size:16px;">❓</span>',
                    unsafe_allow_html=True,
                )
            with st.expander("", expanded=False):
                st.markdown(f'<p class="weight-def">{definition}</p>', unsafe_allow_html=True)

        total_w = sum(raw_weights.values()) or 1
        weights = {k: v/total_w for k, v in raw_weights.items()}
        top_n   = st.slider("Top / Bottom N stocks", 5, 20, 14)

    # Screener page has its own sidebar rendered inside render_screener_page()

    st.markdown("---")
    if st.button("🔄 Refresh Data", use_container_width=True):
        st.cache_data.clear()
        st.rerun()
    st.caption("Data via Yahoo Finance · 15-min cache")
    st.caption("NSE Indices · All 20 sectors vs NIFTY 50")

# ══════════════════════════════════════════════════════════════════════════════
# PAGE 1 — SECTOR DASHBOARD
# ══════════════════════════════════════════════════════════════════════════════
if page == "🏠 Overview":
    sector_data = None
    sector_prices = None
    try:
        sector_data = get_data()
        sector_prices = sector_data["prices"]
    except Exception:
        pass
    render_overview_page(
        universe_df=get_universe(),
        prices_df=sector_prices,
        sectors=list(SECTORS.keys()),
    )


elif page == "📊 Sector Dashboard":
    with st.spinner("⏳ Fetching NSE sector data..."):
        try:
            data    = get_data()
            data_ok = True
        except Exception as e:
            st.error(f"Data fetch error: {e}")
            data_ok = False
    if not data_ok:
        st.stop()

    prices         = data["prices"]
    returns        = data["returns"]
    rel_returns    = data["rel_returns"]
    rrg            = data["rrg"]
    rrg_days       = data.get("rrg_days", pd.DataFrame())
    drawdown       = data["drawdown"]
    volatility     = data["volatility"]
    rank_shift     = data["rank_shift"]
    benchmark_name = data["benchmark_name"]
    norm_data      = data[period_map[chart_period]]
    rs_data        = data["rs_90"]

    def get_metric(name):
        if name in prices.columns and len(prices[name].dropna()) >= 2:
            s = prices[name].dropna()
            return round((s.iloc[-1]-s.iloc[-2])/s.iloc[-2]*100, 2)
        return np.nan

    st.markdown("# 📊 NSE Sector Rotation Dashboard")
    col_h1, _, col_h3 = st.columns([3,1,1])
    with col_h1:
        st.caption(f"Last updated: {data['last_updated']} · Benchmark: NIFTY 50 · 20 Sectors tracked")
    with col_h3:
        st.caption(f"Primary TF: **{selected_tf}**")
    st.markdown("---")

    # ── Section A ────────────────────────────────────────────────────────────
    st.markdown("### 📌 A — Benchmark & Top/Bottom Sectors")
    bm_chg     = get_metric(benchmark_name)
    bm_chg_str = f"{bm_chg:+.2f}%" if pd.notna(bm_chg) else "N/A"

    colA1, colA2, colA3 = st.columns(3)
    with colA1:
        st.metric("NIFTY 50 (1D)", bm_chg_str)
    if selected_tf in returns.columns:
        top3 = returns[selected_tf].dropna().nlargest(3)
        bot3 = returns[selected_tf].dropna().nsmallest(3)
        with colA2:
            st.markdown("**🏆 Top 3**")
            for sec, val in top3.items():
                st.markdown(f"🟢 **{sec.replace('NIFTY ','')}** `{val:+.1f}%`")
        with colA3:
            st.markdown("**📉 Bottom 3**")
            for sec, val in bot3.items():
                st.markdown(f"🔴 **{sec.replace('NIFTY ','')}** `{val:+.1f}%`")
    st.markdown("---")

    # ── Section B ────────────────────────────────────────────────────────────
    st.markdown("### 📊 B — Sector Performance")
    tab1, tab2, tab3 = st.tabs(["📋 Table","🌡️ Heat Map","📊 Relative Bars"])
    with tab1:
        st.plotly_chart(
            ch.make_performance_table(returns, rel_returns, benchmark_name, selected_sectors, selected_tf),
            use_container_width=True)
    with tab2:
        st.plotly_chart(ch.make_heatmap(rel_returns, selected_sectors, benchmark_name),
                        use_container_width=True)
    with tab3:
        if selected_tf in rel_returns.columns:
            sec_rel = rel_returns.loc[
                [s for s in selected_sectors if s in rel_returns.index], selected_tf
            ].dropna().sort_values()
            fig_bar = ch.go.Figure(ch.go.Bar(
                x=sec_rel.index, y=sec_rel.values,
                marker=dict(color=["#16a34a" if v >= 0 else "#dc2626" for v in sec_rel.values]),
                text=[f"{v:+.1f}%" for v in sec_rel.values], textposition="outside",
            ))
            fig_bar.update_layout(
                paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                height=380, xaxis_tickangle=-35,
                yaxis_title=f"vs NIFTY 50 ({selected_tf})",
                margin=dict(l=10,r=10,t=20,b=80))
            st.plotly_chart(fig_bar, use_container_width=True)
    st.markdown("---")

    # ── Section C ────────────────────────────────────────────────────────────
    st.markdown("### 📈 C — Trend & Relative Strength")
    tc1, tc2 = st.tabs(["📉 Normalized Trend","📐 Rolling RS Lines"])
    with tc1:
        st.plotly_chart(
            ch.make_normalized_chart(norm_data, benchmark_name, selected_sectors,
                                     title=f"Normalized ({chart_period})"),
            use_container_width=True)
    with tc2:
        st.plotly_chart(ch.make_rs_chart(rs_data, selected_sectors), use_container_width=True)
    st.markdown("---")

    # ── Section D ────────────────────────────────────────────────────────────
    st.markdown("### 🔄 D — Relative Rotation & Rank Shift")
    dc1, dc2 = st.tabs(["🎯 RRG Chart","📊 Rank Shift"])
    with dc1:
        if not rrg.empty:
            st.plotly_chart(ch.make_rrg_chart(rrg, rrg_days), use_container_width=True)
            if not rrg_days.empty:
                st.markdown("#### 📅 Days in Current Quadrant")
                disp = rrg_days.copy().reset_index()
                disp.columns = ["Sector","Quadrant","Trading Days"]
                disp = disp.sort_values("Trading Days", ascending=False)
                quad_colors = {"Leading":"🟢","Improving":"🔵","Weakening":"🟡","Lagging":"🔴"}
                disp["Quadrant"] = disp["Quadrant"].apply(lambda q: f"{quad_colors.get(q,'⚪')} {q}")
                st.dataframe(disp, use_container_width=True, hide_index=True)
        else:
            st.info("RRG data unavailable.")
    with dc2:
        st.plotly_chart(ch.make_rank_shift_chart(rank_shift), use_container_width=True)
    st.markdown("---")

    # ── Section E ────────────────────────────────────────────────────────────
    st.markdown("### ⚠️ E — Risk Metrics")
    ec1, ec2 = st.tabs(["📊 Volatility","📉 Drawdown"])
    with ec1:
        st.plotly_chart(ch.make_volatility_chart(volatility, benchmark_name, selected_sectors),
                        use_container_width=True)
    with ec2:
        st.plotly_chart(ch.make_drawdown_chart(drawdown, benchmark_name, selected_sectors),
                        use_container_width=True)
    st.markdown("---")

    # ── Section F ────────────────────────────────────────────────────────────
    st.markdown("### 🚨 F — Rotation Alerts")
    alerts = []
    if not rrg.empty:
        for sec, row in rrg.iterrows():
            q     = row.get("Quadrant","")
            d_str = ""
            if not rrg_days.empty and sec in rrg_days.index:
                d     = rrg_days.loc[sec, "Days_In_Quadrant"]
                d_str = f" · {d}d in quadrant"
            if q == "Leading":     alerts.append(f"🟢 **{sec}** — Leading (strong & gaining){d_str}")
            elif q == "Improving": alerts.append(f"🔵 **{sec}** — Improving (weak but recovering){d_str}")
            elif q == "Weakening": alerts.append(f"🟡 **{sec}** — Weakening (strong but fading){d_str}")
            elif q == "Lagging":   alerts.append(f"🔴 **{sec}** — Lagging (weak & falling){d_str}")
    if not rank_shift.empty:
        rs_df = rank_shift.dropna(subset=["Shift"])
        rs_df["Shift"] = rs_df["Shift"].astype(float)
        for sec, row in rs_df.iterrows():
            shift = float(row["Shift"])
            if shift >= 5:    alerts.append(f"🚀 **{sec}** — Rank improved by {shift:.0f} places (4W)")
            elif shift <= -5: alerts.append(f"⚠️ **{sec}** — Rank dropped by {abs(shift):.0f} places (4W)")
    if alerts:
        for a in alerts:
            st.markdown(a)
    else:
        st.info("No significant rotation signals detected.")
    st.markdown("---")

    # ── Section G ────────────────────────────────────────────────────────────
    st.markdown("### 💾 G — Export Data")
    if not returns.empty:
        csv = returns.to_csv()
        st.download_button("📥 Download Returns CSV", csv, "nse_returns.csv", "text/csv")

# ══════════════════════════════════════════════════════════════════════════════
# PAGE 2 — STOCK RANKER
# ══════════════════════════════════════════════════════════════════════════════

elif page == "📈 Stock Ranker":
    st.markdown("# 📈 NSE Stock Ranker")
    st.caption("Scores stocks on 7 technical factors — adjust weights in sidebar")

    universe_df_full = get_universe()

    nifty_filter = None if nifty_idx == "All" else nifty_idx
    universe_df  = filter_universe(
        universe_df_full,
        cap_tiers   = cap_options or None,
        mcap_min_cr = mcap_min,
        mcap_max_cr = mcap_max,
        sectors     = chosen_sectors,
        nifty_index = nifty_filter,
    )

    if universe_df.empty:
        st.warning("No stocks match the current filters. Please relax the criteria.")
        st.stop()

    st.info(f"🔍 Universe: **{len(universe_df):,}** stocks after filters")

    with st.spinner("⏳ Scoring stocks… (parallel fetch)"):
        try:
            data_s   = get_data()
            nifty_px = data_s["prices"].get("NIFTY 50", pd.Series(dtype=float))
            all_sectors_in_universe = universe_df["nse_sector"].dropna().unique().tolist()
            sec_px   = fetch_sector_index_prices(all_sectors_in_universe)

            prog_bar = st.progress(0, text="Fetching prices…")
            def progress_cb(done, total, msg):
                pct = int(done/total*100) if total > 0 else 0
                prog_bar.progress(min(pct,100), text=msg)

            ranked_df = rank_stocks(universe_df, nifty_px, sec_px, weights, progress_cb)
            prog_bar.empty()
            score_ok  = True
        except Exception as e:
            st.error(f"Scoring error: {e}")
            score_ok = False

    if not score_ok or ranked_df.empty:
        st.warning("Could not score stocks. Try refreshing or relaxing filters.")
        st.stop()

    top_df, bot_df = get_top_bottom(ranked_df, n=top_n)

    CAP_COLOR_CSS = {
        "Mega":"#7c3aed","Large":"#2563eb","Mid":"#16a34a",
        "Small":"#ca8a04","Micro":"#dc2626","Unknown":"#9ca3af",
    }

    def render_stock_card(row, card_class):
        tier   = row.get("cap_tier","Unknown")
        color  = CAP_COLOR_CSS.get(tier,"#9ca3af")
        badge  = f'<span class="cap-badge" style="background:{color}">{tier}</span>'
        score  = row.get("composite_score", 0)
        rsi    = row.get("rsi", float("nan"))
        ema    = row.get("ema_label","N/A")
        rs_nf  = row.get("rs_nifty", float("nan"))
        r1m    = row.get("return_1m", float("nan"))
        sec    = row.get("nse_sector","")
        nidx   = row.get("nifty_index","")
        mcap   = row.get("market_cap_cr", float("nan"))
        rsi_s  = f"{rsi:.0f}"    if pd.notna(rsi)   else "N/A"
        rs_s   = f"{rs_nf:.1f}"  if pd.notna(rs_nf) else "N/A"
        r1m_s  = f"{r1m:+.1f}%"  if pd.notna(r1m)   else "N/A"
        mcap_s = f"{mcap:,.0f} Cr" if pd.notna(mcap) else "N/A"
        dr     = row.get("display_rank","")
        name   = row.get("stock", row.get("ticker",""))
        st.markdown(f"""
        <div class="{card_class}">
          <b>#{dr} {name}</b> {badge}
          <span style="float:right;font-weight:700;font-size:15px;">Score: {score:.0f}/100</span><br>
          <small style="color:#6b7280">{sec} · {nidx} · {mcap_s}</small><br>
          <small>RSI {rsi_s} | EMA {ema} | RS/NF {rs_s} | 1M {r1m_s}</small>
        </div>
        """, unsafe_allow_html=True)

    col_top, col_bot = st.columns(2)
    with col_top:
        st.markdown(f"### 🟢 Top {len(top_df)} — Strongest")
        if top_df.empty:
            st.info("Not enough stocks for top list.")
        else:
            for _, row in top_df.iterrows():
                render_stock_card(row, "stock-card-top")

    with col_bot:
        st.markdown(f"### 🔴 Bottom {len(bot_df)} — Weakest")
        if bot_df.empty:
            st.info("Not enough stocks for bottom list.")
        else:
            for _, row in bot_df.iterrows():
                render_stock_card(row, "stock-card-bot")

    st.markdown("---")

    with st.expander("📋 Full Ranked Table", expanded=False):
        import stock_charts as sc
        st.plotly_chart(sc.make_rank_table(ranked_df), use_container_width=True)

    st.markdown("---")
    csv_out = ranked_df.to_csv(index=False)
    st.download_button("📥 Download Ranked Stocks CSV", csv_out, "ranked_stocks.csv", "text/csv")

# ══════════════════════════════════════════════════════════════════════════════
# PAGE 3 — SCREENER
# ══════════════════════════════════════════════════════════════════════════════

elif page == "🔍 Screener":
    render_screener_page(get_universe())
