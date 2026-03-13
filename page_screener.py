"""
page_screener.py
================
Streamlit screener page — import and call render_screener_page()
from your main app.py.

Usage in app.py:
    from page_screener import render_screener_page
    # inside the page == "🔍 Screener" branch:
    render_screener_page(universe_df_full)
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px

from screener_engine import SCREENER_REGISTRY
from universe_builder import filter_universe, build_universe
from stock_engine import CAP_TIER_ORDER, CAP_TIER_COLORS

SECTOR_LIST_ALL = [
    "All Sectors",
    "Automobile","Banking","Capital Goods","Chemicals","Commodities",
    "Consumption","CPSE","Energy","Financial Services","FMCG",
    "Healthcare","Information Technology","Infrastructure","Media",
    "Metals","MNC","Oil & Gas","Others","Pharma","Power","PSE",
    "PSU Banking","Realty","Services","Telecom",
]

INDEX_LIST = ["All","NIFTY 50","NIFTY 100","NIFTY 200","NIFTY 500",
              "NIFTY MIDCAP 150","NIFTY SMALLCAP 250","NIFTY MICROCAP 250"]

CAP_COLOR_CSS = {
    "Mega":"#7c3aed","Large":"#2563eb","Mid":"#16a34a",
    "Small":"#ca8a04","Micro":"#dc2626","Unknown":"#9ca3af",
}

# ── Result table ──────────────────────────────────────────────────────────────

def _make_result_table(df: pd.DataFrame) -> go.Figure:
    if df.empty:
        return go.Figure()

    # Choose columns based on what exists
    priority = ["rank","stock","company","sector","cap_tier","mcap_cr",
                "last_price","rsi","score",
                "dist_52w_pct","dist_from_high_pct","dist_from_ath_pct",
                "turnover_cr","turnover_ratio","vol_ratio",
                "ema10","ema20","ema50","ema200","prev_high","rs_nifty"]
    cols   = [c for c in priority if c in df.columns]
    labels = {
        "rank":"#","stock":"Stock","company":"Company","sector":"Sector",
        "cap_tier":"Cap","mcap_cr":"MCap Cr","last_price":"Price",
        "rsi":"RSI","score":"Score","dist_52w_pct":"52W%",
        "dist_from_high_pct":"52W%","dist_from_ath_pct":"ATH%",
        "turnover_cr":"TO Cr","turnover_ratio":"TO×","vol_ratio":"Vol×",
        "ema10":"EMA10","ema20":"EMA20","ema50":"EMA50","ema200":"EMA200",
        "prev_high":"Prev High","rs_nifty":"RS/NF",
    }
    headers = [labels.get(c, c) for c in cols]

    cell_vals   = []
    cell_colors = []

    CAP_TINT = {
    "Mega":    "rgba(124,58,237,0.12)",
    "Large":   "rgba(37,99,235,0.12)",
    "Mid":     "rgba(22,163,74,0.12)",
    "Small":   "rgba(202,138,4,0.12)",
    "Micro":   "rgba(220,38,38,0.12)",
    "Unknown": "rgba(156,163,175,0.12)",
}

    for c in cols:
        vals   = df[c].tolist()
        colors = []
        formatted = []
        for v in vals:
            if c in ("mcap_cr",):
                formatted.append(f"{v:,.0f}" if pd.notna(v) else "N/A")
                colors.append("#f0f4f8")
            elif c == "score":
                formatted.append(f"{v:.0f}" if pd.notna(v) else "N/A")
                colors.append("#dcfce7" if pd.notna(v) and v >= 70
                              else "#fef9c3" if pd.notna(v) and v >= 50 else "#fee2e2")
            elif c == "rsi":
                formatted.append(f"{v:.0f}" if pd.notna(v) else "N/A")
                colors.append("#dcfce7" if pd.notna(v) and 55<=v<=75
                              else "#fef9c3" if pd.notna(v) and 45<=v<55 else "#f9fafb")
            elif c in ("dist_52w_pct","dist_from_high_pct","dist_from_ath_pct"):
                formatted.append(f"{v:.1f}%" if pd.notna(v) else "N/A")
                colors.append("#dcfce7" if pd.notna(v) and v >= -1 else
                              "#fef9c3" if pd.notna(v) and v >= -5 else "#f9fafb")
            elif c in ("turnover_ratio","vol_ratio"):
                formatted.append(f"{v:.1f}×" if pd.notna(v) else "N/A")
                colors.append("#dbeafe" if pd.notna(v) and v >= 2 else "#f9fafb")
            elif c in ("last_price","ema10","ema20","ema50","ema200","prev_high","turnover_cr"):
                formatted.append(f"{v:,.2f}" if pd.notna(v) else "N/A")
                colors.append("#f9fafb")
            elif c == "cap_tier":
                formatted.append(str(v) if pd.notna(v) else "N/A")
                CAP_TINT = {
                    "Mega":    "rgba(124,58,237,0.12)",
                    "Large":   "rgba(37,99,235,0.12)",
                    "Mid":     "rgba(22,163,74,0.12)",
                    "Small":   "rgba(202,138,4,0.12)",
                    "Micro":   "rgba(220,38,38,0.12)",
                    "Unknown": "rgba(156,163,175,0.12)",
                }
                colors.append(CAP_TINT.get(str(v), "#f9fafb"))
            else:
                formatted.append(str(v) if pd.notna(v) else "N/A")
                colors.append("#f9fafb" if vals.index(v) % 2 == 0 else "#ffffff")
        cell_vals.append(formatted)
        cell_colors.append(colors)

    fig = go.Figure(go.Table(
        columnwidth=[40,80,160,120,60,90,80,60,70]+[80]*(len(cols)-9),
        header=dict(
            values=headers,
            fill_color="#1e3a5f",
            font=dict(color="white", size=11),
            align="center", height=36,
        ),
        cells=dict(
            values=cell_vals,
            fill_color=cell_colors,
            font=dict(size=11),
            align=["center","left","left","left","center"]+["center"]*(len(cols)-5),
            height=32,
        ),
    ))
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        margin=dict(l=0,r=0,t=8,b=0),
        height=max(420, len(df)*34 + 70),
    )
    return fig


def _sector_bar(df: pd.DataFrame) -> go.Figure:
    """Bar chart: count of passing stocks per sector."""
    if df.empty or "sector" not in df.columns:
        return go.Figure()
    counts = df["sector"].value_counts().sort_values(ascending=True)
    fig = go.Figure(go.Bar(
        x=counts.values, y=counts.index, orientation="h",
        marker=dict(color="#2563eb", opacity=0.75),
        text=counts.values, textposition="outside",
    ))
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        height=max(280, len(counts)*28+60),
        margin=dict(l=160,r=60,t=20,b=20),
        xaxis_title="Stocks passing", yaxis_title="",
        font=dict(size=11),
    )
    return fig


def _score_hist(df: pd.DataFrame) -> go.Figure:
    if df.empty or "score" not in df.columns:
        return go.Figure()
    fig = px.histogram(
        df, x="score", nbins=20,
        color_discrete_sequence=["#2563eb"],
    )
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        height=260, margin=dict(l=10,r=10,t=20,b=30),
        xaxis_title="Score", yaxis_title="Count",
        bargap=0.1,
    )
    return fig


# ── Main page renderer ────────────────────────────────────────────────────────

def render_screener_page(universe_df_full: pd.DataFrame = None):
    st.markdown("# 🔍 Stock Screener")
    st.caption("Run any screener on the NSE universe — filter by sector, index, and cap tier")

    if universe_df_full is None:
        with st.spinner("Loading universe…"):
            universe_df_full = build_universe()

    # ── Sidebar controls ──────────────────────────────────────────────────────
    with st.sidebar:
        st.markdown("## 🔍 Screener Controls")
        st.markdown("---")

        # Screener choice
        screener_name = st.selectbox(
            "Select Screener",
            list(SCREENER_REGISTRY.keys()),
        )
        cfg = SCREENER_REGISTRY[screener_name]
        st.caption(cfg["desc"])
        st.markdown("---")

        # Universe filter
        st.markdown("### Universe")
        nifty_filter = st.selectbox("NIFTY Index", INDEX_LIST, index=0)
        cap_filter   = st.multiselect(
            "Cap Tier", CAP_TIER_ORDER[:-1],
            default=["Large","Mid","Small"],
        )
        mcap_min = st.number_input("Min MCap (Cr)", value=1000, step=500)
        mcap_max = st.number_input("Max MCap (Cr)", value=200000, step=5000)

        # Sector filter
        sector_filter = st.multiselect(
            "Sectors (blank = all)",
            SECTOR_LIST_ALL[1:],  # exclude "All Sectors" label
            default=[],
            placeholder="All sectors",
        )
        st.markdown("---")

        # Dynamic screener params
        extra_kwargs = {}
        if cfg["params"]:
            st.markdown("### Screener Parameters")
            for param_key, pdef in cfg["params"].items():
                # Skip mcap params — handled by universe filter above
                if param_key in ("mcap_min_cr","mcap_max_cr"):
                    extra_kwargs[param_key] = (
                        mcap_min if param_key == "mcap_min_cr" else mcap_max
                    )
                    continue
                if pdef["type"] == "slider":
                    if isinstance(pdef["default"], float):
                        extra_kwargs[param_key] = st.slider(
                            pdef["label"],
                            float(pdef.get("min",0)), float(pdef.get("max",100)),
                            float(pdef["default"]), step=0.5,
                        )
                    else:
                        extra_kwargs[param_key] = st.slider(
                            pdef["label"],
                            int(pdef.get("min",0)), int(pdef.get("max",100)),
                            int(pdef["default"]),
                        )
                elif pdef["type"] == "number":
                    extra_kwargs[param_key] = st.number_input(
                        pdef["label"], value=pdef["default"],
                        step=pdef.get("step",1),
                    )
        st.markdown("---")

        run_btn = st.button("▶️ Run Screener", use_container_width=True, type="primary")

    # ── Apply universe filters ────────────────────────────────────────────────
    nifty_f = None if nifty_filter == "All" else nifty_filter
    universe_filtered = filter_universe(
        universe_df_full,
        cap_tiers   = cap_filter or None,
        mcap_min_cr = mcap_min,
        mcap_max_cr = mcap_max,
        sectors     = sector_filter if sector_filter else None,
        nifty_index = nifty_f,
    )

    st.info(
        f"🔍 Screener: **{screener_name}**  ·  "
        f"Universe: **{len(universe_filtered):,}** stocks after filters"
    )

    if not run_btn:
        st.markdown("""
        <div style="text-align:center;padding:60px 0;color:#9ca3af;">
            <div style="font-size:48px">🔍</div>
            <div style="font-size:16px;margin-top:8px">
                Configure filters in the sidebar and click <b>▶️ Run Screener</b>
            </div>
        </div>
        """, unsafe_allow_html=True)
        _show_screener_info()
        return

    if universe_filtered.empty:
        st.warning("No stocks match the current filters. Please relax the criteria.")
        return

    # ── Run ───────────────────────────────────────────────────────────────────
    prog = st.progress(0, text="Initialising screener…")

    def progress_cb(done, total, msg):
        pct = int(done / total * 100) if total > 0 else 0
        prog.progress(min(pct, 99), text=f"{msg} ({done}/{total})")

    with st.spinner(f"Running {screener_name}…"):
        try:
            fn = cfg["fn"]
            result = fn(universe_filtered, progress_cb=progress_cb, **extra_kwargs)
            prog.progress(100, text="Done ✅")
            prog.empty()
            ok = True
        except Exception as e:
            prog.empty()
            st.error(f"Screener error: {e}")
            ok = False

    if not ok:
        return

    # ── Results ───────────────────────────────────────────────────────────────
    if result.empty:
        st.warning("No stocks passed all the screener conditions. Try relaxing the parameters.")
        return

    # Summary metrics
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("✅ Stocks Passed",   len(result))
    m2.metric("📊 Universe Scanned", len(universe_filtered))
    m3.metric("📈 Pass Rate",        f"{len(result)/len(universe_filtered)*100:.1f}%")
    avg_score = result["score"].mean() if "score" in result.columns else 0
    m4.metric("⭐ Avg Score",        f"{avg_score:.1f}")

    st.markdown("---")

    # Tabs
    tab1, tab2, tab3, tab4 = st.tabs(["📋 Results Table","📊 Sector Breakdown","🎯 Score Distribution","💾 Export"])

    with tab1:
        st.plotly_chart(_make_result_table(result), use_container_width=True)

    with tab2:
        c1, c2 = st.columns([2,1])
        with c1:
            st.plotly_chart(_sector_bar(result), use_container_width=True)
        with c2:
            if "cap_tier" in result.columns:
                ct = result["cap_tier"].value_counts()
                fig_pie = px.pie(
                    values=ct.values, names=ct.index,
                    color=ct.index,
                    color_discrete_map=CAP_COLOR_CSS,
                    hole=0.4,
                )
                fig_pie.update_layout(
                    paper_bgcolor="rgba(0,0,0,0)",
                    margin=dict(l=0,r=0,t=20,b=0), height=280,
                    legend=dict(font=dict(size=10)),
                )
                st.plotly_chart(fig_pie, use_container_width=True)

    with tab3:
        st.plotly_chart(_score_hist(result), use_container_width=True)
        if "sector" in result.columns and "score" in result.columns:
            sec_avg = result.groupby("sector")["score"].mean().sort_values(ascending=False)
            st.markdown("**Average Score by Sector**")
            st.dataframe(
                sec_avg.reset_index().rename(columns={"score":"Avg Score","sector":"Sector"}),
                use_container_width=True, hide_index=True,
            )

    with tab4:
        csv = result.to_csv(index=False)
        fname = f"{cfg['key']}_screener_{pd.Timestamp.now().strftime('%Y%m%d_%H%M')}.csv"
        st.download_button(
            "📥 Download CSV", csv, fname, "text/csv",
            use_container_width=True,
        )
        st.caption(f"{len(result)} stocks · Generated {pd.Timestamp.now().strftime('%d %b %Y %H:%M IST')}")


def _show_screener_info():
    """Show screener info cards when not yet run."""
    st.markdown("### Available Screeners")
    cols = st.columns(3)
    for i, (name, cfg) in enumerate(SCREENER_REGISTRY.items()):
        with cols[i % 3]:
            st.markdown(f"""
            <div style="background:#f8fafc;border:1px solid #e2e8f0;
                border-radius:10px;padding:14px;margin-bottom:12px;min-height:120px">
              <div style="font-size:15px;font-weight:600">{name}</div>
              <div style="font-size:12px;color:#6b7280;margin-top:6px;line-height:1.5">
                {cfg["desc"][:160]}{"…" if len(cfg["desc"])>160 else ""}
              </div>
            </div>
            """, unsafe_allow_html=True)
