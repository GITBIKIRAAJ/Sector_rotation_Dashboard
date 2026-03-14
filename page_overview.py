"""
page_overview.py  (v3 — removed PE/FII/ATH, unique keys, theme-safe)
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from overview_engine import load_overview_data

CARD_CSS = """
<style>
.ov-card { border:1px solid rgba(148,163,184,0.2); border-radius:12px;
           padding:16px 18px; text-align:center; min-height:130px; margin-bottom:4px; }
.ov-badge { padding:2px 10px; border-radius:10px;
            font-size:11px; font-weight:600; color:#fff; }
.ov-bar-row { display:flex; align-items:center; margin-bottom:8px; }
.ov-bar-label { width:140px; font-size:13px; }
.ov-bar-track { background:rgba(148,163,184,0.15); border-radius:4px;
                flex:1; height:10px; margin:0 10px; }
.ov-bar-fill  { height:10px; border-radius:4px; }
.ov-bar-val   { font-size:13px; font-weight:700; width:75px; text-align:right; }
.ov-sma-row   { margin-bottom:7px; }
.ov-sma-hdr   { display:flex; justify-content:space-between;
                font-size:12px; margin-bottom:2px; }
.ov-sma-track { background:rgba(148,163,184,0.15); border-radius:3px; height:6px; }
.ov-sma-fill  { height:6px; border-radius:3px; }
.ov-summary   { border:1px solid rgba(148,163,184,0.2); border-radius:10px;
                padding:16px 20px; line-height:1.9; font-size:14px; }
</style>
"""

CAP_TINT = {
    "Mega":    "rgba(124,58,237,0.18)",
    "Large":   "rgba(37,99,235,0.18)",
    "Mid":     "rgba(22,163,74,0.18)",
    "Small":   "rgba(202,138,4,0.18)",
    "Micro":   "rgba(220,38,38,0.18)",
    "Unknown": "rgba(156,163,175,0.15)",
}

def _cc(v):
    if v is None or (isinstance(v, float) and np.isnan(v)): return "#9ca3af"
    return "#16a34a" if v >= 0 else "#dc2626"

def _cs(v):
    if v is None or (isinstance(v, float) and np.isnan(v)): return "—"
    return ("▲ " if v >= 0 else "▼ ") + str(abs(round(v, 2))) + "%"


# ── 1. Index cards ────────────────────────────────────────────────────────────

def _render_index_cards(snap: dict):
    st.markdown("### 📌 Index Snapshot")
    if not snap:
        st.info("Index data unavailable.")
        return
    cols = st.columns(len(snap))
    for i, (name, d) in enumerate(snap.items()):
        chg = d["change_pct"]
        col = _cc(chg)
        with cols[i]:
            st.markdown(f"""
            <div class="ov-card">
              <div style="color:#94a3b8;font-size:13px;font-weight:600">{name}</div>
              <div style="font-size:26px;font-weight:800;color:{col};margin:6px 0">
                {d["last_price"]:,.2f}
              </div>
              <div style="color:{col};font-size:15px;font-weight:700">{_cs(chg)}</div>
              <div style="margin-top:8px">
                <span class="ov-badge" style="background:{d["tag_color"]}">{d["trend_tag"]}</span>
              </div>
              <div style="color:#64748b;font-size:11px;margin-top:4px">
                Prev {d["prev_close"]:,.2f} &nbsp;·&nbsp; EMA20 {d["ema20"]:,.2f}
              </div>
            </div>
            """, unsafe_allow_html=True)

    sp_cols = st.columns(len(snap))
    for i, (name, d) in enumerate(snap.items()):
        spark = d.get("sparkline", [])
        if len(spark) < 2:
            continue
        col  = "#16a34a" if spark[-1] >= spark[0] else "#dc2626"
        fill = "rgba(22,163,74,0.1)" if col == "#16a34a" else "rgba(220,38,38,0.1)"
        fig  = go.Figure(go.Scatter(
            y=spark, mode="lines",
            line=dict(color=col, width=2),
            fill="tozeroy", fillcolor=fill,
        ))
        fig.update_layout(
            height=55, margin=dict(l=0,r=0,t=0,b=0),
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
            xaxis=dict(visible=False), yaxis=dict(visible=False),
            showlegend=False,
        )
        with sp_cols[i]:
            st.plotly_chart(fig, use_container_width=True,
                            config={"displayModeBar": False},
                            key=f"spark_{i}_{name.replace(' ','_')}")


# ── 2. Breadth ────────────────────────────────────────────────────────────────

def _render_breadth(breadth: dict, snap: dict):
    st.markdown("### 📊 Yesterday Performance & Market Breadth")
    left, right = st.columns(2)

    with left:
        st.markdown("**Yesterday Returns**")
        for name, d in snap.items():
            chg   = d["change_pct"]
            color = _cc(chg)
            bar_w = min(abs(chg) * 10, 100)
            st.markdown(f"""
            <div class="ov-bar-row">
              <div class="ov-bar-label">{name}</div>
              <div class="ov-bar-track">
                <div class="ov-bar-fill" style="background:{color};width:{bar_w}%"></div>
              </div>
              <div class="ov-bar-val" style="color:{color}">{_cs(chg)}</div>
            </div>
            """, unsafe_allow_html=True)

    with right:
        if not breadth:
            st.info("Breadth data unavailable.")
            return
        bp    = breadth.get("bullish_pct", 0)
        adv   = breadth.get("advances", 0)
        dec   = breadth.get("declines", 0)
        unch  = breadth.get("unchanged", 0)
        total = breadth.get("total", 1) or 1

        c1, c2 = st.columns(2)
        with c1:
            fig_d = go.Figure(go.Pie(
                values=[max(adv,0), max(dec,0), max(unch,0)],
                labels=["Advances","Declines","Unchanged"],
                marker_colors=["#16a34a","#dc2626","#9ca3af"],
                hole=0.65, textinfo="none",
                hovertemplate="%{label}: %{value}<extra></extra>",
            ))
            fig_d.add_annotation(
                text=f"<b>{bp:.0f}%</b><br>Bullish",
                x=0.5, y=0.5, font_size=16, showarrow=False,
            )
            fig_d.update_layout(
                paper_bgcolor="rgba(0,0,0,0)",
                margin=dict(l=0,r=0,t=24,b=0), height=210,
                showlegend=False,
                title=dict(text="Market Breadth", font=dict(size=13)),
            )
            st.plotly_chart(fig_d, use_container_width=True,
                            config={"displayModeBar": False},
                            key="breadth_donut")
            st.markdown(
                f'<div style="text-align:center;font-size:13px">'
                f'<span style="color:#16a34a;font-weight:700">{adv} ▲</span>&nbsp;'
                f'<span style="color:#dc2626;font-weight:700">{dec} ▼</span>&nbsp;'
                f'<span style="color:#9ca3af">{unch} —</span>'
                f'</div>', unsafe_allow_html=True
            )

        with c2:
            st.markdown("**% Above Moving Avg**")
            for label, info in breadth.get("above_sma", {}).items():
                pct   = info.get("pct", 0)
                cnt   = info.get("count", 0)
                tot   = info.get("total", total)
                bw    = min(pct, 100)
                color = "#dc2626" if pct < 35 else "#ca8a04" if pct < 55 else "#16a34a"
                st.markdown(f"""
                <div class="ov-sma-row">
                  <div class="ov-sma-hdr">
                    <span>{label}</span>
                    <span style="color:{color};font-weight:600">{cnt}/{tot}&nbsp;{pct:.1f}%</span>
                  </div>
                  <div class="ov-sma-track">
                    <div class="ov-sma-fill" style="background:{color};width:{bw}%"></div>
                  </div>
                </div>
                """, unsafe_allow_html=True)


# ── 3. Top Movers (Gainers / Losers / Near 52W High) ─────────────────────────

def _make_table(rows: list, dist_col: str = None) -> go.Figure:
    if not rows:
        fig = go.Figure()
        fig.update_layout(
            paper_bgcolor="rgba(0,0,0,0)", height=200,
            annotations=[dict(text="No data available", showarrow=False,
                              font=dict(size=14, color="#94a3b8"),
                              xref="paper", yref="paper", x=0.5, y=0.5)]
        )
        return fig

    df   = pd.DataFrame(rows)
    cols = ["stock","company","sector","cap_tier","price"]
    if "chg_pct"  in df.columns: cols.append("chg_pct")
    if dist_col and dist_col in df.columns: cols.append(dist_col)
    if "high_52w" in df.columns and "high_52w" not in cols: cols.append("high_52w")
    cols = [c for c in cols if c in df.columns]

    HDR = {"stock":"Stock","company":"Company","sector":"Sector","cap_tier":"Cap",
           "price":"Price","chg_pct":"Chg %","dist_pct":"Dist %","high_52w":"52W High"}

    cell_vals, cell_colors = [], []
    for c in cols:
        vals = df[c].tolist()
        fmt, clr = [], []
        for v in vals:
            if c == "chg_pct":
                fmt.append(f"{float(v):+.2f}%" if pd.notna(v) else "N/A")
                clr.append("#dcfce7" if pd.notna(v) and float(v) > 0 else "#fee2e2")
            elif c == "dist_pct":
                fmt.append(f"{float(v):.2f}%" if pd.notna(v) else "N/A")
                clr.append("#dcfce7" if pd.notna(v) and float(v) >= -1
                           else "#fef9c3" if pd.notna(v) and float(v) >= -5
                           else "rgba(249,250,251,0.5)")
            elif c in ("price","high_52w"):
                fmt.append(f"{float(v):,.2f}" if pd.notna(v) else "N/A")
                clr.append("rgba(249,250,251,0.5)")
            elif c == "cap_tier":
                fmt.append(str(v) if pd.notna(v) else "N/A")
                clr.append(CAP_TINT.get(str(v), "rgba(156,163,175,0.15)"))
            else:
                fmt.append(str(v) if pd.notna(v) else "N/A")
                clr.append("rgba(249,250,251,0.5)")
        cell_vals.append(fmt)
        cell_colors.append(clr)

    fig = go.Figure(go.Table(
        columnwidth=[70,160,120,60,80] + [70]*(len(cols)-5),
        header=dict(values=[HDR.get(c,c) for c in cols],
                    fill_color="#1e3a5f",
                    font=dict(color="white", size=11),
                    align="center", height=32),
        cells=dict(values=cell_vals, fill_color=cell_colors,
                   font=dict(size=11, color="#111827"),
                   align=["left","left","left","center"]+["right"]*(len(cols)-4),
                   height=28),
    ))
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        margin=dict(l=0,r=0,t=4,b=0),
        height=max(300, len(rows)*30 + 52),
    )
    return fig


def _render_movers(movers: dict):
    st.markdown("### 🚀 Top Movers & Breakouts")
    tab1, tab2, tab3 = st.tabs(["📈 Top Gainers","📉 Top Losers","🏔️ Near 52W High"])
    with tab1:
        st.plotly_chart(_make_table(movers.get("gainers",[])),
                        use_container_width=True, key="tbl_gainers")
    with tab2:
        st.plotly_chart(_make_table(movers.get("losers",[])),
                        use_container_width=True, key="tbl_losers")
    with tab3:
        st.plotly_chart(_make_table(movers.get("near_52w",[]), dist_col="dist_pct"),
                        use_container_width=True, key="tbl_52w")


# ── 4. Sector Treemap ─────────────────────────────────────────────────────────

def _render_treemap(treemap_df: pd.DataFrame):
    st.markdown("### 🗺️ Sector Heatmap — Today")
    if treemap_df.empty:
        st.info("Sector data not available.")
        return
    fig = px.treemap(
        treemap_df, path=["sector"],
        values=[1]*len(treemap_df),
        color="change_pct",
        color_continuous_scale=[
            [0.0,"#7f1d1d"],[0.35,"#dc2626"],
            [0.5,"rgba(100,116,139,0.3)"],
            [0.65,"#16a34a"],[1.0,"#14532d"],
        ],
        color_continuous_midpoint=0,
        custom_data=["change_pct"],
    )
    fig.update_traces(
        texttemplate="<b>%{label}</b><br>%{customdata[0]:+.2f}%",
        textfont=dict(size=13, color="white"),
        hovertemplate="%{label}<br>%{customdata[0]:+.2f}%<extra></extra>",
    )
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        margin=dict(l=0,r=0,t=10,b=0), height=360,
        coloraxis_showscale=False,
    )
    st.plotly_chart(fig, use_container_width=True, key="sector_treemap")


# ── 5. Summary ────────────────────────────────────────────────────────────────

def _render_summary(summary: str, updated: str):
    st.markdown("### 💬 Market Summary")
    html = summary.replace("\n","<br>")
    st.markdown(f"""
    <div class="ov-summary">
      {html}
      <hr style="border-color:rgba(148,163,184,0.2);margin:10px 0 6px">
      <span style="color:#64748b;font-size:11px">Updated: {updated}</span>
    </div>
    """, unsafe_allow_html=True)


# ── Master render ─────────────────────────────────────────────────────────────

def render_overview_page(universe_df=None, prices_df=None, sectors=None):
    st.markdown(CARD_CSS, unsafe_allow_html=True)
    st.markdown("# 🏠 Market Overview")
    st.caption("Live NSE snapshot — indices, breadth, top movers, sector heatmap")

    prog = st.progress(0, text="Loading overview data...")
    def pcb(done, total, msg):
        prog.progress(min(int(done/total*100), 95), text=msg)

    with st.spinner("Fetching data..."):
        data = load_overview_data(
            universe_df=universe_df,
            prices_df=prices_df,
            sectors=sectors,
            progress_cb=pcb,
        )
    prog.progress(100, text="Done ✅")
    prog.empty()

    _render_summary(data["summary"], data["last_updated"])
    st.markdown("---")
    _render_index_cards(data["snap"])
    st.markdown("---")
    _render_breadth(data["breadth"], data["snap"])
    st.markdown("---")
    _render_movers(data["movers"])
    st.markdown("---")
    _render_treemap(data["treemap"])
    st.markdown("---")

    st.markdown("### ⚡ Quick Screeners")
    st.caption("Jump to a screener instantly")
    sc = [("📊 Turnover","📊 Turnover Screener"),
          ("🏔️ 52W High","🏔️ 52-Week High"),
          ("💥 Momentum","💥 Momentum Breakout"),
          ("🐂 Bull Trend","🐂 Bull Trend Setup")]
    qc = st.columns(4)
    for i,(label,val) in enumerate(sc):
        with qc[i]:
            if st.button(label, use_container_width=True, key=f"qs_{i}"):
                st.session_state["goto_screener"] = val
    if st.session_state.get("goto_screener"):
        st.info(f"Go to **Screener** page — '{st.session_state['goto_screener']}' pre-selected.")
