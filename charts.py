"""
charts.py  (v2 — RRG shows days-in-quadrant badge)
All Plotly chart builders for the sector rotation dashboard.
"""
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np

QUADRANT_COLORS = {
    "Leading":   "#16a34a",
    "Improving": "#2563eb",
    "Weakening": "#d97706",
    "Lagging":   "#dc2626",
}
SECTOR_COLORS = px.colors.qualitative.Light24

_BASE = dict(
    plot_bgcolor  = "rgba(0,0,0,0)",
    paper_bgcolor = "rgba(0,0,0,0)",
    font          = dict(family="Inter, sans-serif", size=12, color="#374151"),
    legend        = dict(orientation="v", x=1.01, y=1, font=dict(size=10)),
    xaxis         = dict(showgrid=True, gridcolor="rgba(0,0,0,0.06)", zeroline=False),
    yaxis         = dict(showgrid=True, gridcolor="rgba(0,0,0,0.06)", zeroline=False),
)

def _layout(fig, margin=None, **kwargs):
    merged = dict(**_BASE)
    merged.update(kwargs)
    merged["margin"] = margin or dict(l=10, r=10, t=40, b=10)
    fig.update_layout(**merged)
    return fig

# ── 1. Performance Table ──────────────────────────────────────────────────────

def make_performance_table(returns, rel_returns, benchmark_name, sector_names, timeframe="1M"):
    sectors    = [s for s in sector_names if s in returns.index]
    cols       = ["1W","1M","3M","6M","YTD","1Y"]
    avail_cols = [c for c in cols if c in returns.columns]
    abs_vals   = returns.loc[sectors, avail_cols].copy()
    rel_vals   = rel_returns.loc[sectors, avail_cols].copy()
    if timeframe in abs_vals.columns:
        abs_vals = abs_vals.sort_values(timeframe, ascending=False)
        rel_vals = rel_vals.loc[abs_vals.index]
    cell_text, cell_colors = [], []
    for col in avail_cols:
        texts, colors = [], []
        for sec in abs_vals.index:
            a = abs_vals.loc[sec, col]
            r = rel_vals.loc[sec, col]
            if pd.isna(a):
                texts.append("N/A"); colors.append("#f9fafb")
            else:
                sign    = "+" if (pd.notna(r) and r >= 0) else ""
                rel_str = f"{sign}{r:.1f}%" if pd.notna(r) else "N/A"
                texts.append(f"{a:+.1f}%\n({rel_str})")
                colors.append(_heatmap_color(r if pd.notna(r) else 0, vmin=-15, vmax=15))
        cell_text.append(texts); cell_colors.append(colors)
    if timeframe in abs_vals.columns:
        ranks = [str(i+1) for i in range(len(abs_vals))]
    else:
        ranks = ["—"] * len(abs_vals)
    sector_labels = [f"#{r} {s}" for r, s in zip(ranks, abs_vals.index)]
    fig = go.Figure(go.Table(
        columnwidth=[220] + [110]*len(avail_cols),
        header=dict(values=["Sector"] + avail_cols,
                    fill_color="#1e3a5f", font=dict(color="white", size=12),
                    align="center", height=36),
        cells=dict(values=[sector_labels]+cell_text,
                   fill_color=["#f0f4f8"]+cell_colors,
                   font=dict(size=11), align=["left"]+["center"]*len(avail_cols), height=40),
    ))
    fig.update_layout(paper_bgcolor="rgba(0,0,0,0)", margin=dict(l=0,r=0,t=10,b=0),
                      height=max(500, len(sectors)*42+60))
    return fig

def _heatmap_color(val, vmin=-20, vmax=20):
    if pd.isna(val): return "#f9fafb"
    norm = max(0.0, min(1.0, (val-vmin)/(vmax-vmin)))
    if norm >= 0.5:
        t = (norm-0.5)*2
        r = int(255*(1-t)+22*t); g = int(255*(1-t)+163*t); b = int(255*(1-t)+74*t)
    else:
        t = norm*2
        r = int(220*(1-t)+255*t); g = int(38*(1-t)+255*t);  b = int(38*(1-t)+255*t)
    return f"rgb({r},{g},{b})"

# ── 2. Heat Map ───────────────────────────────────────────────────────────────

def make_heatmap(rel_returns, sector_names, benchmark_name):
    sectors = [s for s in sector_names if s in rel_returns.index]
    cols    = ["1W","1M","3M","6M","YTD","1Y"]
    avail   = [c for c in cols if c in rel_returns.columns]
    df      = rel_returns.loc[sectors, avail].copy()
    if "1M" in df.columns:
        df = df.sort_values("1M", ascending=True)
    z    = df.values.tolist()
    text = [[f"{v:+.1f}%" if (v is not None and not (isinstance(v,float) and np.isnan(v))) else "N/A"
             for v in row] for row in z]
    fig = go.Figure(go.Heatmap(
        z=z, x=avail, y=df.index.tolist(), text=text, texttemplate="%{text}",
        textfont=dict(size=10),
        colorscale=[[0.0,"#dc2626"],[0.35,"#fca5a5"],[0.5,"#f9fafb"],[0.65,"#86efac"],[1.0,"#16a34a"]],
        zmid=0, colorbar=dict(title="vs NIFTY", thickness=12, len=0.8),
        hovertemplate="%{y}<br>Period: %{x}<br>vs NIFTY: %{text}",
    ))
    _layout(fig, margin=dict(l=160,r=80,t=20,b=40),
            height=max(450, len(sectors)*26+60), xaxis_title="Timeframe", yaxis_title="")
    return fig

# ── 3. Normalized Line Chart ──────────────────────────────────────────────────

def make_normalized_chart(norm_prices, benchmark_name, selected_sectors=None, title=""):
    fig      = go.Figure()
    all_cols = norm_prices.columns.tolist()
    plot_cols = [c for c in (selected_sectors or all_cols) if c in all_cols and c != benchmark_name]
    if benchmark_name in norm_prices.columns:
        bm = norm_prices[benchmark_name].dropna()
        fig.add_trace(go.Scatter(x=bm.index, y=bm.values, name=benchmark_name,
                                 line=dict(color="#6b7280", width=2, dash="dash"),
                                 hovertemplate=f"{benchmark_name}: %{{y:.1f}}<br>%{{x|%d %b}}"))
    for i, col in enumerate(plot_cols):
        series = norm_prices[col].dropna()
        if series.empty: continue
        fig.add_trace(go.Scatter(x=series.index, y=series.values,
                                 name=f"{col} ({series.iloc[-1]:.1f})",
                                 line=dict(color=SECTOR_COLORS[i%len(SECTOR_COLORS)], width=1.8),
                                 hovertemplate=f"{col}: %{{y:.1f}}<br>%{{x|%d %b}}"))
    fig.add_hline(y=100, line_dash="dot", line_color="rgba(0,0,0,0.2)", line_width=1)
    _layout(fig, height=420, xaxis_title="", yaxis_title="Normalized (base=100)",
            title=dict(text=title, font=dict(size=13)), hovermode="x unified",
            legend=dict(orientation="v", x=1.02, y=1, font=dict(size=9), itemsizing="constant"))
    return fig

# ── 4. Relative Strength Line Chart ──────────────────────────────────────────

def make_rs_chart(rs_data, selected_sectors=None):
    fig  = go.Figure()
    cols = [c for c in (selected_sectors or rs_data.columns.tolist()) if c in rs_data.columns]
    for i, col in enumerate(cols):
        series = rs_data[col].dropna()
        if series.empty: continue
        fig.add_trace(go.Scatter(x=series.index, y=series.values,
                                 name=f"{col} ({series.iloc[-1]:.1f})",
                                 line=dict(color=SECTOR_COLORS[i%len(SECTOR_COLORS)], width=1.8),
                                 hovertemplate=f"{col}: %{{y:.1f}}<br>%{{x|%d %b}}"))
    fig.add_hline(y=100, line_dash="dot", line_color="rgba(0,0,0,0.3)", line_width=1.5,
                  annotation_text="Benchmark (NIFTY)", annotation_position="bottom right")
    _layout(fig, height=420, xaxis_title="", yaxis_title="Relative Strength (base=100)",
            hovermode="x unified", legend=dict(orientation="v", x=1.02, y=1, font=dict(size=9)))
    return fig

# ── 5. RRG Chart (with days-in-quadrant badge) ────────────────────────────────

def make_rrg_chart(rrg: pd.DataFrame, rrg_days: pd.DataFrame = None) -> go.Figure:
    fig = go.Figure()
    if not rrg.empty:
        xmin = min(85, rrg["RS_Ratio"].min()-2);   xmax = max(115, rrg["RS_Ratio"].max()+2)
        ymin = min(-10, rrg["RS_Momentum"].min()-1); ymax = max(10, rrg["RS_Momentum"].max()+1)
    else:
        xmin, xmax, ymin, ymax = 85, 115, -10, 10

    quads = [
        ("Leading",   dict(x0=100,x1=xmax,y0=0,y1=ymax),  "rgba(22,163,74,0.07)"),
        ("Weakening", dict(x0=100,x1=xmax,y0=ymin,y1=0),  "rgba(217,119,6,0.07)"),
        ("Improving", dict(x0=xmin,x1=100,y0=0,y1=ymax),  "rgba(37,99,235,0.07)"),
        ("Lagging",   dict(x0=xmin,x1=100,y0=ymin,y1=0),  "rgba(220,38,38,0.07)"),
    ]
    for _, coords, color in quads:
        fig.add_shape(type="rect", **coords, fillcolor=color, line=dict(width=0), layer="below")
    fig.add_hline(y=0, line_color="rgba(0,0,0,0.2)", line_width=1)
    fig.add_vline(x=100, line_color="rgba(0,0,0,0.2)", line_width=1)
    for label, x, y in [("LEADING",xmax-1,ymax-0.5),("WEAKENING",xmax-1,ymin+0.5),
                          ("IMPROVING",xmin+1,ymax-0.5),("LAGGING",xmin+1,ymin+0.5)]:
        fig.add_annotation(x=x, y=y, text=label, showarrow=False,
                           font=dict(size=9, color=QUADRANT_COLORS.get(label.title(),"#6b7280")),
                           opacity=0.6)

    for quadrant, color in QUADRANT_COLORS.items():
        sub = rrg[rrg["Quadrant"] == quadrant]
        if sub.empty: continue

        # Build hover text with days-in-quadrant
        hover_texts = []
        labels      = []
        for sector in sub.index:
            days_str = ""
            if rrg_days is not None and sector in rrg_days.index:
                d = rrg_days.loc[sector, "Days_In_Quadrant"]
                days_str = f"<br>📅 {d}d in {quadrant}"
            hover_texts.append(
                f"<b>{sector}</b>"
                f"<br>RS-Ratio: {sub.loc[sector,'RS_Ratio']:.1f}"
                f"<br>RS-Mom: {sub.loc[sector,'RS_Momentum']:.2f}%"
                f"{days_str}"
            )
            if rrg_days is not None and sector in rrg_days.index:
                d = rrg_days.loc[sector, "Days_In_Quadrant"]
                labels.append(f"{sector}\n({d}d)")
            else:
                labels.append(sector)

        fig.add_trace(go.Scatter(
            x=sub["RS_Ratio"], y=sub["RS_Momentum"],
            mode="markers+text",
            name=quadrant,
            marker=dict(color=color, size=12, line=dict(color="white", width=1.5)),
            text=labels,
            textposition="top center",
            textfont=dict(size=9),
            hovertext=hover_texts,
            hoverinfo="text",
        ))

    _layout(fig, height=520,
            xaxis=dict(title="RS-Ratio (strength vs benchmark)", range=[xmin,xmax],
                       showgrid=True, gridcolor="rgba(0,0,0,0.05)"),
            yaxis=dict(title="RS-Momentum (trend direction)", range=[ymin,ymax],
                       showgrid=True, gridcolor="rgba(0,0,0,0.05)"),
            legend=dict(orientation="h", y=-0.12, x=0.5, xanchor="center"))
    return fig

# ── 6. Rank Shift ─────────────────────────────────────────────────────────────

def make_rank_shift_chart(rank_shift):
    df = rank_shift.copy()
    df = df[df["Shift"].notna()].copy()
    df["Shift"] = df["Shift"].astype(float)
    df = df.sort_values("Shift", ascending=True)
    if df.empty: return go.Figure()
    colors = ["#16a34a" if v >= 0 else "#dc2626" for v in df["Shift"]]
    def _fmt(v):
        try: return f"#{int(v)}" if pd.notna(v) else "N/A"
        except: return "N/A"
    fig = go.Figure(go.Bar(
        y=df.index, x=df["Shift"].astype(float), orientation="h",
        marker=dict(color=colors),
        text=[_fmt(r) for r in df["Rank_Now"]], textposition="outside",
        hovertemplate="%{y}<br>Rank Now: %{text}<br>Rank 4W Ago: %{customdata}<br>Shift: %{x:+.0f}",
        customdata=[_fmt(r) for r in df["Rank_4W_Ago"]],
    ))
    fig.add_vline(x=0, line_color="rgba(0,0,0,0.3)", line_width=1)
    _layout(fig, margin=dict(l=160,r=60,t=20,b=40),
            height=max(400,len(df)*26+80), xaxis_title="Rank Change (positive=improved)",
            yaxis_title="", bargap=0.3)
    return fig

# ── 7. Volatility ─────────────────────────────────────────────────────────────

def make_volatility_chart(volatility, benchmark_name, sector_names):
    sectors = [s for s in sector_names if s in volatility.index and pd.notna(volatility[s])]
    df      = volatility[sectors].sort_values(ascending=False)
    bm_vol  = float(volatility.get(benchmark_name, np.nan))
    colors  = ["#1d4ed8" if (np.isfinite(bm_vol) and v > bm_vol) else "#64748b" for v in df.values]
    fig = go.Figure(go.Bar(
        x=df.index, y=df.values, marker=dict(color=colors),
        text=[f"{v:.1f}%" for v in df.values], textposition="outside",
        hovertemplate="%{x}<br>Volatility: %{y:.1f}%",
    ))
    if np.isfinite(bm_vol):
        fig.add_hline(y=bm_vol, line_dash="dash", line_color="#dc2626", line_width=1.5,
                      annotation_text=f"NIFTY: {bm_vol:.1f}%", annotation_position="top right")
    _layout(fig, height=380, xaxis_title="", yaxis_title="Annualised Volatility (%)",
            xaxis_tickangle=-35, bargap=0.3)
    return fig

# ── 8. Drawdown ───────────────────────────────────────────────────────────────

def make_drawdown_chart(drawdown, benchmark_name, sector_names):
    sectors = [s for s in sector_names if s in drawdown.index and pd.notna(drawdown[s])]
    df      = drawdown[sectors].sort_values(ascending=True)
    colors  = ["#dc2626" if v < -20 else "#f97316" if v < -10 else "#facc15" if v < -5
               else "#86efac" for v in df.values]
    fig = go.Figure(go.Bar(
        x=df.index, y=df.values, marker=dict(color=colors),
        text=[f"{v:.1f}%" for v in df.values], textposition="outside",
        hovertemplate="%{x}<br>Max Drawdown: %{y:.1f}%",
    ))
    _layout(fig, height=380, xaxis_title="", yaxis_title="Max Drawdown YTD (%)",
            xaxis_tickangle=-35, bargap=0.3)
    return fig
