"""
stock_charts.py
Plotly chart builders for the stock ranker page.
"""
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
import numpy as np

SECTOR_COLORS = px.colors.qualitative.Light24

def _base_layout(fig, **kwargs):
    layout = dict(
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        font=dict(family="Inter, sans-serif", size=12, color="#374151"),
        margin=dict(l=10, r=10, t=36, b=10),
    )
    layout.update(kwargs)
    fig.update_layout(**layout)
    return fig

def make_score_bars(df, title="", top=True):
    if df.empty: return go.Figure()
    col    = "stock" if "stock" in df.columns else "ticker"
    df     = df.copy().sort_values("composite_score", ascending=True)
    colors = ["#16a34a" if top else "#dc2626"] * len(df)
    fig = go.Figure(go.Bar(
        y=df[col], x=df["composite_score"], orientation="h",
        marker=dict(color=colors, opacity=[0.4+0.6*(i/max(len(df)-1,1)) for i in range(len(df))]),
        text=[f"{v:.0f}" for v in df["composite_score"]], textposition="outside",
        hovertemplate="%{y}<br>Score: %{x:.1f}<extra></extra>",
    ))
    _base_layout(fig, margin=dict(l=100,r=60,t=36,b=10),
                 height=max(280,len(df)*36+60),
                 xaxis=dict(range=[0,105], title="Composite score",
                            showgrid=True, gridcolor="rgba(0,0,0,0.05)"),
                 yaxis_title="", title=dict(text=title, font=dict(size=13)))
    return fig

def make_radar_chart(row):
    categories = ["RS vs NIFTY","RS vs Sector","RSI strength","EMA align","52W proximity"]
    score_cols = ["rs_nifty_score","rs_sector_score","rsi_score","ema_norm","dist_score"]
    values = [float(row.get(c,50)) if pd.notna(row.get(c,50)) else 50 for c in score_cols]
    fig = go.Figure(go.Scatterpolar(
        r=values+[values[0]], theta=categories+[categories[0]],
        fill="toself", fillcolor="rgba(22,163,74,0.15)",
        line=dict(color="#16a34a", width=2),
        name=str(row.get("stock", row.get("ticker",""))),
    ))
    fig.update_layout(
        plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
        font=dict(family="Inter, sans-serif", size=11),
        margin=dict(l=30,r=30,t=30,b=30),
        polar=dict(
            radialaxis=dict(visible=True, range=[0,100],
                            tickvals=[25,50,75,100], tickfont=dict(size=9),
                            gridcolor="rgba(0,0,0,0.1)"),
            angularaxis=dict(tickfont=dict(size=10)),
            bgcolor="rgba(0,0,0,0)",
        ),
        showlegend=False, height=280,
    )
    return fig

def make_rank_table(df):
    if df.empty: return go.Figure()
    col          = "stock" if "stock" in df.columns else "ticker"
    display_cols = [col,"composite_score","rs_nifty","rs_sector",
                    "rsi","ema_label","dist_52w","return_1w","return_1m","return_3m"]
    headers      = ["Stock","Score","RS/NIFTY","RS/Sector","RSI","EMA","52W%","1W%","1M%","3M%"]
    avail        = [c for c in display_cols if c in df.columns]
    avail_headers= [headers[display_cols.index(c)] for c in avail]

    cell_vals, cell_colors_col = [], []
    for c in avail:
        vals = df[c].tolist()
        if c == "composite_score":
            cell_vals.append([f"{v:.0f}" if pd.notna(v) else "N/A" for v in vals])
            cell_colors_col.append([_score_color(v) for v in vals])
        elif c in ("rs_nifty","rs_sector"):
            cell_vals.append([f"{v:.1f}" if pd.notna(v) else "N/A" for v in vals])
            cell_colors_col.append([_rs_color(v) for v in vals])
        elif c == "rsi":
            cell_vals.append([f"{v:.0f}" if pd.notna(v) else "N/A" for v in vals])
            cell_colors_col.append([_rsi_color(v) for v in vals])
        elif c in ("return_1w","return_1m","return_3m"):
            cell_vals.append([f"{v:+.1f}%" if pd.notna(v) else "N/A" for v in vals])
            cell_colors_col.append([_return_color(v) for v in vals])
        elif c == "dist_52w":
            cell_vals.append([f"{v:.1f}%" if pd.notna(v) else "N/A" for v in vals])
            cell_colors_col.append([_return_color(v) for v in vals])
        else:
            cell_vals.append([str(v) if pd.notna(v) else "N/A" for v in vals])
            cell_colors_col.append(["#f0f4f8" if j%2==0 else "#ffffff" for j in range(len(df))])

    fig = go.Figure(go.Table(
        columnwidth=[160]+[90]*(len(avail)-1),
        header=dict(values=avail_headers, fill_color="#1e3a5f",
                    font=dict(color="white",size=11), align="center", height=34),
        cells=dict(values=cell_vals, fill_color=cell_colors_col,
                   font=dict(size=11), align=["left"]+["center"]*(len(avail)-1), height=32),
    ))
    fig.update_layout(paper_bgcolor="rgba(0,0,0,0)", margin=dict(l=0,r=0,t=10,b=0),
                      height=max(400,len(df)*34+60))
    return fig

def make_multi_sector_heatmap(all_results):
    rows, y_labels, max_stocks = [], [], 7
    for sector, df in all_results.items():
        if df.empty: continue
        top    = df.head(max_stocks)
        col    = "stock" if "stock" in top.columns else "ticker"
        scores = top["composite_score"].tolist()
        while len(scores) < max_stocks:
            scores.append(np.nan)
        rows.append(scores)
        y_labels.append(sector)
    if not rows: return go.Figure()
    fig = go.Figure(go.Heatmap(
        z=rows, x=[f"#{i+1}" for i in range(max_stocks)], y=y_labels,
        colorscale=[[0.0,"#dc2626"],[0.4,"#fca5a5"],[0.5,"#f9fafb"],[0.65,"#86efac"],[1.0,"#16a34a"]],
        zmin=0, zmax=100, colorbar=dict(title="Score",thickness=12),
        hovertemplate="%{y} — Rank %{x}<br>Score: %{z:.0f}",
    ))
    _base_layout(fig, margin=dict(l=160,r=80,t=20,b=40),
                 height=max(400,len(y_labels)*28+80), xaxis_title="Rank within sector", yaxis_title="")
    return fig

def make_sparkline(close, color="#16a34a"):
    s = close.dropna().tail(30)
    fig = go.Figure(go.Scatter(
        x=list(range(len(s))), y=s.values, mode="lines",
        line=dict(color=color, width=1.5), fill="tozeroy",
        fillcolor=color+"26",
    ))
    fig.update_layout(plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
                      margin=dict(l=0,r=0,t=0,b=0), height=50,
                      xaxis=dict(visible=False), yaxis=dict(visible=False), showlegend=False)
    return fig

def make_cross_sector_bar(all_results, metric="composite_score", top_n=1):
    records = []
    for sector, df in all_results.items():
        if df.empty: continue
        if "stock" not in df.columns and "ticker" in df.columns:
            df = df.copy()
            df["stock"] = df["ticker"].str.replace(".NS","",regex=False)
        for _, row in df.head(top_n).iterrows():
            records.append({"sector":sector.replace("NIFTY ",""),
                            "stock":row.get("stock",row.get("ticker","")),
                            "value":row.get(metric,np.nan)})
    if not records: return go.Figure()
    df_plot = pd.DataFrame(records).dropna(subset=["value"]).sort_values("value",ascending=False)
    fig = go.Figure(go.Bar(
        x=df_plot["sector"], y=df_plot["value"],
        text=df_plot["stock"], textposition="outside", textfont=dict(size=9),
        marker=dict(color=df_plot["value"],
                    colorscale=[[0,"#dc2626"],[0.5,"#facc15"],[1,"#16a34a"]],
                    cmin=0, cmax=100, showscale=False),
        hovertemplate="%{x}<br>%{text}<br>Score: %{y:.1f}",
    ))
    _base_layout(fig, height=380, xaxis_title="", yaxis_title=metric.replace("_"," ").title(),
                 xaxis_tickangle=-35, bargap=0.3, margin=dict(l=10,r=10,t=36,b=80))
    return fig

def _score_color(v):
    if pd.isna(v): return "#f9fafb"
    if v >= 70: return "#dcfce7"
    if v >= 50: return "#fef9c3"
    return "#fee2e2"

def _rs_color(v):
    if pd.isna(v): return "#f9fafb"
    if v >= 105: return "#dcfce7"
    if v >= 95:  return "#f9fafb"
    return "#fee2e2"

def _rsi_color(v):
    if pd.isna(v): return "#f9fafb"
    if 55 <= v <= 75: return "#dcfce7"
    if 40 <= v < 55:  return "#fef9c3"
    if v > 75:        return "#fef9c3"
    return "#fee2e2"

def _return_color(v):
    if pd.isna(v): return "#f9fafb"
    if v > 0: return "#dcfce7"
    if v < 0: return "#fee2e2"
    return "#f9fafb"
