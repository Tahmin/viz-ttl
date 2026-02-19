#!/usr/bin/env python3
"""
TASS Report Explorer — Interactive Genomic Surveillance Dashboard
Influenza A Virus | Positive Control QC Analysis
"""

import json
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from dash import Dash, dcc, html, Input, Output, State, callback, ctx, no_update
from plotly.subplots import make_subplots

CSV_PATH = "tass_report_mqc.csv"

# ── Influenza A segment annotations (by reference size descending) ──
SEGMENT_ANNOTATION = {
    "HM773207": {"gene": "PB2", "protein": "Polymerase Basic 2", "order": 1},
    "HM773208": {"gene": "PB1", "protein": "Polymerase Basic 1", "order": 2},
    "HM773209": {"gene": "PA",  "protein": "Polymerase Acidic",  "order": 3},
    "HM773210": {"gene": "HA",  "protein": "Hemagglutinin",      "order": 4},
    "HM773211": {"gene": "NP",  "protein": "Nucleoprotein",      "order": 5},
    "HM773212": {"gene": "NA",  "protein": "Neuraminidase",      "order": 6},
}

# State coordinates for US map
STATE_COORDS = {
    "Georgia": {"lat": 33.2490, "lon": -83.4426},
    "California": {"lat": 36.7783, "lon": -119.4179},
    "Texas": {"lat": 31.9686, "lon": -99.9018},
    "New York": {"lat": 42.1657, "lon": -74.9481},
    "Florida": {"lat": 27.6648, "lon": -81.5158},
}

NUMERIC_COLS = [
    "TASS Score", "Ref Size", "ANIr", "% Reads Aligned",
    "Average Read Length", "Positions Mapped", "% Breadth Coverage",
    "Mean Depth", "Depth Evenness (Gini)",
]

CATEGORY_COLS = ["Sample", "Gene Segment", "state"]
CHART_TYPES = ["Scatter", "Bar", "Box", "Histogram", "Violin", "Heatmap"]

# ── Color palette ──
COLORS = {
    "bg": "#0f1117",
    "card": "#1a1d29",
    "card_border": "#2a2d3a",
    "accent": "#00d4aa",
    "accent2": "#7c3aed",
    "accent3": "#f59e0b",
    "text": "#e2e8f0",
    "text_muted": "#94a3b8",
    "danger": "#ef4444",
    "success": "#22c55e",
    "grid": "#1e2130",
}

SEGMENT_COLORS = {
    "PB2": "#06b6d4",
    "PB1": "#8b5cf6",
    "PA":  "#f43f5e",
    "HA":  "#f59e0b",
    "NP":  "#22c55e",
    "NA":  "#3b82f6",
}

SAMPLE_COLORS = {
    "positive_control_1": "#06b6d4",
    "positive_control_2": "#8b5cf6",
    "positive_control_3": "#f59e0b",
    "positive_control_4": "#f43f5e",
    "positive_control_5": "#22c55e",
}


def load_data(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    df["date_collected"] = pd.to_datetime(df["date_collected"], format="%m/%d/%y")
    df["Segment ID"] = df["Reference Contig"].str.extract(r"\|(\w+)\|")[0]
    df["Gene Segment"] = df["Segment ID"].map(lambda x: SEGMENT_ANNOTATION.get(x, {}).get("gene", x))
    df["Protein"] = df["Segment ID"].map(lambda x: SEGMENT_ANNOTATION.get(x, {}).get("protein", x))
    df["Segment Order"] = df["Segment ID"].map(lambda x: SEGMENT_ANNOTATION.get(x, {}).get("order", 99))
    df = df.sort_values(["Sample", "Segment Order"])
    # Coverage gap = reference positions NOT covered
    df["Coverage Gap (bp)"] = df["Ref Size"] - df["Positions Mapped"]
    df["Coverage Gap (%)"] = 100 - df["% Breadth Coverage"]
    return df


df = load_data(CSV_PATH)

# ── Pre-compute summary stats ──
n_samples = df["Sample"].nunique()
n_segments = df["Gene Segment"].nunique()
mean_tass = df["TASS Score"].mean()
mean_breadth = df["% Breadth Coverage"].mean()
min_tass = df["TASS Score"].min()
max_tass = df["TASS Score"].max()
total_positions = df["Positions Mapped"].sum()


# ── Plotly template ──
def dark_layout(**overrides):
    base = dict(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family="Inter, system-ui, sans-serif", color=COLORS["text"], size=12),
        margin=dict(l=60, r=30, t=50, b=50),
        xaxis=dict(gridcolor=COLORS["grid"], zerolinecolor=COLORS["grid"]),
        yaxis=dict(gridcolor=COLORS["grid"], zerolinecolor=COLORS["grid"]),
        hoverlabel=dict(bgcolor=COLORS["card"], font_size=12, bordercolor=COLORS["accent"]),
        legend=dict(bgcolor="rgba(0,0,0,0)", font=dict(size=11)),
    )
    base.update(overrides)
    return base


# ── Component helpers ──
def kpi_card(title, value, subtitle="", color=COLORS["accent"], icon=""):
    return html.Div(
        style={
            "background": f"linear-gradient(135deg, {COLORS['card']} 0%, {COLORS['bg']} 100%)",
            "border": f"1px solid {COLORS['card_border']}",
            "borderRadius": "12px",
            "padding": "20px 24px",
            "flex": "1",
            "minWidth": "170px",
            "position": "relative",
            "overflow": "hidden",
        },
        children=[
            html.Div(
                style={
                    "position": "absolute", "top": "-10px", "right": "-10px",
                    "fontSize": "64px", "opacity": "0.06", "lineHeight": "1",
                },
                children=icon,
            ),
            html.Div(title, style={
                "fontSize": "11px", "fontWeight": "600", "color": COLORS["text_muted"],
                "textTransform": "uppercase", "letterSpacing": "1.2px", "marginBottom": "8px",
            }),
            html.Div(value, style={
                "fontSize": "28px", "fontWeight": "700", "color": color,
                "lineHeight": "1.1",
            }),
            html.Div(subtitle, style={
                "fontSize": "11px", "color": COLORS["text_muted"], "marginTop": "6px",
            }) if subtitle else html.Div(),
        ],
    )


def section_header(title, subtitle=""):
    return html.Div(
        style={"marginBottom": "16px", "marginTop": "8px"},
        children=[
            html.H2(title, style={
                "fontSize": "20px", "fontWeight": "700", "color": COLORS["text"],
                "margin": "0 0 4px 0", "letterSpacing": "-0.3px",
            }),
            html.P(subtitle, style={
                "fontSize": "13px", "color": COLORS["text_muted"], "margin": "0",
            }) if subtitle else html.Div(),
        ],
    )


def card_wrapper(children, **style_overrides):
    style = {
        "background": COLORS["card"],
        "border": f"1px solid {COLORS['card_border']}",
        "borderRadius": "12px",
        "padding": "20px",
        "marginBottom": "20px",
    }
    style.update(style_overrides)
    return html.Div(style=style, children=children)


# ── Dash app ──
CUSTOM_CSS = """
* { box-sizing: border-box; }
::-webkit-scrollbar { width: 8px; height: 8px; }
::-webkit-scrollbar-track { background: #0f1117; }
::-webkit-scrollbar-thumb { background: #2a2d3a; border-radius: 4px; }
::-webkit-scrollbar-thumb:hover { background: #3a3d4a; }
.Select-control, .Select-menu-outer { background-color: #1a1d29 !important; }
.dash-dropdown .Select-value-label { color: #e2e8f0 !important; }
.VirtualizedSelectOption { background-color: #1a1d29; color: #e2e8f0; }
.VirtualizedSelectFocusedOption { background-color: #2a2d3a !important; }
@keyframes fadeIn { from { opacity: 0; transform: translateY(10px); }
                    to   { opacity: 1; transform: translateY(0); } }
.fade-in { animation: fadeIn 0.5s ease-out forwards; }
@keyframes pulse { 0%, 100% { opacity: 1; } 50% { opacity: 0.7; } }
.pulse { animation: pulse 2s ease-in-out infinite; }
"""

app = Dash(__name__)
app.title = "TASS Report Explorer | Influenza A Surveillance"
# Inject custom CSS via index_string
app.index_string = '''
<!DOCTYPE html>
<html>
    <head>
        {%metas%}
        <title>{%title%}</title>
        {%favicon%}
        {%css%}
        <style>''' + CUSTOM_CSS + '''</style>
    </head>
    <body>
        {%app_entry%}
        <footer>
            {%config%}
            {%scripts%}
            {%renderer%}
        </footer>
    </body>
</html>
'''

app.layout = html.Div(
    style={
        "fontFamily": "Inter, system-ui, -apple-system, sans-serif",
        "backgroundColor": COLORS["bg"],
        "color": COLORS["text"],
        "minHeight": "100vh",
        "padding": "0",
    },
    children=[

        # ─── Header ───
        html.Div(
            style={
                "background": f"linear-gradient(135deg, {COLORS['bg']} 0%, #161929 50%, {COLORS['bg']} 100%)",
                "borderBottom": f"1px solid {COLORS['card_border']}",
                "padding": "28px 40px 24px",
            },
            children=[
                html.Div(
                    style={"display": "flex", "alignItems": "center", "gap": "16px", "marginBottom": "6px"},
                    children=[
                        html.Div(
                            style={
                                "width": "40px", "height": "40px", "borderRadius": "10px",
                                "background": f"linear-gradient(135deg, {COLORS['accent']} 0%, {COLORS['accent2']} 100%)",
                                "display": "flex", "alignItems": "center", "justifyContent": "center",
                                "fontSize": "20px",
                            },
                            children=html.Span("\U0001f9ec"),
                        ),
                        html.Div([
                            html.H1("TASS Report Explorer", style={
                                "fontSize": "26px", "fontWeight": "800", "margin": "0",
                                "background": f"linear-gradient(135deg, {COLORS['accent']}, {COLORS['accent2']})",
                                "WebkitBackgroundClip": "text", "WebkitTextFillColor": "transparent",
                                "letterSpacing": "-0.5px",
                            }),
                            html.P(
                                "Influenza A Virus  |  Positive Control QC  |  Genomic Surveillance Dashboard",
                                style={"fontSize": "13px", "color": COLORS["text_muted"], "margin": "2px 0 0 0"},
                            ),
                        ]),
                    ],
                ),
                html.Div(
                    style={"display": "flex", "gap": "8px", "marginTop": "12px", "flexWrap": "wrap"},
                    children=[
                        html.Span(
                            f"NCBI: {df['NCBI Accession'].iloc[0]}",
                            style={
                                "fontSize": "11px", "padding": "3px 10px", "borderRadius": "20px",
                                "background": "rgba(0,212,170,0.12)", "color": COLORS["accent"],
                                "border": f"1px solid rgba(0,212,170,0.25)",
                            },
                        ),
                        html.Span(
                            f"Run Date: {df['TaxTriage Run Date'].iloc[0]}",
                            style={
                                "fontSize": "11px", "padding": "3px 10px", "borderRadius": "20px",
                                "background": "rgba(124,58,237,0.12)", "color": COLORS["accent2"],
                                "border": f"1px solid rgba(124,58,237,0.25)",
                            },
                        ),
                        html.Span(
                            f"Organism: {df['Organism'].iloc[0]}",
                            style={
                                "fontSize": "11px", "padding": "3px 10px", "borderRadius": "20px",
                                "background": "rgba(245,158,11,0.12)", "color": COLORS["accent3"],
                                "border": f"1px solid rgba(245,158,11,0.25)",
                            },
                        ),
                    ],
                ),
            ],
        ),

        # ─── Main content ───
        html.Div(
            style={"padding": "28px 40px", "maxWidth": "1480px", "margin": "0 auto"},
            children=[

                # ── KPI row ──
                html.Div(
                    style={"display": "flex", "gap": "16px", "flexWrap": "wrap", "marginBottom": "28px"},
                    children=[
                        kpi_card("Samples Analyzed", str(n_samples), f"{n_segments} genome segments each", COLORS["accent"], "\U0001f9ea"),
                        kpi_card("Mean TASS Score", f"{mean_tass:.2f}", f"Range: {min_tass:.2f} \u2013 {max_tass:.2f}", COLORS["accent2"], "\u2b50"),
                        kpi_card("Mean Breadth Coverage", f"{mean_breadth:.1f}%", "Across all segments", COLORS["accent3"], "\U0001f4ca"),
                        kpi_card("Total Positions Mapped", f"{total_positions:,}", "Sum across all alignments", COLORS["success"], "\U0001f9ec"),
                        kpi_card("Pipeline QC Status",
                                 "PASS" if min_tass > 90 else "REVIEW",
                                 "All scores > 90" if min_tass > 90 else "Some scores below threshold",
                                 COLORS["success"] if min_tass > 90 else COLORS["danger"], "\u2705"),
                    ],
                ),

                # ── Filter bar ──
                card_wrapper(
                    children=[
                        html.Div(
                            style={"display": "flex", "alignItems": "center", "gap": "12px", "marginBottom": "14px"},
                            children=[
                                html.Span("\u2699\ufe0f", style={"fontSize": "16px"}),
                                html.Span("Global Filters", style={
                                    "fontWeight": "700", "fontSize": "14px",
                                }),
                                html.Span(
                                    "All charts respond to these filters in real time",
                                    style={"fontSize": "11px", "color": COLORS["text_muted"], "marginLeft": "8px"},
                                ),
                            ],
                        ),
                        html.Div(
                            style={"display": "flex", "gap": "20px", "flexWrap": "wrap"},
                            children=[
                                html.Div([
                                    html.Label("Sample", style={
                                        "fontSize": "11px", "fontWeight": "600", "color": COLORS["text_muted"],
                                        "textTransform": "uppercase", "letterSpacing": "0.8px", "marginBottom": "6px",
                                        "display": "block",
                                    }),
                                    dcc.Dropdown(
                                        id="filter-sample",
                                        options=[{"label": s.replace("positive_control_", "Control "), "value": s}
                                                 for s in sorted(df["Sample"].unique())],
                                        value=sorted(df["Sample"].unique()),
                                        multi=True, placeholder="All samples",
                                        style={"backgroundColor": COLORS["bg"], "color": COLORS["text"]},
                                    ),
                                ], style={"flex": "1", "minWidth": "200px"}),
                                html.Div([
                                    html.Label("State", style={
                                        "fontSize": "11px", "fontWeight": "600", "color": COLORS["text_muted"],
                                        "textTransform": "uppercase", "letterSpacing": "0.8px", "marginBottom": "6px",
                                        "display": "block",
                                    }),
                                    dcc.Dropdown(
                                        id="filter-state",
                                        options=[{"label": s, "value": s} for s in sorted(df["state"].unique())],
                                        value=sorted(df["state"].unique()),
                                        multi=True, placeholder="All states",
                                        style={"backgroundColor": COLORS["bg"]},
                                    ),
                                ], style={"flex": "1", "minWidth": "180px"}),
                                html.Div([
                                    html.Label("Gene Segment", style={
                                        "fontSize": "11px", "fontWeight": "600", "color": COLORS["text_muted"],
                                        "textTransform": "uppercase", "letterSpacing": "0.8px", "marginBottom": "6px",
                                        "display": "block",
                                    }),
                                    dcc.Dropdown(
                                        id="filter-segment",
                                        options=[{"label": f"{s} ({SEGMENT_ANNOTATION.get('HM7732' + str(6+i+1), {}).get('protein', '')})", "value": s}
                                                 for i, s in enumerate(["PB2", "PB1", "PA", "HA", "NP", "NA"])],
                                        value=sorted(df["Gene Segment"].unique()),
                                        multi=True, placeholder="All segments",
                                        style={"backgroundColor": COLORS["bg"]},
                                    ),
                                ], style={"flex": "1", "minWidth": "200px"}),
                            ],
                        ),
                    ],
                ),

                # ── Row 1: Genome overview + US map ──
                html.Div(
                    style={"display": "grid", "gridTemplateColumns": "1fr 1fr", "gap": "20px", "marginBottom": "20px"},
                    children=[
                        card_wrapper([
                            section_header("Genome Segment Coverage Profile",
                                           "TASS Score and breadth coverage for each Influenza A segment"),
                            dcc.Graph(id="genome-radar", config={"displayModeBar": False}),
                        ]),
                        card_wrapper([
                            section_header("Sample Collection Geography",
                                           "Geographic distribution of positive controls across the US"),
                            dcc.Graph(id="geo-map", config={"displayModeBar": False}),
                        ]),
                    ],
                ),

                # ── Row 2: Segment comparison heatmap ──
                card_wrapper([
                    html.Div(
                        style={"display": "flex", "justifyContent": "space-between", "alignItems": "flex-end",
                               "flexWrap": "wrap", "gap": "12px"},
                        children=[
                            section_header("Segment-Level QC Heatmap",
                                           "Compare any metric across all samples and genome segments"),
                            html.Div([
                                html.Label("Metric", style={
                                    "fontSize": "11px", "fontWeight": "600", "color": COLORS["text_muted"],
                                    "textTransform": "uppercase", "letterSpacing": "0.8px",
                                    "marginBottom": "4px", "display": "block",
                                }),
                                dcc.Dropdown(
                                    id="heatmap-metric",
                                    options=[{"label": c, "value": c} for c in NUMERIC_COLS],
                                    value="TASS Score", clearable=False,
                                    style={"width": "220px", "backgroundColor": COLORS["bg"]},
                                ),
                            ]),
                        ],
                    ),
                    dcc.Graph(id="heatmap-chart", config={"displayModeBar": False}),
                ]),

                # ── Row 3: Bar + Box side by side ──
                html.Div(
                    style={"display": "grid", "gridTemplateColumns": "1fr 1fr", "gap": "20px", "marginBottom": "20px"},
                    children=[
                        card_wrapper([
                            html.Div(
                                style={"display": "flex", "justifyContent": "space-between", "alignItems": "flex-end",
                                       "flexWrap": "wrap", "gap": "12px"},
                                children=[
                                    section_header("Per-Sample Breakdown", "Metric by sample, colored by gene segment"),
                                    dcc.Dropdown(
                                        id="bar-metric",
                                        options=[{"label": c, "value": c} for c in NUMERIC_COLS],
                                        value="TASS Score", clearable=False,
                                        style={"width": "200px", "backgroundColor": COLORS["bg"]},
                                    ),
                                ],
                            ),
                            dcc.Graph(id="bar-chart", config={"displayModeBar": False}),
                        ]),
                        card_wrapper([
                            section_header("Distribution by State",
                                           "Statistical spread across collection sites"),
                            dcc.Graph(id="box-chart", config={"displayModeBar": False}),
                        ]),
                    ],
                ),

                # ── Row 4: Coverage gap lollipop + Timeline ──
                html.Div(
                    style={"display": "grid", "gridTemplateColumns": "1fr 1fr", "gap": "20px", "marginBottom": "20px"},
                    children=[
                        card_wrapper([
                            section_header("Coverage Gap Analysis",
                                           "Unmapped positions per segment — lower is better"),
                            dcc.Graph(id="coverage-gap", config={"displayModeBar": False}),
                        ]),
                        card_wrapper([
                            section_header("Temporal Trend",
                                           "Mean TASS Score trajectory across collection dates"),
                            dcc.Graph(id="timeline-chart", config={"displayModeBar": False}),
                        ]),
                    ],
                ),

                # ── Row 5: Parallel coordinates ──
                card_wrapper([
                    section_header("Multi-Dimensional QC Explorer",
                                   "Parallel coordinates view — each line is one sample-segment alignment"),
                    dcc.Graph(id="parallel-coords", config={"displayModeBar": False}),
                ]),

                # ── Row 6: Custom chart builder ──
                card_wrapper([
                    html.Div(
                        style={"display": "flex", "alignItems": "center", "gap": "12px", "marginBottom": "16px"},
                        children=[
                            html.Span("\U0001f528", style={"fontSize": "18px"}),
                            section_header("Custom Chart Builder",
                                           "Build any visualization from the dataset — choose chart type, axes, and color grouping"),
                        ],
                    ),
                    html.Div(
                        style={"display": "flex", "gap": "16px", "flexWrap": "wrap", "marginBottom": "16px"},
                        children=[
                            html.Div([
                                html.Label("Chart Type", style={
                                    "fontSize": "11px", "fontWeight": "600", "color": COLORS["text_muted"],
                                    "textTransform": "uppercase", "letterSpacing": "0.8px",
                                    "marginBottom": "4px", "display": "block",
                                }),
                                dcc.Dropdown(
                                    id="chart-type",
                                    options=[{"label": t, "value": t} for t in CHART_TYPES],
                                    value="Scatter", clearable=False,
                                    style={"backgroundColor": COLORS["bg"]},
                                ),
                            ], style={"flex": "1", "minWidth": "130px"}),
                            html.Div([
                                html.Label("X Axis", style={
                                    "fontSize": "11px", "fontWeight": "600", "color": COLORS["text_muted"],
                                    "textTransform": "uppercase", "letterSpacing": "0.8px",
                                    "marginBottom": "4px", "display": "block",
                                }),
                                dcc.Dropdown(
                                    id="x-axis",
                                    options=[{"label": c, "value": c} for c in NUMERIC_COLS],
                                    value="% Breadth Coverage", clearable=False,
                                    style={"backgroundColor": COLORS["bg"]},
                                ),
                            ], style={"flex": "1", "minWidth": "160px"}),
                            html.Div([
                                html.Label("Y Axis", style={
                                    "fontSize": "11px", "fontWeight": "600", "color": COLORS["text_muted"],
                                    "textTransform": "uppercase", "letterSpacing": "0.8px",
                                    "marginBottom": "4px", "display": "block",
                                }),
                                dcc.Dropdown(
                                    id="y-axis",
                                    options=[{"label": c, "value": c} for c in NUMERIC_COLS],
                                    value="TASS Score", clearable=False,
                                    style={"backgroundColor": COLORS["bg"]},
                                ),
                            ], style={"flex": "1", "minWidth": "160px"}),
                            html.Div([
                                html.Label("Color By", style={
                                    "fontSize": "11px", "fontWeight": "600", "color": COLORS["text_muted"],
                                    "textTransform": "uppercase", "letterSpacing": "0.8px",
                                    "marginBottom": "4px", "display": "block",
                                }),
                                dcc.Dropdown(
                                    id="color-by",
                                    options=[{"label": c, "value": c} for c in CATEGORY_COLS],
                                    value="Sample", clearable=False,
                                    style={"backgroundColor": COLORS["bg"]},
                                ),
                            ], style={"flex": "1", "minWidth": "140px"}),
                        ],
                    ),
                    dcc.Graph(id="custom-chart", config={"displayModeBar": False}),
                ]),

                # ── Row 7: Data table ──
                card_wrapper([
                    section_header("Raw Data Explorer",
                                   "Filtered dataset — sortable and scrollable"),
                    html.Div(id="data-table", style={"overflowX": "auto", "maxHeight": "420px", "overflowY": "auto"}),
                ]),

                # ── Footer ──
                html.Div(
                    style={
                        "textAlign": "center", "padding": "28px 0 12px",
                        "fontSize": "11px", "color": COLORS["text_muted"],
                        "borderTop": f"1px solid {COLORS['card_border']}", "marginTop": "20px",
                    },
                    children=[
                        html.P("TASS Report Explorer | TaxTriage Genomic Surveillance Pipeline"),
                        html.P(f"Data: {df['NCBI Accession'].iloc[0]} | {df['Organism'].iloc[0]} | "
                               f"Analysis Run: {df['TaxTriage Run Date'].iloc[0]}"),
                    ],
                ),
            ],
        ),
    ],
)


# ────────────────────────────────────────────────────────────
# CALLBACKS
# ────────────────────────────────────────────────────────────

def filtered_df(samples, states, segments):
    mask = (
        df["Sample"].isin(samples if samples else df["Sample"].unique())
        & df["state"].isin(states if states else df["state"].unique())
        & df["Gene Segment"].isin(segments if segments else df["Gene Segment"].unique())
    )
    return df[mask]


# ── Genome radar ──
@callback(
    Output("genome-radar", "figure"),
    Input("filter-sample", "value"),
    Input("filter-state", "value"),
    Input("filter-segment", "value"),
)
def update_radar(samples, states, segments):
    dff = filtered_df(samples, states, segments)
    fig = go.Figure()

    segment_order = ["PB2", "PB1", "PA", "HA", "NP", "NA"]
    for sample in sorted(dff["Sample"].unique()):
        sub = dff[dff["Sample"] == sample].set_index("Gene Segment")
        vals = []
        for seg in segment_order:
            if seg in sub.index:
                vals.append(sub.loc[seg, "TASS Score"] if isinstance(sub.loc[seg, "TASS Score"], (int, float, np.floating)) else sub.loc[seg, "TASS Score"].iloc[0])
            else:
                vals.append(None)
        label = sample.replace("positive_control_", "Ctrl ")
        fig.add_trace(go.Scatterpolar(
            r=vals + [vals[0]] if vals[0] is not None else vals,
            theta=segment_order + [segment_order[0]],
            name=label,
            line=dict(color=SAMPLE_COLORS.get(sample, "#888"), width=2),
            fill="toself",
            fillcolor=SAMPLE_COLORS.get(sample, "#888").replace(")", ",0.08)").replace("rgb", "rgba") if "rgb" in SAMPLE_COLORS.get(sample, "#888") else None,
            opacity=0.85,
        ))

    fig.update_layout(
        **dark_layout(
            margin=dict(l=60, r=60, t=30, b=30),
            height=380,
            polar=dict(
                bgcolor="rgba(0,0,0,0)",
                radialaxis=dict(
                    visible=True, range=[94, 99], gridcolor=COLORS["grid"],
                    tickfont=dict(size=10, color=COLORS["text_muted"]),
                ),
                angularaxis=dict(
                    gridcolor=COLORS["grid"],
                    tickfont=dict(size=12, color=COLORS["text"]),
                ),
            ),
            showlegend=True,
            legend=dict(font=dict(size=10), orientation="h", y=-0.15, x=0.5, xanchor="center"),
        ),
    )
    return fig


# ── Geo map ──
@callback(
    Output("geo-map", "figure"),
    Input("filter-sample", "value"),
    Input("filter-state", "value"),
    Input("filter-segment", "value"),
)
def update_geo(samples, states, segments):
    dff = filtered_df(samples, states, segments)
    agg = dff.groupby(["Sample", "state", "date_collected"]).agg(
        mean_tass=("TASS Score", "mean"),
        mean_breadth=("% Breadth Coverage", "mean"),
    ).reset_index()
    agg["lat"] = agg["state"].map(lambda s: STATE_COORDS.get(s, {}).get("lat"))
    agg["lon"] = agg["state"].map(lambda s: STATE_COORDS.get(s, {}).get("lon"))
    agg = agg.dropna(subset=["lat", "lon"])

    fig = go.Figure()
    for _, row in agg.iterrows():
        label = row["Sample"].replace("positive_control_", "Ctrl ")
        fig.add_trace(go.Scattergeo(
            lat=[row["lat"]],
            lon=[row["lon"]],
            text=f"<b>{label}</b><br>{row['state']}<br>"
                 f"Collected: {row['date_collected'].strftime('%b %d, %Y')}<br>"
                 f"Mean TASS: {row['mean_tass']:.2f}<br>"
                 f"Mean Breadth: {row['mean_breadth']:.1f}%",
            hoverinfo="text",
            marker=dict(
                size=row["mean_tass"] - 90,  # scale: ~6-8
                sizemin=6,
                sizemode="diameter",
                color=SAMPLE_COLORS.get(row["Sample"], COLORS["accent"]),
                opacity=0.9,
                line=dict(width=1.5, color="white"),
            ),
            name=label,
            showlegend=False,
        ))

    fig.update_layout(
        **dark_layout(height=380, margin=dict(l=0, r=0, t=10, b=0)),
        geo=dict(
            scope="usa",
            bgcolor="rgba(0,0,0,0)",
            lakecolor="rgba(0,0,0,0)",
            landcolor=COLORS["card"],
            subunitcolor=COLORS["card_border"],
            showlakes=False,
            coastlinecolor=COLORS["card_border"],
        ),
    )
    return fig


# ── Heatmap ──
@callback(
    Output("heatmap-chart", "figure"),
    Input("heatmap-metric", "value"),
    Input("filter-sample", "value"),
    Input("filter-state", "value"),
    Input("filter-segment", "value"),
)
def update_heatmap(metric, samples, states, segments):
    dff = filtered_df(samples, states, segments)
    segment_order = [s for s in ["PB2", "PB1", "PA", "HA", "NP", "NA"] if s in dff["Gene Segment"].values]
    sample_order = sorted(dff["Sample"].unique())
    pivot = dff.pivot_table(index="Sample", columns="Gene Segment", values=metric, aggfunc="mean")
    pivot = pivot.reindex(index=sample_order, columns=segment_order)

    labels = [[f"{v:.2f}" if pd.notna(v) else "" for v in row] for row in pivot.values]
    sample_labels = [s.replace("positive_control_", "Control ") for s in pivot.index]

    fig = go.Figure(go.Heatmap(
        z=pivot.values,
        x=pivot.columns.tolist(),
        y=sample_labels,
        text=labels,
        texttemplate="%{text}",
        textfont=dict(size=12, color="white"),
        colorscale=[
            [0, "#1e1b4b"], [0.25, "#3730a3"], [0.5, "#7c3aed"],
            [0.75, "#a78bfa"], [1, "#00d4aa"],
        ],
        colorbar=dict(title=dict(text=metric, font=dict(size=11)), thickness=12, len=0.8),
        hovertemplate=f"<b>%{{y}}</b><br>Segment: %{{x}}<br>{metric}: %{{z:.3f}}<extra></extra>",
    ))

    fig.update_layout(**dark_layout(height=300, margin=dict(l=100, r=30, t=20, b=50)))
    return fig


# ── Bar chart ──
@callback(
    Output("bar-chart", "figure"),
    Input("bar-metric", "value"),
    Input("filter-sample", "value"),
    Input("filter-state", "value"),
    Input("filter-segment", "value"),
)
def update_bar(metric, samples, states, segments):
    dff = filtered_df(samples, states, segments)
    dff = dff.copy()
    dff["Sample Label"] = dff["Sample"].str.replace("positive_control_", "Ctrl ")

    fig = px.bar(
        dff, x="Sample Label", y=metric, color="Gene Segment",
        barmode="group",
        color_discrete_map=SEGMENT_COLORS,
        hover_data=["state", "Protein", "Ref Size"],
    )
    fig.update_layout(**dark_layout(height=360, margin=dict(l=60, r=20, t=20, b=50)))
    fig.update_xaxes(title="")
    return fig


# ── Box chart ──
@callback(
    Output("box-chart", "figure"),
    Input("bar-metric", "value"),
    Input("filter-sample", "value"),
    Input("filter-state", "value"),
    Input("filter-segment", "value"),
)
def update_box(metric, samples, states, segments):
    dff = filtered_df(samples, states, segments)
    fig = px.box(
        dff, x="state", y=metric, color="state", points="all",
        hover_data=["Sample", "Gene Segment"],
    )
    fig.update_layout(**dark_layout(height=360, margin=dict(l=60, r=20, t=20, b=50), showlegend=False))
    fig.update_xaxes(title="")
    return fig


# ── Coverage gap lollipop ──
@callback(
    Output("coverage-gap", "figure"),
    Input("filter-sample", "value"),
    Input("filter-state", "value"),
    Input("filter-segment", "value"),
)
def update_coverage_gap(samples, states, segments):
    dff = filtered_df(samples, states, segments)
    agg = dff.groupby("Gene Segment").agg(
        mean_gap=("Coverage Gap (%)", "mean"),
        std_gap=("Coverage Gap (%)", "std"),
    ).reindex(["PB2", "PB1", "PA", "HA", "NP", "NA"]).reset_index()

    fig = go.Figure()
    for i, row in agg.iterrows():
        color = SEGMENT_COLORS.get(row["Gene Segment"], "#888")
        fig.add_trace(go.Scatter(
            x=[row["mean_gap"]], y=[row["Gene Segment"]],
            mode="markers+text",
            marker=dict(size=14, color=color, line=dict(width=2, color="white")),
            text=f"  {row['mean_gap']:.1f}%",
            textposition="middle right",
            textfont=dict(size=12, color=color),
            showlegend=False,
            hovertemplate=f"<b>{row['Gene Segment']}</b><br>Mean gap: {row['mean_gap']:.2f}%<br>"
                          f"Std: {row['std_gap']:.2f}%<extra></extra>",
        ))
        # stem line
        fig.add_shape(
            type="line", x0=0, x1=row["mean_gap"],
            y0=row["Gene Segment"], y1=row["Gene Segment"],
            line=dict(color=color, width=3, dash="dot"),
        )

    fig.update_layout(**dark_layout(
        height=360, margin=dict(l=60, r=40, t=20, b=50),
        xaxis=dict(title="Mean Coverage Gap (%)", gridcolor=COLORS["grid"], range=[0, max(agg["mean_gap"]) * 1.4]),
        yaxis=dict(title=""),
    ))
    return fig


# ── Timeline ──
@callback(
    Output("timeline-chart", "figure"),
    Input("bar-metric", "value"),
    Input("filter-sample", "value"),
    Input("filter-state", "value"),
    Input("filter-segment", "value"),
)
def update_timeline(metric, samples, states, segments):
    dff = filtered_df(samples, states, segments)
    agg = dff.groupby(["Sample", "date_collected", "state"])[metric].mean().reset_index()
    agg = agg.sort_values("date_collected")

    fig = go.Figure()
    # Area fill
    fig.add_trace(go.Scatter(
        x=agg["date_collected"], y=agg[metric],
        mode="lines",
        line=dict(color=COLORS["accent"], width=0),
        fill="tozeroy",
        fillcolor="rgba(0,212,170,0.08)",
        showlegend=False,
        hoverinfo="skip",
    ))
    # Main line
    fig.add_trace(go.Scatter(
        x=agg["date_collected"], y=agg[metric],
        mode="lines+markers+text",
        line=dict(color=COLORS["accent"], width=3),
        marker=dict(size=10, color=COLORS["accent"], line=dict(width=2, color="white")),
        text=agg["state"],
        textposition="top center",
        textfont=dict(size=10, color=COLORS["text_muted"]),
        hovertemplate="<b>%{text}</b><br>Date: %{x|%b %d, %Y}<br>" + metric + ": %{y:.3f}<extra></extra>",
        showlegend=False,
    ))

    fig.update_layout(**dark_layout(
        height=360, margin=dict(l=60, r=30, t=20, b=50),
        xaxis=dict(title="Collection Date"),
        yaxis=dict(title=f"Mean {metric}"),
    ))
    return fig


# ── Parallel coordinates ──
@callback(
    Output("parallel-coords", "figure"),
    Input("filter-sample", "value"),
    Input("filter-state", "value"),
    Input("filter-segment", "value"),
)
def update_parallel(samples, states, segments):
    dff = filtered_df(samples, states, segments).copy()
    # Encode sample as numeric for coloring
    sample_map = {s: i for i, s in enumerate(sorted(dff["Sample"].unique()))}
    dff["sample_code"] = dff["Sample"].map(sample_map)

    dims = [
        dict(label="TASS Score", values=dff["TASS Score"]),
        dict(label="Breadth Cov %", values=dff["% Breadth Coverage"]),
        dict(label="Positions Mapped", values=dff["Positions Mapped"]),
        dict(label="Ref Size", values=dff["Ref Size"]),
        dict(label="ANIr", values=dff["ANIr"]),
        dict(label="Reads Aligned %", values=dff["% Reads Aligned"]),
        dict(label="Avg Read Len", values=dff["Average Read Length"]),
    ]

    color_vals = list(SAMPLE_COLORS.values())
    n = len(sample_map)
    colorscale = []
    for i in range(n):
        lo = i / n
        hi = (i + 1) / n
        c = color_vals[i % len(color_vals)]
        colorscale.append([lo, c])
        colorscale.append([hi, c])

    fig = go.Figure(go.Parcoords(
        line=dict(
            color=dff["sample_code"],
            colorscale=colorscale,
            showscale=False,
        ),
        dimensions=dims,
        labelfont=dict(size=11, color=COLORS["text"]),
        tickfont=dict(size=10, color=COLORS["text_muted"]),
    ))
    fig.update_layout(**dark_layout(height=400, margin=dict(l=60, r=30, t=30, b=40)))
    return fig


# ── Custom chart ──
@callback(
    Output("custom-chart", "figure"),
    Input("chart-type", "value"),
    Input("x-axis", "value"),
    Input("y-axis", "value"),
    Input("color-by", "value"),
    Input("filter-sample", "value"),
    Input("filter-state", "value"),
    Input("filter-segment", "value"),
)
def update_custom(chart_type, x, y, color, samples, states, segments):
    dff = filtered_df(samples, states, segments)

    color_map = SEGMENT_COLORS if color == "Gene Segment" else SAMPLE_COLORS if color == "Sample" else None
    common = dict(color_discrete_map=color_map) if color_map else {}

    if chart_type == "Scatter":
        fig = px.scatter(dff, x=x, y=y, color=color, hover_data=["Sample", "Gene Segment", "state"], **common)
    elif chart_type == "Bar":
        fig = px.bar(dff, x=x, y=y, color=color, barmode="group", **common)
    elif chart_type == "Box":
        fig = px.box(dff, x=color, y=y, color=color, points="all", **common)
    elif chart_type == "Histogram":
        fig = px.histogram(dff, x=x, color=color, barmode="overlay", opacity=0.7, **common)
    elif chart_type == "Violin":
        fig = px.violin(dff, x=color, y=y, color=color, box=True, points="all", **common)
    elif chart_type == "Heatmap":
        pivot = dff.pivot_table(index="Sample", columns="Gene Segment", values=y, aggfunc="mean")
        fig = px.imshow(pivot, text_auto=".2f", color_continuous_scale="Viridis", aspect="auto")
    else:
        fig = go.Figure()

    fig.update_layout(**dark_layout(height=420))
    return fig


# ── Data table ──
@callback(
    Output("data-table", "children"),
    Input("filter-sample", "value"),
    Input("filter-state", "value"),
    Input("filter-segment", "value"),
)
def update_table(samples, states, segments):
    dff = filtered_df(samples, states, segments)
    display_cols = ["Sample", "Gene Segment", "Protein", "state", "date_collected"] + NUMERIC_COLS
    cols = [c for c in display_cols if c in dff.columns]

    header_style = {
        "border": f"1px solid {COLORS['card_border']}",
        "padding": "8px 12px",
        "background": "#12141f",
        "textAlign": "left",
        "fontSize": "11px",
        "fontWeight": "600",
        "color": COLORS["text_muted"],
        "textTransform": "uppercase",
        "letterSpacing": "0.5px",
        "position": "sticky",
        "top": "0",
        "zIndex": "1",
    }
    cell_style = {
        "border": f"1px solid {COLORS['card_border']}",
        "padding": "6px 12px",
        "fontSize": "12px",
        "color": COLORS["text"],
    }

    return html.Table(
        style={"borderCollapse": "collapse", "width": "100%"},
        children=[
            html.Thead(html.Tr([html.Th(c, style=header_style) for c in cols])),
            html.Tbody([
                html.Tr(
                    style={"backgroundColor": COLORS["card"] if i % 2 == 0 else "#151825"},
                    children=[
                        html.Td(str(row[c]) if not pd.isna(row[c]) else "", style=cell_style)
                        for c in cols
                    ],
                )
                for i, (_, row) in enumerate(dff[cols].iterrows())
            ]),
        ],
    )


if __name__ == "__main__":
    print("\n  TASS Report Explorer")
    print("  http://127.0.0.1:8050\n")
    app.run(debug=True)
