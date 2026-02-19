#!/usr/bin/env python3
"""
TASS Report Explorer — Interactive Genomic Surveillance Dashboard
Influenza A Virus | Positive Control QC Analysis
Supports uploading one or multiple CSV files via the UI.
"""

import base64
import io
import json
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from dash import Dash, dcc, html, Input, Output, State, callback, ctx, no_update
from plotly.subplots import make_subplots

# ── Influenza A segment annotations (by reference size descending) ──
SEGMENT_ANNOTATION = {
    "HM773207": {"gene": "PB2", "protein": "Polymerase Basic 2", "order": 1},
    "HM773208": {"gene": "PB1", "protein": "Polymerase Basic 1", "order": 2},
    "HM773209": {"gene": "PA",  "protein": "Polymerase Acidic",  "order": 3},
    "HM773210": {"gene": "HA",  "protein": "Hemagglutinin",      "order": 4},
    "HM773211": {"gene": "NP",  "protein": "Nucleoprotein",      "order": 5},
    "HM773212": {"gene": "NA",  "protein": "Neuraminidase",      "order": 6},
}

# State coordinates for US map (all 50 states + DC + territories)
STATE_COORDS = {
    "Alabama": {"lat": 32.3182, "lon": -86.9023},
    "Alaska": {"lat": 64.2008, "lon": -152.4937},
    "Arizona": {"lat": 34.0489, "lon": -111.0937},
    "Arkansas": {"lat": 34.7465, "lon": -92.2896},
    "California": {"lat": 36.7783, "lon": -119.4179},
    "Colorado": {"lat": 39.5501, "lon": -105.7821},
    "Connecticut": {"lat": 41.6032, "lon": -73.0877},
    "Delaware": {"lat": 38.9108, "lon": -75.5277},
    "District of Columbia": {"lat": 38.9072, "lon": -77.0369},
    "Florida": {"lat": 27.6648, "lon": -81.5158},
    "Georgia": {"lat": 33.2490, "lon": -83.4426},
    "Hawaii": {"lat": 19.8968, "lon": -155.5828},
    "Idaho": {"lat": 44.0682, "lon": -114.7420},
    "Illinois": {"lat": 40.6331, "lon": -89.3985},
    "Indiana": {"lat": 40.2672, "lon": -86.1349},
    "Iowa": {"lat": 41.8780, "lon": -93.0977},
    "Kansas": {"lat": 39.0119, "lon": -98.4842},
    "Kentucky": {"lat": 37.8393, "lon": -84.2700},
    "Louisiana": {"lat": 30.9843, "lon": -91.9623},
    "Maine": {"lat": 45.2538, "lon": -69.4455},
    "Maryland": {"lat": 39.0458, "lon": -76.6413},
    "Massachusetts": {"lat": 42.4072, "lon": -71.3824},
    "Michigan": {"lat": 44.3148, "lon": -85.6024},
    "Minnesota": {"lat": 46.7296, "lon": -94.6859},
    "Mississippi": {"lat": 32.3547, "lon": -89.3985},
    "Missouri": {"lat": 37.9643, "lon": -91.8318},
    "Montana": {"lat": 46.8797, "lon": -110.3626},
    "Nebraska": {"lat": 41.4925, "lon": -99.9018},
    "Nevada": {"lat": 38.8026, "lon": -116.4194},
    "New Hampshire": {"lat": 43.1939, "lon": -71.5724},
    "New Jersey": {"lat": 40.0583, "lon": -74.4057},
    "New Mexico": {"lat": 34.5199, "lon": -105.8701},
    "New York": {"lat": 42.1657, "lon": -74.9481},
    "North Carolina": {"lat": 35.7596, "lon": -79.0193},
    "North Dakota": {"lat": 47.5515, "lon": -101.0020},
    "Ohio": {"lat": 40.4173, "lon": -82.9071},
    "Oklahoma": {"lat": 35.4676, "lon": -97.5164},
    "Oregon": {"lat": 43.8041, "lon": -120.5542},
    "Pennsylvania": {"lat": 41.2033, "lon": -77.1945},
    "Puerto Rico": {"lat": 18.2208, "lon": -66.5901},
    "Rhode Island": {"lat": 41.5801, "lon": -71.4774},
    "South Carolina": {"lat": 33.8361, "lon": -81.1637},
    "South Dakota": {"lat": 43.9695, "lon": -99.9018},
    "Tennessee": {"lat": 35.5175, "lon": -86.5804},
    "Texas": {"lat": 31.9686, "lon": -99.9018},
    "Utah": {"lat": 39.3210, "lon": -111.0937},
    "Vermont": {"lat": 44.5588, "lon": -72.5778},
    "Virginia": {"lat": 37.4316, "lon": -78.6569},
    "Washington": {"lat": 47.7511, "lon": -120.7401},
    "West Virginia": {"lat": 38.5976, "lon": -80.4549},
    "Wisconsin": {"lat": 43.7844, "lon": -88.7879},
    "Wyoming": {"lat": 43.0760, "lon": -107.2903},
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

# Palette for dynamically-assigned sample colors
SAMPLE_PALETTE = [
    "#06b6d4", "#8b5cf6", "#f59e0b", "#f43f5e", "#22c55e",
    "#3b82f6", "#ec4899", "#14b8a6", "#f97316", "#a855f7",
    "#eab308", "#ef4444", "#0ea5e9", "#84cc16", "#d946ef",
]


def enrich_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """Add derived columns to a raw CSV dataframe."""
    df["date_collected"] = pd.to_datetime(df["date_collected"], format="%m/%d/%y")
    df["Segment ID"] = df["Reference Contig"].str.extract(r"\|(\w+)\|")[0]
    df["Gene Segment"] = df["Segment ID"].map(lambda x: SEGMENT_ANNOTATION.get(x, {}).get("gene", x))
    df["Protein"] = df["Segment ID"].map(lambda x: SEGMENT_ANNOTATION.get(x, {}).get("protein", x))
    df["Segment Order"] = df["Segment ID"].map(lambda x: SEGMENT_ANNOTATION.get(x, {}).get("order", 99))
    df = df.sort_values(["Sample", "Segment Order"])
    df["Coverage Gap (bp)"] = df["Ref Size"] - df["Positions Mapped"]
    df["Coverage Gap (%)"] = 100 - df["% Breadth Coverage"]
    return df


def build_sample_colors(df: pd.DataFrame) -> dict:
    """Dynamically assign colors to samples."""
    samples = sorted(df["Sample"].unique())
    return {s: SAMPLE_PALETTE[i % len(SAMPLE_PALETTE)] for i, s in enumerate(samples)}


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


def empty_fig(msg="Upload CSV files to begin"):
    """Return an empty figure with a centered message."""
    fig = go.Figure()
    fig.add_annotation(
        text=msg, xref="paper", yref="paper", x=0.5, y=0.5,
        showarrow=False, font=dict(size=16, color=COLORS["text_muted"]),
    )
    fig.update_layout(**dark_layout(height=300))
    return fig


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


def dropdown_label_style():
    return {
        "fontSize": "11px", "fontWeight": "600", "color": COLORS["text_muted"],
        "textTransform": "uppercase", "letterSpacing": "0.8px",
        "marginBottom": "4px", "display": "block",
    }


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
        # ─── Data store (holds merged CSV as JSON) ───
        dcc.Store(id="data-store"),

        # ─── Header ───
        html.Div(
            style={
                "background": f"linear-gradient(135deg, {COLORS['bg']} 0%, #161929 50%, {COLORS['bg']} 100%)",
                "borderBottom": f"1px solid {COLORS['card_border']}",
                "padding": "28px 40px 24px",
            },
            children=[
                html.Div(
                    style={"display": "flex", "alignItems": "center", "justifyContent": "space-between",
                           "flexWrap": "wrap", "gap": "16px"},
                    children=[
                        # Left: title
                        html.Div(
                            style={"display": "flex", "alignItems": "center", "gap": "16px"},
                            children=[
                                html.Div(
                                    style={
                                        "width": "40px", "height": "40px", "borderRadius": "10px",
                                        "background": f"linear-gradient(135deg, {COLORS['accent']} 0%, {COLORS['accent2']} 100%)",
                                        "display": "flex", "alignItems": "center", "justifyContent": "center",
                                        "fontSize": "20px", "flexShrink": "0",
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
                        # Right: upload button
                        dcc.Upload(
                            id="csv-upload",
                            children=html.Div(
                                style={
                                    "display": "flex", "alignItems": "center", "gap": "8px",
                                    "padding": "10px 20px", "borderRadius": "8px",
                                    "background": f"linear-gradient(135deg, {COLORS['accent']}, {COLORS['accent2']})",
                                    "cursor": "pointer", "fontWeight": "600", "fontSize": "13px",
                                    "color": "white", "whiteSpace": "nowrap",
                                },
                                children=[
                                    html.Span("\U0001f4c2"),
                                    html.Span("Upload CSV Files"),
                                ],
                            ),
                            multiple=True,
                        ),
                    ],
                ),
                # Metadata pills (dynamic)
                html.Div(id="header-pills", style={"display": "flex", "gap": "8px", "marginTop": "12px", "flexWrap": "wrap"}),
            ],
        ),

        # ─── Main content ───
        html.Div(
            style={"padding": "28px 40px", "maxWidth": "1480px", "margin": "0 auto"},
            children=[

                # ── Upload status bar ──
                html.Div(id="upload-status", style={"marginBottom": "20px"}),

                # ── KPI row (dynamic) ──
                html.Div(id="kpi-row", style={"display": "flex", "gap": "16px", "flexWrap": "wrap", "marginBottom": "28px"}),

                # ── Filter bar ──
                card_wrapper(
                    children=[
                        html.Div(
                            style={"display": "flex", "alignItems": "center", "gap": "12px", "marginBottom": "14px"},
                            children=[
                                html.Span("\u2699\ufe0f", style={"fontSize": "16px"}),
                                html.Span("Global Filters", style={"fontWeight": "700", "fontSize": "14px"}),
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
                                    html.Label("Sample", style=dropdown_label_style()),
                                    dcc.Dropdown(id="filter-sample", multi=True, placeholder="Upload data first",
                                                 style={"backgroundColor": COLORS["bg"], "color": COLORS["text"]}),
                                ], style={"flex": "1", "minWidth": "200px"}),
                                html.Div([
                                    html.Label("State", style=dropdown_label_style()),
                                    dcc.Dropdown(id="filter-state", multi=True, placeholder="Upload data first",
                                                 style={"backgroundColor": COLORS["bg"]}),
                                ], style={"flex": "1", "minWidth": "180px"}),
                                html.Div([
                                    html.Label("Gene Segment", style=dropdown_label_style()),
                                    dcc.Dropdown(id="filter-segment", multi=True, placeholder="Upload data first",
                                                 style={"backgroundColor": COLORS["bg"]}),
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
                            dcc.Graph(id="genome-radar", figure=empty_fig(), config={"displayModeBar": False}),
                        ]),
                        card_wrapper([
                            section_header("Sample Collection Geography",
                                           "Geographic distribution of positive controls across the US"),
                            dcc.Graph(id="geo-map", figure=empty_fig(), config={"displayModeBar": False}),
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
                                html.Label("Metric", style=dropdown_label_style()),
                                dcc.Dropdown(
                                    id="heatmap-metric",
                                    options=[{"label": c, "value": c} for c in NUMERIC_COLS],
                                    value="TASS Score", clearable=False,
                                    style={"width": "220px", "backgroundColor": COLORS["bg"]},
                                ),
                            ]),
                        ],
                    ),
                    dcc.Graph(id="heatmap-chart", figure=empty_fig(), config={"displayModeBar": False}),
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
                            dcc.Graph(id="bar-chart", figure=empty_fig(), config={"displayModeBar": False}),
                        ]),
                        card_wrapper([
                            section_header("Distribution by State",
                                           "Statistical spread across collection sites"),
                            dcc.Graph(id="box-chart", figure=empty_fig(), config={"displayModeBar": False}),
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
                            dcc.Graph(id="coverage-gap", figure=empty_fig(), config={"displayModeBar": False}),
                        ]),
                        card_wrapper([
                            section_header("Temporal Trend",
                                           "Mean TASS Score trajectory across collection dates"),
                            dcc.Graph(id="timeline-chart", figure=empty_fig(), config={"displayModeBar": False}),
                        ]),
                    ],
                ),

                # ── Row 5: Parallel coordinates ──
                card_wrapper([
                    section_header("Multi-Dimensional QC Explorer",
                                   "Parallel coordinates view — each line is one sample-segment alignment"),
                    dcc.Graph(id="parallel-coords", figure=empty_fig(), config={"displayModeBar": False}),
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
                                html.Label("Chart Type", style=dropdown_label_style()),
                                dcc.Dropdown(
                                    id="chart-type",
                                    options=[{"label": t, "value": t} for t in CHART_TYPES],
                                    value="Scatter", clearable=False,
                                    style={"backgroundColor": COLORS["bg"]},
                                ),
                            ], style={"flex": "1", "minWidth": "130px"}),
                            html.Div([
                                html.Label("X Axis", style=dropdown_label_style()),
                                dcc.Dropdown(
                                    id="x-axis",
                                    options=[{"label": c, "value": c} for c in NUMERIC_COLS],
                                    value="% Breadth Coverage", clearable=False,
                                    style={"backgroundColor": COLORS["bg"]},
                                ),
                            ], style={"flex": "1", "minWidth": "160px"}),
                            html.Div([
                                html.Label("Y Axis", style=dropdown_label_style()),
                                dcc.Dropdown(
                                    id="y-axis",
                                    options=[{"label": c, "value": c} for c in NUMERIC_COLS],
                                    value="TASS Score", clearable=False,
                                    style={"backgroundColor": COLORS["bg"]},
                                ),
                            ], style={"flex": "1", "minWidth": "160px"}),
                            html.Div([
                                html.Label("Color By", style=dropdown_label_style()),
                                dcc.Dropdown(
                                    id="color-by",
                                    options=[{"label": c, "value": c} for c in CATEGORY_COLS],
                                    value="Sample", clearable=False,
                                    style={"backgroundColor": COLORS["bg"]},
                                ),
                            ], style={"flex": "1", "minWidth": "140px"}),
                        ],
                    ),
                    dcc.Graph(id="custom-chart", figure=empty_fig(), config={"displayModeBar": False}),
                ]),

                # ── Row 7: Data table ──
                card_wrapper([
                    section_header("Raw Data Explorer",
                                   "Filtered dataset — sortable and scrollable"),
                    html.Div(id="data-table", style={"overflowX": "auto", "maxHeight": "420px", "overflowY": "auto"}),
                ]),

                # ── Footer ──
                html.Div(
                    id="footer-info",
                    style={
                        "textAlign": "center", "padding": "28px 0 12px",
                        "fontSize": "11px", "color": COLORS["text_muted"],
                        "borderTop": f"1px solid {COLORS['card_border']}", "marginTop": "20px",
                    },
                    children=[
                        html.P("TASS Report Explorer | TaxTriage Genomic Surveillance Pipeline"),
                    ],
                ),
            ],
        ),
    ],
)


# ────────────────────────────────────────────────────────────
# HELPERS
# ────────────────────────────────────────────────────────────

def parse_uploads(contents_list, filenames_list):
    """Parse one or more uploaded CSV files and return a merged, enriched DataFrame."""
    frames = []
    for contents, fname in zip(contents_list, filenames_list):
        content_type, content_string = contents.split(",")
        decoded = base64.b64decode(content_string)
        raw = pd.read_csv(io.StringIO(decoded.decode("utf-8")))
        raw["_source_file"] = fname
        frames.append(raw)
    merged = pd.concat(frames, ignore_index=True)
    return enrich_dataframe(merged)


def df_from_store(store_data):
    """Reconstruct DataFrame from JSON store data."""
    if not store_data:
        return None
    df = pd.DataFrame(store_data)
    if "date_collected" in df.columns:
        df["date_collected"] = pd.to_datetime(df["date_collected"])
    return df


def get_sample_colors(df):
    return build_sample_colors(df)


def filtered_from_store(store_data, samples, states, segments):
    """Get a filtered DataFrame from the store."""
    df = df_from_store(store_data)
    if df is None or df.empty:
        return None
    mask = (
        df["Sample"].isin(samples if samples else df["Sample"].unique())
        & df["state"].isin(states if states else df["state"].unique())
        & df["Gene Segment"].isin(segments if segments else df["Gene Segment"].unique())
    )
    return df[mask]


# ────────────────────────────────────────────────────────────
# CALLBACK: Upload → Store
# ────────────────────────────────────────────────────────────

@callback(
    Output("data-store", "data"),
    Input("csv-upload", "contents"),
    State("csv-upload", "filename"),
    prevent_initial_call=True,
)
def on_upload(contents_list, filenames_list):
    if not contents_list:
        return no_update
    df = parse_uploads(contents_list, filenames_list)
    # Convert to JSON-serializable dict (dates as ISO strings)
    return json.loads(df.to_json(orient="records", date_format="iso"))


# ────────────────────────────────────────────────────────────
# CALLBACK: Store → Upload status + pills + KPIs + filters
# ────────────────────────────────────────────────────────────

@callback(
    Output("upload-status", "children"),
    Output("header-pills", "children"),
    Output("kpi-row", "children"),
    Output("filter-sample", "options"),
    Output("filter-sample", "value"),
    Output("filter-state", "options"),
    Output("filter-state", "value"),
    Output("filter-segment", "options"),
    Output("filter-segment", "value"),
    Output("footer-info", "children"),
    Input("data-store", "data"),
)
def update_ui_from_data(store_data):
    df = df_from_store(store_data)
    if df is None or df.empty:
        no_data_msg = card_wrapper([
            html.Div(
                style={"textAlign": "center", "padding": "40px 20px"},
                children=[
                    html.Div("\U0001f4c2", style={"fontSize": "48px", "marginBottom": "12px", "opacity": "0.5"}),
                    html.H3("No Data Loaded", style={"color": COLORS["text"], "margin": "0 0 8px 0"}),
                    html.P("Click the Upload CSV Files button in the header to get started.",
                           style={"color": COLORS["text_muted"], "fontSize": "14px", "margin": "0"}),
                    html.P("You can upload one or multiple CSV files — they will be merged automatically.",
                           style={"color": COLORS["text_muted"], "fontSize": "13px", "margin": "8px 0 0 0"}),
                ],
            ),
        ])
        return (
            no_data_msg, [], [], [], [], [], [], [], [],
            [html.P("TASS Report Explorer | TaxTriage Genomic Surveillance Pipeline")],
        )

    # Stats
    n_samples = df["Sample"].nunique()
    n_segments = df["Gene Segment"].nunique()
    mean_tass = df["TASS Score"].mean()
    mean_breadth = df["% Breadth Coverage"].mean()
    min_tass = df["TASS Score"].min()
    max_tass = df["TASS Score"].max()
    total_positions = int(df["Positions Mapped"].sum())
    n_files = df["_source_file"].nunique() if "_source_file" in df.columns else 1

    # Upload status
    file_names = sorted(df["_source_file"].unique()) if "_source_file" in df.columns else ["unknown"]
    status = card_wrapper([
        html.Div(
            style={"display": "flex", "alignItems": "center", "gap": "10px", "flexWrap": "wrap"},
            children=[
                html.Span("\u2705", style={"fontSize": "16px"}),
                html.Span(
                    f"{n_files} file{'s' if n_files > 1 else ''} loaded  \u2022  "
                    f"{len(df)} rows  \u2022  {n_samples} samples  \u2022  {n_segments} segments",
                    style={"fontWeight": "600", "fontSize": "13px"},
                ),
            ] + [
                html.Span(
                    fname,
                    style={
                        "fontSize": "11px", "padding": "2px 8px", "borderRadius": "4px",
                        "background": "rgba(0,212,170,0.1)", "color": COLORS["accent"],
                        "border": f"1px solid rgba(0,212,170,0.2)",
                    },
                )
                for fname in file_names
            ],
        ),
    ], background=f"linear-gradient(135deg, rgba(0,212,170,0.06) 0%, {COLORS['card']} 100%)")

    # Header pills
    pills = []
    if "NCBI Accession" in df.columns:
        accessions = df["NCBI Accession"].unique()
        for acc in accessions[:3]:
            pills.append(html.Span(
                f"NCBI: {acc}",
                style={
                    "fontSize": "11px", "padding": "3px 10px", "borderRadius": "20px",
                    "background": "rgba(0,212,170,0.12)", "color": COLORS["accent"],
                    "border": "1px solid rgba(0,212,170,0.25)",
                },
            ))
    if "TaxTriage Run Date" in df.columns:
        for rd in df["TaxTriage Run Date"].unique()[:2]:
            pills.append(html.Span(
                f"Run Date: {rd}",
                style={
                    "fontSize": "11px", "padding": "3px 10px", "borderRadius": "20px",
                    "background": "rgba(124,58,237,0.12)", "color": COLORS["accent2"],
                    "border": "1px solid rgba(124,58,237,0.25)",
                },
            ))
    if "Organism" in df.columns:
        for org in df["Organism"].unique()[:2]:
            pills.append(html.Span(
                f"Organism: {org}",
                style={
                    "fontSize": "11px", "padding": "3px 10px", "borderRadius": "20px",
                    "background": "rgba(245,158,11,0.12)", "color": COLORS["accent3"],
                    "border": "1px solid rgba(245,158,11,0.25)",
                },
            ))

    # KPIs
    kpis = [
        kpi_card("Samples Analyzed", str(n_samples), f"{n_segments} genome segments each", COLORS["accent"], "\U0001f9ea"),
        kpi_card("Mean TASS Score", f"{mean_tass:.2f}", f"Range: {min_tass:.2f} \u2013 {max_tass:.2f}", COLORS["accent2"], "\u2b50"),
        kpi_card("Mean Breadth Coverage", f"{mean_breadth:.1f}%", "Across all segments", COLORS["accent3"], "\U0001f4ca"),
        kpi_card("Total Positions Mapped", f"{total_positions:,}", "Sum across all alignments", COLORS["success"], "\U0001f9ec"),
        kpi_card("Pipeline QC Status",
                 "PASS" if min_tass > 90 else "REVIEW",
                 "All scores > 90" if min_tass > 90 else "Some scores below threshold",
                 COLORS["success"] if min_tass > 90 else COLORS["danger"], "\u2705"),
    ]

    # Filter options
    samples_sorted = sorted(df["Sample"].unique())
    states_sorted = sorted(df["state"].unique())
    segments_sorted = sorted(df["Gene Segment"].unique())

    sample_opts = [{"label": s, "value": s} for s in samples_sorted]
    state_opts = [{"label": s, "value": s} for s in states_sorted]
    seg_opts = [{"label": f"{s} ({SEGMENT_ANNOTATION.get('HM7732' + str(6 + i + 1), {}).get('protein', '')})" if s in SEGMENT_COLORS else s, "value": s}
                for i, s in enumerate(["PB2", "PB1", "PA", "HA", "NP", "NA"]) if s in segments_sorted]
    # Also add any segments not in the standard list
    standard_segs = {"PB2", "PB1", "PA", "HA", "NP", "NA"}
    for s in segments_sorted:
        if s not in standard_segs:
            seg_opts.append({"label": s, "value": s})

    # Footer
    footer = [html.P("TASS Report Explorer | TaxTriage Genomic Surveillance Pipeline")]
    if "NCBI Accession" in df.columns and "Organism" in df.columns:
        footer.append(html.P(
            f"Data: {', '.join(df['NCBI Accession'].unique()[:3])} | "
            f"{', '.join(df['Organism'].unique()[:2])} | "
            f"{n_files} file{'s' if n_files > 1 else ''} merged"
        ))

    return (
        status, pills, kpis,
        sample_opts, samples_sorted,
        state_opts, states_sorted,
        seg_opts, [o["value"] for o in seg_opts],
        footer,
    )


# ────────────────────────────────────────────────────────────
# CHART CALLBACKS (all read from data-store)
# ────────────────────────────────────────────────────────────

# ── Genome radar ──
@callback(
    Output("genome-radar", "figure"),
    Input("data-store", "data"),
    Input("filter-sample", "value"),
    Input("filter-state", "value"),
    Input("filter-segment", "value"),
)
def update_radar(store_data, samples, states, segments):
    dff = filtered_from_store(store_data, samples, states, segments)
    if dff is None or dff.empty:
        return empty_fig()

    sample_colors = get_sample_colors(dff)
    fig = go.Figure()
    segment_order = ["PB2", "PB1", "PA", "HA", "NP", "NA"]

    for sample in sorted(dff["Sample"].unique()):
        sub = dff[dff["Sample"] == sample].set_index("Gene Segment")
        vals = []
        for seg in segment_order:
            if seg in sub.index:
                v = sub.loc[seg, "TASS Score"]
                vals.append(v if isinstance(v, (int, float, np.floating)) else v.iloc[0])
            else:
                vals.append(None)
        label = sample.replace("positive_control_", "Ctrl ")
        fig.add_trace(go.Scatterpolar(
            r=vals + [vals[0]] if vals[0] is not None else vals,
            theta=segment_order + [segment_order[0]],
            name=label,
            line=dict(color=sample_colors.get(sample, "#888"), width=2),
            fill="toself",
            opacity=0.85,
        ))

    min_score = dff["TASS Score"].min()
    max_score = dff["TASS Score"].max()
    range_lo = max(0, min_score - 2)
    range_hi = min(100, max_score + 1)

    fig.update_layout(
        **dark_layout(
            margin=dict(l=60, r=60, t=30, b=30), height=380,
            polar=dict(
                bgcolor="rgba(0,0,0,0)",
                radialaxis=dict(visible=True, range=[range_lo, range_hi], gridcolor=COLORS["grid"],
                                tickfont=dict(size=10, color=COLORS["text_muted"])),
                angularaxis=dict(gridcolor=COLORS["grid"], tickfont=dict(size=12, color=COLORS["text"])),
            ),
            showlegend=True,
            legend=dict(font=dict(size=10), orientation="h", y=-0.15, x=0.5, xanchor="center"),
        ),
    )
    return fig


# ── Geo map ──
@callback(
    Output("geo-map", "figure"),
    Input("data-store", "data"),
    Input("filter-sample", "value"),
    Input("filter-state", "value"),
    Input("filter-segment", "value"),
)
def update_geo(store_data, samples, states, segments):
    dff = filtered_from_store(store_data, samples, states, segments)
    if dff is None or dff.empty:
        return empty_fig()

    sample_colors = get_sample_colors(dff)
    agg = dff.groupby(["Sample", "state", "date_collected"]).agg(
        mean_tass=("TASS Score", "mean"),
        mean_breadth=("% Breadth Coverage", "mean"),
    ).reset_index()
    agg["lat"] = agg["state"].map(lambda s: STATE_COORDS.get(s, {}).get("lat"))
    agg["lon"] = agg["state"].map(lambda s: STATE_COORDS.get(s, {}).get("lon"))
    agg = agg.dropna(subset=["lat", "lon"])

    if agg.empty:
        return empty_fig("No geographic coordinates for selected states")

    fig = go.Figure()
    for _, row in agg.iterrows():
        label = row["Sample"].replace("positive_control_", "Ctrl ")
        fig.add_trace(go.Scattergeo(
            lat=[row["lat"]], lon=[row["lon"]],
            text=f"<b>{label}</b><br>{row['state']}<br>"
                 f"Collected: {row['date_collected'].strftime('%b %d, %Y')}<br>"
                 f"Mean TASS: {row['mean_tass']:.2f}<br>"
                 f"Mean Breadth: {row['mean_breadth']:.1f}%",
            hoverinfo="text",
            marker=dict(
                size=max(6, row["mean_tass"] - 90),
                sizemin=6, sizemode="diameter",
                color=sample_colors.get(row["Sample"], COLORS["accent"]),
                opacity=0.9, line=dict(width=1.5, color="white"),
            ),
            name=label, showlegend=False,
        ))

    fig.update_layout(
        **dark_layout(height=380, margin=dict(l=0, r=0, t=10, b=0)),
        geo=dict(
            scope="usa", bgcolor="rgba(0,0,0,0)", lakecolor="rgba(0,0,0,0)",
            landcolor=COLORS["card"], subunitcolor=COLORS["card_border"],
            showlakes=False, coastlinecolor=COLORS["card_border"],
        ),
    )
    return fig


# ── Heatmap ──
@callback(
    Output("heatmap-chart", "figure"),
    Input("data-store", "data"),
    Input("heatmap-metric", "value"),
    Input("filter-sample", "value"),
    Input("filter-state", "value"),
    Input("filter-segment", "value"),
)
def update_heatmap(store_data, metric, samples, states, segments):
    dff = filtered_from_store(store_data, samples, states, segments)
    if dff is None or dff.empty:
        return empty_fig()

    segment_order = [s for s in ["PB2", "PB1", "PA", "HA", "NP", "NA"] if s in dff["Gene Segment"].values]
    extra_segs = [s for s in sorted(dff["Gene Segment"].unique()) if s not in segment_order]
    segment_order += extra_segs

    sample_order = sorted(dff["Sample"].unique())
    pivot = dff.pivot_table(index="Sample", columns="Gene Segment", values=metric, aggfunc="mean")
    pivot = pivot.reindex(index=sample_order, columns=segment_order)
    labels = [[f"{v:.2f}" if pd.notna(v) else "" for v in row] for row in pivot.values]
    sample_labels = [s.replace("positive_control_", "Control ") for s in pivot.index]

    fig = go.Figure(go.Heatmap(
        z=pivot.values, x=pivot.columns.tolist(), y=sample_labels,
        text=labels, texttemplate="%{text}", textfont=dict(size=12, color="white"),
        colorscale=[[0, "#1e1b4b"], [0.25, "#3730a3"], [0.5, "#7c3aed"], [0.75, "#a78bfa"], [1, "#00d4aa"]],
        colorbar=dict(title=dict(text=metric, font=dict(size=11)), thickness=12, len=0.8),
        hovertemplate=f"<b>%{{y}}</b><br>Segment: %{{x}}<br>{metric}: %{{z:.3f}}<extra></extra>",
    ))
    fig.update_layout(**dark_layout(height=max(250, 50 * len(sample_order)), margin=dict(l=100, r=30, t=20, b=50)))
    return fig


# ── Bar chart ──
@callback(
    Output("bar-chart", "figure"),
    Input("data-store", "data"),
    Input("bar-metric", "value"),
    Input("filter-sample", "value"),
    Input("filter-state", "value"),
    Input("filter-segment", "value"),
)
def update_bar(store_data, metric, samples, states, segments):
    dff = filtered_from_store(store_data, samples, states, segments)
    if dff is None or dff.empty:
        return empty_fig()

    dff = dff.copy()
    dff["Sample Label"] = dff["Sample"].str.replace("positive_control_", "Ctrl ")
    fig = px.bar(
        dff, x="Sample Label", y=metric, color="Gene Segment", barmode="group",
        color_discrete_map=SEGMENT_COLORS, hover_data=["state", "Protein", "Ref Size"],
    )
    fig.update_layout(**dark_layout(height=360, margin=dict(l=60, r=20, t=20, b=50)))
    fig.update_xaxes(title="")
    return fig


# ── Box chart ──
@callback(
    Output("box-chart", "figure"),
    Input("data-store", "data"),
    Input("bar-metric", "value"),
    Input("filter-sample", "value"),
    Input("filter-state", "value"),
    Input("filter-segment", "value"),
)
def update_box(store_data, metric, samples, states, segments):
    dff = filtered_from_store(store_data, samples, states, segments)
    if dff is None or dff.empty:
        return empty_fig()

    fig = px.box(dff, x="state", y=metric, color="state", points="all",
                 hover_data=["Sample", "Gene Segment"])
    fig.update_layout(**dark_layout(height=360, margin=dict(l=60, r=20, t=20, b=50), showlegend=False))
    fig.update_xaxes(title="")
    return fig


# ── Coverage gap lollipop ──
@callback(
    Output("coverage-gap", "figure"),
    Input("data-store", "data"),
    Input("filter-sample", "value"),
    Input("filter-state", "value"),
    Input("filter-segment", "value"),
)
def update_coverage_gap(store_data, samples, states, segments):
    dff = filtered_from_store(store_data, samples, states, segments)
    if dff is None or dff.empty:
        return empty_fig()

    seg_order = [s for s in ["PB2", "PB1", "PA", "HA", "NP", "NA"] if s in dff["Gene Segment"].values]
    extra = [s for s in sorted(dff["Gene Segment"].unique()) if s not in seg_order]
    seg_order += extra

    agg = dff.groupby("Gene Segment").agg(
        mean_gap=("Coverage Gap (%)", "mean"),
        std_gap=("Coverage Gap (%)", "std"),
    ).reindex(seg_order).reset_index()
    agg["std_gap"] = agg["std_gap"].fillna(0)

    fig = go.Figure()
    for _, row in agg.iterrows():
        color = SEGMENT_COLORS.get(row["Gene Segment"], "#888")
        fig.add_trace(go.Scatter(
            x=[row["mean_gap"]], y=[row["Gene Segment"]],
            mode="markers+text",
            marker=dict(size=14, color=color, line=dict(width=2, color="white")),
            text=f"  {row['mean_gap']:.1f}%", textposition="middle right",
            textfont=dict(size=12, color=color), showlegend=False,
            hovertemplate=f"<b>{row['Gene Segment']}</b><br>Mean gap: {row['mean_gap']:.2f}%<br>"
                          f"Std: {row['std_gap']:.2f}%<extra></extra>",
        ))
        fig.add_shape(
            type="line", x0=0, x1=row["mean_gap"],
            y0=row["Gene Segment"], y1=row["Gene Segment"],
            line=dict(color=color, width=3, dash="dot"),
        )

    max_gap = agg["mean_gap"].max() if not agg.empty else 10
    fig.update_layout(**dark_layout(
        height=360, margin=dict(l=60, r=40, t=20, b=50),
        xaxis=dict(title="Mean Coverage Gap (%)", gridcolor=COLORS["grid"], range=[0, max_gap * 1.4]),
        yaxis=dict(title=""),
    ))
    return fig


# ── Timeline ──
@callback(
    Output("timeline-chart", "figure"),
    Input("data-store", "data"),
    Input("bar-metric", "value"),
    Input("filter-sample", "value"),
    Input("filter-state", "value"),
    Input("filter-segment", "value"),
)
def update_timeline(store_data, metric, samples, states, segments):
    dff = filtered_from_store(store_data, samples, states, segments)
    if dff is None or dff.empty:
        return empty_fig()

    agg = dff.groupby(["Sample", "date_collected", "state"])[metric].mean().reset_index()
    agg = agg.sort_values("date_collected")

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=agg["date_collected"], y=agg[metric], mode="lines",
        line=dict(color=COLORS["accent"], width=0),
        fill="tozeroy", fillcolor="rgba(0,212,170,0.08)",
        showlegend=False, hoverinfo="skip",
    ))
    fig.add_trace(go.Scatter(
        x=agg["date_collected"], y=agg[metric], mode="lines+markers+text",
        line=dict(color=COLORS["accent"], width=3),
        marker=dict(size=10, color=COLORS["accent"], line=dict(width=2, color="white")),
        text=agg["state"], textposition="top center",
        textfont=dict(size=10, color=COLORS["text_muted"]),
        hovertemplate="<b>%{text}</b><br>Date: %{x|%b %d, %Y}<br>" + metric + ": %{y:.3f}<extra></extra>",
        showlegend=False,
    ))
    fig.update_layout(**dark_layout(
        height=360, margin=dict(l=60, r=30, t=20, b=50),
        xaxis=dict(title="Collection Date"), yaxis=dict(title=f"Mean {metric}"),
    ))
    return fig


# ── Parallel coordinates ──
@callback(
    Output("parallel-coords", "figure"),
    Input("data-store", "data"),
    Input("filter-sample", "value"),
    Input("filter-state", "value"),
    Input("filter-segment", "value"),
)
def update_parallel(store_data, samples, states, segments):
    dff = filtered_from_store(store_data, samples, states, segments)
    if dff is None or dff.empty:
        return empty_fig()

    dff = dff.copy()
    sample_colors = get_sample_colors(dff)
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

    color_vals = list(sample_colors.values())
    n = len(sample_map)
    colorscale = []
    for i in range(n):
        lo = i / max(n, 1)
        hi = (i + 1) / max(n, 1)
        c = color_vals[i % len(color_vals)]
        colorscale.append([lo, c])
        colorscale.append([hi, c])

    fig = go.Figure(go.Parcoords(
        line=dict(color=dff["sample_code"], colorscale=colorscale, showscale=False),
        dimensions=dims,
        labelfont=dict(size=11, color=COLORS["text"]),
        tickfont=dict(size=10, color=COLORS["text_muted"]),
    ))
    fig.update_layout(**dark_layout(height=400, margin=dict(l=60, r=30, t=30, b=40)))
    return fig


# ── Custom chart ──
@callback(
    Output("custom-chart", "figure"),
    Input("data-store", "data"),
    Input("chart-type", "value"),
    Input("x-axis", "value"),
    Input("y-axis", "value"),
    Input("color-by", "value"),
    Input("filter-sample", "value"),
    Input("filter-state", "value"),
    Input("filter-segment", "value"),
)
def update_custom(store_data, chart_type, x, y, color, samples, states, segments):
    dff = filtered_from_store(store_data, samples, states, segments)
    if dff is None or dff.empty:
        return empty_fig()

    sample_colors = get_sample_colors(dff)
    color_map = SEGMENT_COLORS if color == "Gene Segment" else sample_colors if color == "Sample" else None
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
    Input("data-store", "data"),
    Input("filter-sample", "value"),
    Input("filter-state", "value"),
    Input("filter-segment", "value"),
)
def update_table(store_data, samples, states, segments):
    dff = filtered_from_store(store_data, samples, states, segments)
    if dff is None or dff.empty:
        return html.P("No data loaded.", style={"color": COLORS["text_muted"], "textAlign": "center", "padding": "30px"})

    display_cols = ["Sample", "Gene Segment", "Protein", "state", "date_collected"] + NUMERIC_COLS
    if "_source_file" in dff.columns:
        display_cols = ["_source_file"] + display_cols
    cols = [c for c in display_cols if c in dff.columns]

    header_style = {
        "border": f"1px solid {COLORS['card_border']}", "padding": "8px 12px",
        "background": "#12141f", "textAlign": "left", "fontSize": "11px",
        "fontWeight": "600", "color": COLORS["text_muted"],
        "textTransform": "uppercase", "letterSpacing": "0.5px",
        "position": "sticky", "top": "0", "zIndex": "1",
    }
    cell_style = {
        "border": f"1px solid {COLORS['card_border']}", "padding": "6px 12px",
        "fontSize": "12px", "color": COLORS["text"],
    }

    col_labels = {"_source_file": "Source File"}

    return html.Table(
        style={"borderCollapse": "collapse", "width": "100%"},
        children=[
            html.Thead(html.Tr([html.Th(col_labels.get(c, c), style=header_style) for c in cols])),
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
