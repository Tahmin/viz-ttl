#!/usr/bin/env python3
"""Interactive TASS report dashboard using Plotly Dash."""

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from dash import Dash, dcc, html, Input, Output, callback

CSV_PATH = "tass_report_mqc.csv"

# ── Numeric columns available for axis selection ──
NUMERIC_COLS = [
    "TASS Score",
    "Ref Size",
    "Mapped Reads",
    "ANIr",
    "% Reads Aligned",
    "Average Read Length",
    "Average MAPQ",
    "Positions Mapped",
    "% Breadth Coverage",
    "Mean Depth",
    "Depth Stdev",
    "Depth Evenness (Gini)",
]

# ── Categorical columns for color / grouping ──
CATEGORY_COLS = ["Sample", "Segment", "state", "Organism"]

CHART_TYPES = ["Scatter", "Bar", "Box", "Histogram", "Heatmap"]


def load_data(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    df["date_collected"] = pd.to_datetime(df["date_collected"], format="%m/%d/%y")
    df["Segment"] = df["Reference Contig"].str.extract(r"\|(\w+)\|")[0]
    return df


df = load_data(CSV_PATH)

# ── Dash app ──
app = Dash(__name__)
app.title = "TASS Report Explorer"

app.layout = html.Div(
    style={"fontFamily": "sans-serif", "margin": "30px 40px"},
    children=[
        html.H1("TASS Report Explorer"),
        html.P("Use the controls below to build custom visualizations from the dataset."),

        # ──────────── Filter panel ────────────
        html.Div(
            style={
                "display": "flex",
                "gap": "30px",
                "flexWrap": "wrap",
                "padding": "16px",
                "background": "#f5f7fa",
                "borderRadius": "8px",
                "marginBottom": "24px",
            },
            children=[
                html.Div([
                    html.Label("Filter by Sample", style={"fontWeight": "bold"}),
                    dcc.Dropdown(
                        id="filter-sample",
                        options=[{"label": s, "value": s} for s in sorted(df["Sample"].unique())],
                        value=sorted(df["Sample"].unique()),
                        multi=True,
                        placeholder="All samples",
                    ),
                ], style={"minWidth": "220px", "flex": "1"}),

                html.Div([
                    html.Label("Filter by State", style={"fontWeight": "bold"}),
                    dcc.Dropdown(
                        id="filter-state",
                        options=[{"label": s, "value": s} for s in sorted(df["state"].unique())],
                        value=sorted(df["state"].unique()),
                        multi=True,
                        placeholder="All states",
                    ),
                ], style={"minWidth": "180px", "flex": "1"}),

                html.Div([
                    html.Label("Filter by Segment", style={"fontWeight": "bold"}),
                    dcc.Dropdown(
                        id="filter-segment",
                        options=[{"label": s, "value": s} for s in sorted(df["Segment"].unique())],
                        value=sorted(df["Segment"].unique()),
                        multi=True,
                        placeholder="All segments",
                    ),
                ], style={"minWidth": "180px", "flex": "1"}),
            ],
        ),

        # ──────────── Custom chart builder ────────────
        html.Div(
            style={
                "padding": "16px",
                "background": "#eef1f6",
                "borderRadius": "8px",
                "marginBottom": "24px",
            },
            children=[
                html.H3("Custom Chart Builder"),
                html.Div(
                    style={"display": "flex", "gap": "24px", "flexWrap": "wrap"},
                    children=[
                        html.Div([
                            html.Label("Chart Type", style={"fontWeight": "bold"}),
                            dcc.Dropdown(
                                id="chart-type",
                                options=[{"label": t, "value": t} for t in CHART_TYPES],
                                value="Scatter",
                                clearable=False,
                            ),
                        ], style={"minWidth": "140px", "flex": "1"}),

                        html.Div([
                            html.Label("X Axis", style={"fontWeight": "bold"}),
                            dcc.Dropdown(
                                id="x-axis",
                                options=[{"label": c, "value": c} for c in NUMERIC_COLS],
                                value="% Breadth Coverage",
                                clearable=False,
                            ),
                        ], style={"minWidth": "180px", "flex": "1"}),

                        html.Div([
                            html.Label("Y Axis", style={"fontWeight": "bold"}),
                            dcc.Dropdown(
                                id="y-axis",
                                options=[{"label": c, "value": c} for c in NUMERIC_COLS],
                                value="TASS Score",
                                clearable=False,
                            ),
                        ], style={"minWidth": "180px", "flex": "1"}),

                        html.Div([
                            html.Label("Color By", style={"fontWeight": "bold"}),
                            dcc.Dropdown(
                                id="color-by",
                                options=[{"label": c, "value": c} for c in CATEGORY_COLS],
                                value="Sample",
                                clearable=False,
                            ),
                        ], style={"minWidth": "160px", "flex": "1"}),
                    ],
                ),
            ],
        ),

        # ──────────── Custom chart output ────────────
        dcc.Graph(id="custom-chart", style={"marginBottom": "40px"}),

        html.Hr(),

        # ──────────── Pre-built charts ────────────
        html.H2("Pre-built Charts"),

        html.Div([
            html.Label("Y-axis metric for bar chart", style={"fontWeight": "bold"}),
            dcc.Dropdown(
                id="bar-metric",
                options=[{"label": c, "value": c} for c in NUMERIC_COLS],
                value="TASS Score",
                clearable=False,
                style={"maxWidth": "300px"},
            ),
        ], style={"marginBottom": "12px"}),
        dcc.Graph(id="bar-chart"),

        html.Div([
            html.Label("Heatmap metric", style={"fontWeight": "bold"}),
            dcc.Dropdown(
                id="heatmap-metric",
                options=[{"label": c, "value": c} for c in NUMERIC_COLS],
                value="% Breadth Coverage",
                clearable=False,
                style={"maxWidth": "300px"},
            ),
        ], style={"marginBottom": "12px", "marginTop": "32px"}),
        dcc.Graph(id="heatmap-chart"),

        dcc.Graph(id="box-chart", style={"marginTop": "32px"}),
        dcc.Graph(id="timeline-chart", style={"marginTop": "32px"}),

        # ──────────── Data table ────────────
        html.Hr(),
        html.H2("Filtered Data Table"),
        html.Div(id="data-table", style={"overflowX": "auto", "marginBottom": "40px"}),
    ],
)


# ── Helper: apply filters ──
def filtered_df(samples, states, segments):
    mask = (
        df["Sample"].isin(samples or df["Sample"].unique())
        & df["state"].isin(states or df["state"].unique())
        & df["Segment"].isin(segments or df["Segment"].unique())
    )
    return df[mask]


# ── Callback: custom chart ──
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
def update_custom_chart(chart_type, x, y, color, samples, states, segments):
    dff = filtered_df(samples, states, segments)
    title = f"{chart_type}: {y} vs {x} (color: {color})"

    if chart_type == "Scatter":
        fig = px.scatter(dff, x=x, y=y, color=color, title=title,
                         hover_data=["Sample", "Segment", "state"])
    elif chart_type == "Bar":
        fig = px.bar(dff, x=x, y=y, color=color, barmode="group", title=title)
    elif chart_type == "Box":
        fig = px.box(dff, x=color, y=y, color=color, points="all", title=f"Box: {y} by {color}")
    elif chart_type == "Histogram":
        fig = px.histogram(dff, x=x, color=color, title=f"Histogram: {x} (color: {color})",
                           barmode="overlay", opacity=0.7)
    elif chart_type == "Heatmap":
        pivot = dff.pivot_table(index="Sample", columns="Segment", values=y, aggfunc="mean")
        fig = px.imshow(pivot, text_auto=".2f", color_continuous_scale="YlGnBu",
                        title=f"Heatmap: {y} (Sample x Segment)", aspect="auto")
    else:
        fig = go.Figure()

    fig.update_layout(height=500)
    return fig


# ── Callback: bar chart ──
@callback(
    Output("bar-chart", "figure"),
    Input("bar-metric", "value"),
    Input("filter-sample", "value"),
    Input("filter-state", "value"),
    Input("filter-segment", "value"),
)
def update_bar(metric, samples, states, segments):
    dff = filtered_df(samples, states, segments)
    fig = px.bar(
        dff, x="Sample", y=metric, color="Segment", barmode="group",
        title=f"{metric} by Sample and Segment",
        hover_data=["state", "date_collected"],
    )
    fig.update_layout(xaxis_tickangle=-30, height=450)
    return fig


# ── Callback: heatmap ──
@callback(
    Output("heatmap-chart", "figure"),
    Input("heatmap-metric", "value"),
    Input("filter-sample", "value"),
    Input("filter-state", "value"),
    Input("filter-segment", "value"),
)
def update_heatmap(metric, samples, states, segments):
    dff = filtered_df(samples, states, segments)
    pivot = dff.pivot_table(index="Sample", columns="Segment", values=metric, aggfunc="mean")
    fig = px.imshow(
        pivot, text_auto=".2f", color_continuous_scale="YlGnBu",
        title=f"{metric} by Sample and Segment",
        labels=dict(color=metric), aspect="auto",
    )
    fig.update_layout(height=400)
    return fig


# ── Callback: box chart ──
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
        title=f"{metric} Distribution by State",
        hover_data=["Sample", "Segment"],
    )
    fig.update_layout(showlegend=False, height=400)
    return fig


# ── Callback: timeline ──
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
    fig = px.line(
        agg, x="date_collected", y=metric, markers=True, text="state",
        title=f"Mean {metric} Over Collection Date",
        labels={"date_collected": "Collection Date", metric: f"Mean {metric}"},
    )
    fig.update_traces(textposition="top center")
    fig.update_layout(height=400)
    return fig


# ── Callback: data table ──
@callback(
    Output("data-table", "children"),
    Input("filter-sample", "value"),
    Input("filter-state", "value"),
    Input("filter-segment", "value"),
)
def update_table(samples, states, segments):
    dff = filtered_df(samples, states, segments)
    display_cols = ["Sample", "Segment", "state", "date_collected"] + NUMERIC_COLS
    cols = [c for c in display_cols if c in dff.columns]
    return html.Table(
        style={"borderCollapse": "collapse", "width": "100%", "fontSize": "13px"},
        children=[
            html.Thead(html.Tr([
                html.Th(c, style={"border": "1px solid #ccc", "padding": "6px 10px",
                                   "background": "#e8ecf1", "textAlign": "left"})
                for c in cols
            ])),
            html.Tbody([
                html.Tr([
                    html.Td(
                        str(row[c]) if not pd.isna(row[c]) else "",
                        style={"border": "1px solid #ddd", "padding": "4px 10px"},
                    )
                    for c in cols
                ])
                for _, row in dff[cols].iterrows()
            ]),
        ],
    )


if __name__ == "__main__":
    print("Starting TASS Report Explorer at http://127.0.0.1:8050")
    app.run(debug=True)
