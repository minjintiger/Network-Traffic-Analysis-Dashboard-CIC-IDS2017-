# Created by Minjin Kim on 2025.11.30
# CS 5764 – Final Term Project

import numpy as np
import pandas as pd
import scipy.stats as stats
from scipy.stats import shapiro, kstest, normaltest
import gc

import dash
from dash import dcc, html, dash_table
from dash.dependencies import Input, Output, State
import dash_bootstrap_components as dbc

import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.decomposition import PCA

# -------------------------------------------------------------------
# 1. Global Configuration & Data Loading
# -------------------------------------------------------------------

pd.options.display.float_format = lambda x: f"{x:.2f}"
DATA_PATH = "Wednesday-workingHours.pcap_ISCX.csv"

NUMERIC_COLS = [
    "Flow Duration", "Total Fwd Packets", "Total Backward Packets",
    "Total Length of Fwd Packets", "Total Length of Bwd Packets",
    "Fwd Packet Length Mean", "Bwd Packet Length Mean",
    "Flow Bytes/s", "Flow Packets/s", "Flow IAT Mean",
    "Flow IAT Std", "Fwd Packets/s", "Bwd Packets/s",
    "Average Packet Size", "Packet Length Variance",
    "Active Mean", "Idle Mean"
]

CAT_COLS = [
    "Destination Port", "Protocol", "Fwd PSH Flags", "Bwd PSH Flags",
    "Fwd URG Flags", "Bwd URG Flags", "FIN Flag Count", "SYN Flag Count",
    "RST Flag Count", "PSH Flag Count", "ACK Flag Count", "URG Flag Count",
    "CWE Flag Count", "ECE Flag Count", "Label"
]

CONST_FLAG_COLS = ["Bwd PSH Flags", "Fwd URG Flags", "Bwd URG Flags", "CWE Flag Count"]
PORT_BUCKET_COL = "Destination Port Bucket"


def reduce_mem_usage(df):
    start_mem = df.memory_usage().sum() / 1024 ** 2
    print(f'Memory usage of dataframe is {start_mem:.2f} MB')

    for col in df.columns:
        col_type = df[col].dtype

        if col_type != object:
            c_min = df[col].min()
            c_max = df[col].max()
            if str(col_type)[:3] == 'int':
                if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                    df[col] = df[col].astype(np.int8)
                elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                    df[col] = df[col].astype(np.int16)
                elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                    df[col] = df[col].astype(np.int32)
                elif c_min > np.iinfo(np.int64).min and c_max < np.iinfo(np.int64).max:
                    df[col] = df[col].astype(np.int64)
            else:
                df[col] = df[col].astype(np.float32)
        else:
            num_unique_values = len(df[col].unique())
            num_total_values = len(df[col])
            if num_unique_values / num_total_values < 0.5:
                df[col] = df[col].astype('category')

    end_mem = df.memory_usage().sum() / 1024 ** 2
    print(f'Memory usage after optimization is: {end_mem:.2f} MB')
    print(f'Decreased by {100 * (start_mem - end_mem) / start_mem:.1f}%')

    return df


def add_port_bucket_column(df: pd.DataFrame) -> pd.DataFrame:
    if "Destination Port" not in df.columns:
        return df
    port = pd.to_numeric(df["Destination Port"], errors="coerce")
    bins = [-1, 1023, 49151, 65535]
    labels = ["Well-known (0–1023)", "Registered (1024–49151)", "Dynamic (49152–65535)"]

    df[PORT_BUCKET_COL] = pd.cut(port, bins=bins, labels=labels)
    df[PORT_BUCKET_COL] = df[PORT_BUCKET_COL].astype("category")
    return df


def load_raw_dataset(path: str) -> pd.DataFrame:

    df = pd.read_csv(path)
    df.columns = df.columns.str.strip()
    cols_present = [c for c in (NUMERIC_COLS + CAT_COLS) if c in df.columns]
    df = df[cols_present]

    df = reduce_mem_usage(df)

    df = add_port_bucket_column(df)

    return df


print("Loading and optimizing data...")
df_raw = load_raw_dataset(DATA_PATH)
print("Data loaded successfully.")

numeric_options = [{"label": c, "value": c} for c in NUMERIC_COLS if c in df_raw.columns]
cat_options = [{"label": c, "value": c} for c in (CAT_COLS + [PORT_BUCKET_COL])
               if c in df_raw.columns
               and c != "Destination Port"
               and c not in CONST_FLAG_COLS]


# -------------------------------------------------------------------
# 2. Data Processing Logic
# -------------------------------------------------------------------

def clean_dataset(df: pd.DataFrame, method: str = "basic") -> pd.DataFrame:
    df_clean = df.copy()

    num_cols = df_clean.select_dtypes(include=[np.number]).columns
    df_clean[num_cols] = df_clean[num_cols].replace([np.inf, -np.inf], np.nan)

    if method in ("basic", "strict"):
        cols_nonpositive_to_nan = ["Flow Duration", "Flow IAT Mean", "Flow Bytes/s", "Flow Packets/s"]
        for col in cols_nonpositive_to_nan:
            if col in df_clean.columns:
                mask = df_clean[col] <= 0
                if mask.any():
                    df_clean.loc[mask, col] = np.nan

        critical = [c for c in ["Flow Duration", "Flow Bytes/s"] if c in df_clean.columns]
        if critical:
            df_clean = df_clean.dropna(subset=critical)

    if method == "strict":
        numeric_present = [c for c in NUMERIC_COLS if c in df_clean.columns]
        df_clean = df_clean.dropna(subset=numeric_present)
    elif method == "fill_mean":
        numeric_present = [c for c in NUMERIC_COLS if c in df_clean.columns]
        df_clean[numeric_present] = df_clean[numeric_present].fillna(df_clean[numeric_present].mean())

    if "Label" in df_clean.columns and df_clean["Label"].dtype == 'object':
        df_clean["Label"] = df_clean["Label"].astype("category")

    print(df_clean.head())
    return df_clean


def apply_outlier_method(df: pd.DataFrame, method: str = "none") -> pd.DataFrame:
    if method == "none":
        return df

    cols = [c for c in ["Flow Duration", "Flow Bytes/s", "Average Packet Size"] if c in df.columns]
    if not cols:
        return df

    # df_out = df.copy()
    df_out = df

    mask = pd.Series(True, index=df_out.index)

    if method == "iqr":
        for col in cols:
            q1 = df_out[col].quantile(0.25)
            q3 = df_out[col].quantile(0.75)
            iqr = q3 - q1
            mask &= df_out[col].between(q1 - 1.5 * iqr, q3 + 1.5 * iqr)
    elif method == "zscore":
        for col in cols:
            z = np.abs(stats.zscore(df_out[col].fillna(0)))
            mask &= (z < 3)

    return df_out[mask]


def apply_transform_method(df, method="none"):
    if method == "none":
        return df

    df_tr = df.copy()
    num_cols = [c for c in NUMERIC_COLS if c in df_tr.columns]

    if method == "log1p":
        for col in num_cols:
            if df_tr[col].min() >= 0:
                df_tr[col] = np.log1p(df_tr[col])
    elif method == "minmax":
        scaler = MinMaxScaler()
        df_tr[num_cols] = scaler.fit_transform(df_tr[num_cols])
    elif method == "standard":
        scaler = StandardScaler()
        df_tr[num_cols] = scaler.fit_transform(df_tr[num_cols])

    return df_tr


def get_processed_df(clean, outlier, transform, range_val=None):
    gc.collect()

    df = clean_dataset(df_raw, method=clean)
    df = apply_outlier_method(df, method=outlier)
    df = apply_transform_method(df, method=transform)

    if range_val and "Flow Duration" in df.columns:
        min_p, max_p = range_val
        if len(df) > 0:
            low_val = np.percentile(df["Flow Duration"], min_p)
            high_val = np.percentile(df["Flow Duration"], max_p)
            df = df[(df["Flow Duration"] >= low_val) & (df["Flow Duration"] <= high_val)]

    return df


# -------------------------------------------------------------------
# 3. Helper Functions (Including Normality Logic)
# -------------------------------------------------------------------

def make_warning_figure(msg: str) -> go.Figure:
    fig = go.Figure()
    fig.add_annotation(text=msg, showarrow=False, font=dict(size=16))
    fig.update_xaxes(visible=False)
    fig.update_yaxes(visible=False)
    return fig


def perform_normality_tests(data, title, method="all"):
    results = []
    alpha = 0.01

    sample_size = 2000
    sample_data = data if len(data) < sample_size else np.random.choice(data, sample_size, replace=False)

    if method in ["all", "shapiro"]:
        stat, p = shapiro(sample_data)
        res_str = f"Shapiro test: {title} : stat={stat:.2f} p={p:.4f}"
        res_str += " -> Normal" if p > alpha else " -> NOT Normal"
        results.append(res_str)

    if method in ["all", "ks"]:
        mean = np.mean(data)
        std = np.std(data)
        stat, p = kstest(data, 'norm', args=(mean, std))
        res_str = f"K-S test: {title} : stat={stat:.2f} p={p:.4f}"
        res_str += " -> Normal" if p > alpha else " -> NOT Normal"
        results.append(res_str)

    if method in ["all", "k2"]:
        try:
            stat, p = normaltest(data)
            res_str = f"da_k_squared: {title} : stat={stat:.2f} p={p:.4f}"
            res_str += " -> Normal" if p > alpha else " -> NOT Normal"
            results.append(res_str)
        except Exception as e:
            results.append(f"da_k_squared failed: {str(e)}")

    return "\n\n".join(results)


# -------------------------------------------------------------------
# 4. Layout & App
# -------------------------------------------------------------------
external_stylesheets = [dbc.themes.BOOTSTRAP]
app = dash.Dash(__name__, external_stylesheets=external_stylesheets)
server = app.server

navbar = dbc.Navbar(
    dbc.Container([
        html.A(
            dbc.Row([
                dbc.Col(
                    html.Img(src="https://upload.wikimedia.org/wikipedia/commons/6/60/Virginia_Tech_Hokies_logo.svg",
                             height="50px")),
                dbc.Col(dbc.NavbarBrand("CS 5764 Final Project", className="ms-2")),
            ], align="center", className="g-0"),
            href="#", style={"textDecoration": "none"},
        ),
    ]),
    color="primary", dark=True,
)

# --- Global Controls ---
controls = dbc.Card([
    dbc.CardHeader("Preprocessing & Controls"),
    dbc.CardBody([
        dbc.Row([
            dbc.Col([
                html.Label("1. Data Cleaning"),
                dcc.Dropdown(
                    id="clean-method",
                    options=[
                        {"label": "Keep Original", "value": "none"},
                        {"label": "Basic (Drop Negatives)", "value": "basic"},
                        {"label": "Strict (Drop All NaNs)", "value": "strict"},
                        {"label": "Fill Mean", "value": "fill_mean"}
                    ],
                    value="basic", clearable=False
                )
            ], width=4),
            dbc.Col([
                html.Label("2. Outlier Removal"),
                dcc.Dropdown(
                    id="outlier-method",
                    options=[
                        {"label": "None", "value": "none"},
                        {"label": "IQR Method", "value": "iqr"},
                        {"label": "Z-Score Method", "value": "zscore"}
                    ],
                    value="none", clearable=False
                )
            ], width=4),
            dbc.Col([
                html.Label("3. Transformation"),
                dcc.Dropdown(
                    id="transform-method",
                    options=[
                        {"label": "None", "value": "none"},
                        {"label": "Log1p", "value": "log1p"},
                        {"label": "MinMax Scaling", "value": "minmax"},
                        {"label": "Standard Scaling", "value": "standard"}
                    ],
                    value="none", clearable=False
                )
            ], width=4),
        ]),
        html.Hr(),
        dbc.Row([
            dbc.Col([
                html.Label("4. Filter by Flow Duration (Percentile)"),
                dcc.RangeSlider(
                    id='range-slider',
                    min=0, max=100, step=5,
                    value=[0, 100],
                    marks={0: '0%', 50: '50%', 100: '100%'},
                    tooltip={"placement": "bottom", "always_visible": True}
                )
            ], width=6),
            dbc.Col([
                html.Label("5. Visual Options"),
                dcc.Checklist(
                    id='graph-options',
                    options=[
                        {'label': ' Show Grid', 'value': 'grid'},
                        {'label': ' Show Legend', 'value': 'legend'}
                    ],
                    value=['grid', 'legend'],
                    inline=True,
                    inputStyle={"margin-right": "5px"}
                )
            ], width=6)
        ]),
        html.Hr(),
        dbc.Row([
            dbc.Col([
                html.Label("Export Data:"),
                html.Br(),
                dbc.Button("Download CSV", id="btn-download", color="success", size="sm"),
                dcc.Download(id="download-dataframe-csv"),
                dbc.Tooltip(
                    "Download the currently processed dataset as CSV",
                    target="btn-download",
                )
            ], width=12)
        ])
    ])
], className="mb-3")

# --- Tab 1: Data & Stats ---
tab1 = dbc.Container([
    html.H4("Data Overview & Statistics"),
    html.Div(id="data-shape-info", className="mb-3 text-primary fw-bold"),
    html.H5("Descriptive Statistics"),
    html.Div(id="stats-table-container")
], className="mt-3")

# --- Tab 2: PCA ---
tab2 = dbc.Container([
    dbc.Row([
        dbc.Col([
            dbc.Card([
                dbc.CardHeader("Dimensionality Reduction (PCA)"),
                dbc.CardBody(
                    dcc.Loading(
                        id="loading-pca",
                        type="default",
                        children=dcc.Graph(id="pca-graph")
                    )
                )
            ])
        ], width=12),
    ])
], className="mt-3")

# --- Tab 3: Normality Tests ---
tab3 = dbc.Container([
    dbc.Row([
        dbc.Col([
            dbc.Card([
                dbc.CardHeader("Normality Test Configuration"),
                dbc.CardBody([
                    html.Label("Select Feature to Test:"),
                    dcc.Dropdown(id="norm-feature", options=numeric_options, value="Flow Duration"),
                    html.Br(),
                    html.Label("Select Test Method:"),
                    dcc.Dropdown(
                        id="norm-method",
                        options=[
                            {'label': 'All Tests', 'value': 'all'},
                            {'label': 'Shapiro-Wilk', 'value': 'shapiro'},
                            {'label': 'Kolmogorov-Smirnov (K-S)', 'value': 'ks'},
                            {'label': "D'Agostino's K-squared", 'value': 'k2'}
                        ],
                        value='all', clearable=False
                    ),
                    html.Hr(),
                    html.Div(id="norm-result-text", style={"whiteSpace": "pre-wrap", "fontFamily": "monospace"})
                ])
            ])
        ], width=12)
    ])
], className="mt-3")

# --- Tab 4: Numeric Plots ---
tab4 = dbc.Container([
    dbc.Row([
        dbc.Col([
            dbc.Card([
                dbc.CardHeader("Numeric Plot Configuration"),
                dbc.CardBody([
                    html.Label("Feature 1 (X):"),
                    dcc.Dropdown(id="num-f1", options=numeric_options, value="Flow Duration"),
                    html.Label("Feature 2 (Y - Optional):"),
                    dcc.Dropdown(id="num-f2", options=numeric_options, value="Flow Bytes/s"),
                    html.Label("Feature 3 (Z - Optional):"),
                    dcc.Dropdown(id="num-f3", options=numeric_options, value="Average Packet Size"),
                    html.Hr(),
                    html.Label("Select Plot Type:"),
                    dcc.Dropdown(
                        id="num-type",
                        options=[
                            {"label": "1. Line Plot", "value": "line"},
                            {"label": "2. Dist Plot (Histogram)", "value": "dist"},
                            {"label": "3. Pair Plot", "value": "pair"},
                            {"label": "4. Heatmap (Correlation)", "value": "heatmap"},
                            {"label": "5. Histogram + KDE", "value": "hist_kde"},
                            {"label": "6. QQ Plot", "value": "qq"},
                            {"label": "7. KDE Plot (Custom)", "value": "kde_custom"},
                            {"label": "8. Lm/Reg Plot", "value": "reg"},
                            {"label": "9. Area Plot", "value": "area"},
                            {"label": "10. Joint Plot", "value": "joint"},
                            {"label": "11. Rug Plot", "value": "rug"},
                            {"label": "12. 3D Plot", "value": "3d"},
                            {"label": "13. Contour Plot", "value": "contour"},
                            {"label": "14. Cluster Map", "value": "cluster"},
                            {"label": "15. Hexbin Plot", "value": "hexbin"}
                        ],
                        value="line", clearable=False
                    ),
                    html.Hr(),
                    html.Label("KDE / Line Settings:", className="fw-bold"),
                    html.Label("Palette / Color:"),
                    dcc.Dropdown(
                        id="kde-palette",
                        options=[
                            {"label": "Blue", "value": "blue"},
                            {"label": "Red", "value": "red"},
                            {"label": "Green", "value": "green"},
                            {"label": "Purple", "value": "purple"}
                        ],
                        value="blue", clearable=False
                    ),
                    html.Label("Line Width:"),
                    dcc.Slider(id="kde-width", min=1, max=5, step=0.5, value=2,
                               marks={1: '1', 2: '2', 3: '3', 4: '4', 5: '5'})
                ])
            ])
        ], width=3),
        dbc.Col([
            dcc.Loading(
                id="loading-num",
                type="default",
                children=dcc.Graph(id="numeric-graph", style={"height": "700px"})
            )
        ], width=9)
    ])
], className="mt-3")

# --- Tab 5: Categorical Plots ---
tab5 = dbc.Container([
    dbc.Row([
        dbc.Col([
            dbc.Card([
                dbc.CardHeader("Categorical Config"),
                dbc.CardBody([
                    html.Label("Categorical Feature (X):"),
                    dcc.Dropdown(id="cat-col", options=cat_options, value="Label"),
                    html.Label("Numeric Feature (Y - Optional):"),
                    dcc.Dropdown(id="cat-num", options=numeric_options, value="Flow Bytes/s"),
                    html.Hr(),
                    html.Label("Select Plot Type:"),
                    dcc.Dropdown(
                        id="cat-type",
                        options=[
                            {"label": "1. Bar Plot", "value": "bar"},
                            {"label": "2. Count Plot", "value": "count"},
                            {"label": "3. Pie Chart", "value": "pie"},
                            {"label": "4. Multivariate Box", "value": "box_multi"},
                            {"label": "5. Multivariate Boxen", "value": "boxen_multi"},
                            {"label": "6. Violin Plot", "value": "violin"},
                            {"label": "7. Strip Plot", "value": "strip"},
                            {"label": "8. Swarm Plot", "value": "swarm"}
                        ],
                        value="bar", clearable=False
                    ),
                    html.Hr(),
                    html.Label("Bar Plot Settings:", className="fw-bold"),
                    dcc.RadioItems(
                        id="bar-mode",
                        options=[
                            {"label": "Grouped", "value": "group"},
                            {"label": "Stacked", "value": "stack"}
                        ],
                        value="group",
                        inline=True
                    )
                ])
            ])
        ], width=3),
        dbc.Col([
            dcc.Loading(
                id="loading-cat",
                type="default",
                children=dcc.Graph(id="cat-graph", style={"height": "600px"})
            )
        ], width=9)
    ])
], className="mt-3")

# --- Tab 6: Storytelling ---
tab6 = dbc.Container([
    html.H3("Network Traffic Storytelling Dashboard", className="text-center my-3"),

    dcc.Loading(
        id="loading-story",
        type="default",
        children=dcc.Graph(id="story-graph", style={"height": "800px"})
    ),

    html.Br(),
    html.Label("Notes:"),
    dcc.Textarea(
        id="story-notes",
        placeholder="Write down any observations or takeaways from the storytelling view here.",
        style={
            "width": "100%",
            "height": "150px",
            "fontFamily": "monospace"
        }
    )
], className="mt-3")

# --- Main Layout ---
app.layout = html.Div([
    navbar,
    dbc.Container([
        controls,
        dcc.Tabs(
            id="tabs",
            value="tab-1",
            children=[
                dcc.Tab(label="1. Stats", value="tab-1", children=tab1),
                dcc.Tab(label="2. PCA", value="tab-2", children=tab2),
                dcc.Tab(label="3. Normality", value="tab-3", children=tab3),
                dcc.Tab(label="4. Numeric", value="tab-4", children=tab4),
                dcc.Tab(label="5. Categorical", value="tab-5", children=tab5),
                dcc.Tab(label="6. Story", value="tab-6", children=tab6),
            ],
        ),
    ], fluid=True, className="mt-3"),
])


# -------------------------------------------------------------------
# 5. Callbacks
# -------------------------------------------------------------------

@app.callback(
    Output("download-dataframe-csv", "data"),
    Input("btn-download", "n_clicks"),
    State("clean-method", "value"),
    State("outlier-method", "value"),
    State("transform-method", "value"),
    State("range-slider", "value"),
    prevent_initial_call=True
)
def download_processed_data(n_clicks, clean, outlier, transform, range_val):
    if n_clicks is None:
        return dash.no_update
    df = get_processed_df(clean, outlier, transform, range_val)
    return dcc.send_data_frame(df.to_csv, "processed_data.csv")


@app.callback(
    Output("data-shape-info", "children"),
    Output("stats-table-container", "children"),
    Input("clean-method", "value"),
    Input("outlier-method", "value"),
    Input("transform-method", "value"),
    Input("range-slider", "value")
)
def update_stats(clean, outlier, transform, range_val):
    df = get_processed_df(clean, outlier, transform, range_val)
    msg = f"Raw Shape: {df_raw.shape}  ->  Processed Shape: {df.shape}"

    desc = df[NUMERIC_COLS].describe().T.reset_index()
    desc = desc.round(3)

    table = dash_table.DataTable(
        data=desc.to_dict('records'),
        columns=[{"name": i, "id": i} for i in desc.columns],
        style_table={'overflowX': 'auto'},
        page_size=10,
        style_header={'fontWeight': 'bold'}
    )
    return msg, table


@app.callback(
    Output("pca-graph", "figure"),
    Input("clean-method", "value"),
    Input("outlier-method", "value"),
    Input("transform-method", "value"),
    Input("range-slider", "value")
)
def update_pca(clean, outlier, transform, range_val):
    df = get_processed_df(clean, outlier, transform, range_val)
    if df.empty:
        return make_warning_figure("No Data after filtering")

    df_sub = df.sample(min(len(df), 1000), random_state=42)

    X = df_sub[NUMERIC_COLS].select_dtypes(include=[np.number]).fillna(0)

    n_samples, n_features = X.shape

    if n_samples < 2 or n_features < 2:
        return make_warning_figure("Not enough data for PCA")

    target_n = 5
    n_components = min(target_n, n_samples, n_features)

    X_std = StandardScaler().fit_transform(X)

    pca = PCA(n_components=n_components)
    components = pca.fit_transform(X_std)

    # print(f"PCA performed with n_components={n_components}")

    color_col = df_sub["Label"] if "Label" in df_sub.columns else None
    if color_col is not None and hasattr(color_col, 'cat'):
        color_col = color_col.astype(str)

    pca_fig = px.scatter(
        x=components[:, 0],
        y=components[:, 1],
        color=color_col,
        title=f"PCA Projection (Calculated {n_components} components, Showing PC1 vs PC2)",
        labels={'x': f'PC1 ({pca.explained_variance_ratio_[0]:.2%})',
                'y': f'PC2 ({pca.explained_variance_ratio_[1]:.2%})'}
    )

    pca_fig.update_layout(
        title_font_family="serif",
        title_font_color="blue",
        font_family="serif",
        font_size=14,
        showlegend=True
    )
    pca_fig.update_xaxes(title_font_color="darkred", showgrid=True)
    pca_fig.update_yaxes(title_font_color="darkred", showgrid=True)

    return pca_fig


@app.callback(
    Output("norm-result-text", "children"),
    Input("clean-method", "value"),
    Input("outlier-method", "value"),
    Input("transform-method", "value"),
    Input("range-slider", "value"),
    Input("norm-feature", "value"),
    Input("norm-method", "value")
)
def update_normality_test(clean, outlier, transform, range_val, feature, method):
    if not feature:
        return "Please select a feature."

    df = get_processed_df(clean, outlier, transform, range_val)
    data = df[feature].dropna()

    if len(data) < 3:
        return "Not enough data points to perform normality tests."

    result_string = perform_normality_tests(data, feature, method)
    return result_string


@app.callback(
    Output("numeric-graph", "figure"),
    Input("clean-method", "value"),
    Input("outlier-method", "value"),
    Input("transform-method", "value"),
    Input("range-slider", "value"),
    Input("num-f1", "value"),
    Input("num-f2", "value"),
    Input("num-f3", "value"),
    Input("num-type", "value"),
    Input("kde-palette", "value"),
    Input("kde-width", "value"),
    Input("graph-options", "value")
)
def update_numeric_plot(clean, outlier, transform, range_val, f1, f2, f3, plot_type, palette, width, options):
    df = get_processed_df(clean, outlier, transform, range_val)
    if df.empty: return make_warning_figure("No Data")

    df_sub = df.sample(min(len(df), 5000), random_state=42)

    if "Label" in df_sub.columns and hasattr(df_sub["Label"], 'cat'):
        df_sub["Label"] = df_sub["Label"].astype(str)

    if not f1: return make_warning_figure("Select Feature 1")

    fig = go.Figure()

    if plot_type == "line":
        df_sorted = df_sub.sort_index()
        fig = px.line(df_sorted, x=df_sorted.index, y=f1, title=f"Line Plot: {f1}")
        fig.update_layout(xaxis_title="Index")
        fig.update_traces(line=dict(width=width))

    elif plot_type == "dist":
        fig = px.histogram(df_sub, x=f1, title=f"Distribution Plot: {f1}")

    elif plot_type == "pair":
        cols = [c for c in [f1, f2, f3] if c]
        if len(cols) < 2: return make_warning_figure("Select at least 2 features")
        fig = px.scatter_matrix(df_sub, dimensions=cols, color="Label" if "Label" in df_sub.columns else None,
                                title="Pair Plot")

    elif plot_type == "heatmap":
        cols = [c for c in [f1, f2, f3] if c]
        if len(cols) < 2: return make_warning_figure("Select at least 2 features")
        fig = px.density_heatmap(df_sub, x=f1, y=f2, title=f"Heatmap: {f1} vs {f2}")

    elif plot_type == "hist_kde":
        fig = px.histogram(df_sub, x=f1, marginal="violin", title=f"Histogram with KDE: {f1}")

    elif plot_type == "qq":
        data = df_sub[f1].dropna()
        (osm, osr), _ = stats.probplot(data, dist="norm")
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=osm, y=osr, mode='markers', name='Data'))
        fig.add_trace(go.Scatter(x=[min(osm), max(osm)], y=[min(osm), max(osm)], mode='lines', line=dict(color='red'),
                                 name='Normal Line'))
        fig.update_layout(
            title=f"QQ Plot: {f1}",
            xaxis_title="Theoretical Quantiles",
            yaxis_title="Sample Quantiles"
        )

    elif plot_type == "kde_custom":
        data = df_sub[f1].dropna()
        if len(data) > 1:
            kde = stats.gaussian_kde(data)
            x_range = np.linspace(data.min(), data.max(), 200)
            y_val = kde(x_range)
            color_map = {"blue": "#1f77b4", "red": "#d62728", "green": "#2ca02c", "purple": "#9467bd"}
            line_color = color_map.get(palette, "blue")
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=x_range, y=y_val, mode='lines', fill='tozeroy',
                                     line=dict(color=line_color, width=width), opacity=0.6, name="KDE"))
            fig.update_layout(title=f"KDE Plot (Alpha=0.6, Width={width}): {f1}",
                              xaxis_title=f"{f1}",
                              yaxis_title="Density", )
        else:
            return make_warning_figure("Not enough data for KDE")

    elif plot_type == "reg":
        if not f2: return make_warning_figure("Select Feature 2")
        fig = px.scatter(df_sub, x=f1, y=f2, trendline="ols", title=f"Regression: {f1} vs {f2}",
                         trendline_color_override="red")

    elif plot_type == "area":
        if not f2: return make_warning_figure("Select Feature 2")
        df_sorted = df_sub.sort_values(by=f1)
        fig = px.area(df_sorted, x=f1, y=f2, title=f"Area Plot: {f1} vs {f2}")

    elif plot_type == "joint":
        if not f2: return make_warning_figure("Select Feature 2")
        fig = px.scatter(df_sub, x=f1, y=f2, marginal_x="violin", marginal_y="violin",
                         title="Joint Plot (Scatter + KDE-style Marginals)")

    elif plot_type == "rug":
        fig = px.strip(df_sub, x=f1, title=f"Rug Plot: {f1}")
        fig.update_layout(
            xaxis_title=f"{f1}",
            yaxis_title="Occurrences",
        )

    elif plot_type == "3d":
        if not f3: return make_warning_figure("Select Feature 3")
        fig = px.scatter_3d(df_sub, x=f1, y=f2, z=f3, color="Label" if "Label" in df_sub.columns else None,
                            title="3D Plot")

    elif plot_type == "contour":
        if not f2: return make_warning_figure("Select Feature 2")
        fig = px.density_contour(df_sub, x=f1, y=f2, title=f"Contour Plot: {f1} vs {f2}")

    elif plot_type == "cluster":
        cols = [c for c in NUMERIC_COLS if c in df.columns][:10]
        corr = df[cols].corr()
        fig = px.imshow(corr, title="Cluster Map")

    elif plot_type == "hexbin":
        if not f2: return make_warning_figure("Select Feature 2")
        fig = px.density_heatmap(df_sub, x=f1, y=f2, nbinsx=30, nbinsy=30, title=f"Hexbin Plot: {f1} vs {f2}")

    else:
        return make_warning_figure("Unknown Plot Type")

    show_grid = 'grid' in options
    show_legend = 'legend' in options
    fig.update_layout(
        title_font_family="serif",
        title_font_color="blue",
        font_family="serif",
        showlegend=show_legend,
        font_size=14,
    )
    fig.update_xaxes(title_font_color="darkred", showgrid=show_grid)
    fig.update_yaxes(title_font_color="darkred", showgrid=show_grid)

    return fig


@app.callback(
    Output("cat-graph", "figure"),
    Input("clean-method", "value"),
    Input("outlier-method", "value"),
    Input("transform-method", "value"),
    Input("range-slider", "value"),
    Input("cat-col", "value"),
    Input("cat-num", "value"),
    Input("cat-type", "value"),
    Input("bar-mode", "value"),
    Input("graph-options", "value")
)
def update_categorical_plot(clean, outlier, transform, range_val, cat, num, plot_type, bar_mode, options):
    df = get_processed_df(clean, outlier, transform, range_val)
    if df.empty: return make_warning_figure("No Data")

    df_sub = df.sample(min(len(df), 5000), random_state=42)

    if "Label" in df_sub.columns and hasattr(df_sub["Label"], 'cat'):
        df_sub["Label"] = df_sub["Label"].astype(str)

    if not cat: return make_warning_figure("Select Categorical Feature")

    fig = go.Figure()

    if plot_type == "bar":
        if not num: return make_warning_figure("Select Numeric Feature")
        group_cols = [cat]
        if "Label" in df_sub.columns and cat != "Label":
            group_cols.append("Label")
        df_agg = df_sub.groupby(group_cols, observed=False)[num].mean().reset_index()
        color_col = "Label" if "Label" in df_sub.columns else None
        fig = px.bar(df_agg, x=cat, y=num, color=color_col, barmode=bar_mode, title=f"Bar Plot ({bar_mode})")

    elif plot_type == "count":
        fig = px.histogram(df_sub, x=cat, color="Label" if "Label" in df_sub.columns else None, barmode=bar_mode,
                           title=f"Count Plot: {cat}")

    elif plot_type == "pie":
        fig = px.pie(df_sub, names=cat, title=f"Pie Chart: {cat}")

    elif plot_type == "box_multi":
        if not num: return make_warning_figure("Select Numeric Feature")
        fig = px.box(df_sub, x=cat, y=num, color="Label" if "Label" in df_sub.columns else None,
                     title=f"Box Plot: {num} by {cat}")

    elif plot_type == "boxen_multi":
        if not num: return make_warning_figure("Select Numeric Feature")
        fig = px.box(df_sub, x=cat, y=num, color="Label" if "Label" in df_sub.columns else None, points="all",
                     title=f"Boxen Plot")

    elif plot_type == "violin":
        if not num: return make_warning_figure("Select Numeric Feature")
        fig = px.violin(df_sub, x=cat, y=num, color="Label" if "Label" in df_sub.columns else None, box=True,
                        title=f"Violin Plot")

    elif plot_type == "strip":
        if not num: return make_warning_figure("Select Numeric Feature")
        fig = px.strip(df_sub, x=cat, y=num, color="Label" if "Label" in df_sub.columns else None, title=f"Strip Plot")

    elif plot_type == "swarm":
        if not num: return make_warning_figure("Select Numeric Feature")
        fig = px.strip(df_sub, x=cat, y=num, color="Label" if "Label" in df_sub.columns else None, stripmode='overlay',
                       title=f"Swarm Plot")

    else:
        return make_warning_figure("Unknown Plot Type")

    show_grid = 'grid' in options
    show_legend = 'legend' in options
    fig.update_layout(
        title_font_family="serif",
        title_font_color="blue",
        font_family="serif",
        showlegend=show_legend,
        font_size=14,
    )
    fig.update_xaxes(title_font_color="darkred", showgrid=show_grid)
    fig.update_yaxes(title_font_color="darkred", showgrid=show_grid)

    return fig


@app.callback(
    Output("story-graph", "figure"),
    Input("clean-method", "value"),
    Input("outlier-method", "value"),
    Input("transform-method", "value"),
    Input("range-slider", "value")
)
def update_storytelling(clean, outlier, transform, range_val):
    df = get_processed_df(clean, outlier, transform, range_val)

    if df.empty:
        return make_warning_figure("No data available with current settings.")

    df_sub = df.sample(min(len(df), 3000), random_state=42)

    if "Label" in df_sub.columns and hasattr(df_sub["Label"], 'cat'):
        df_sub["Label"] = df_sub["Label"].astype(str)

    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=("Attack Distribution", "Flow Duration by Label",
                        "Traffic Volume (Bytes vs Packets)", "Average Packet Size"),
        specs=[[{"type": "domain"}, {"type": "xy"}],
               [{"type": "xy"}, {"type": "xy"}]]
    )

    if "Label" in df_sub.columns:
        counts = df_sub["Label"].value_counts().reset_index()
        counts.columns = ["Label", "count"]
        fig.add_trace(go.Pie(labels=counts["Label"], values=counts["count"], name="Attacks"), row=1, col=1)

        for label in df_sub["Label"].unique():
            subset = df_sub[df_sub["Label"] == label]
            fig.add_trace(go.Box(y=subset["Flow Duration"], name=str(label), showlegend=False), row=1, col=2)

        avg_size = df_sub.groupby("Label", observed=False)["Average Packet Size"].mean().reset_index()
        fig.add_trace(go.Bar(x=avg_size["Label"], y=avg_size["Average Packet Size"], name="Avg Size", showlegend=False),
                      row=2, col=2)
    else:
        fig.add_annotation(text="No Label Column", row=1, col=1)

    fig.add_trace(go.Scatter(x=df_sub["Flow Packets/s"], y=df_sub["Flow Bytes/s"],
                             mode='markers', marker=dict(size=5, opacity=0.5),
                             name="Traffic", showlegend=False), row=2, col=1)

    fig.update_layout(height=800, title_text="Comprehensive Network Traffic Analysis")
    return fig


if __name__ == "__main__":
    app.run(debug=False, host="0.0.0.0", port=8080)