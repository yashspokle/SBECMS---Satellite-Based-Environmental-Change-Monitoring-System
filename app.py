import io
import math
from typing import List, Dict, Tuple

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import streamlit as st

# Optional map support
try:
    import pydeck as pdk
    PYDECK_AVAILABLE = True
except Exception:
    PYDECK_AVAILABLE = False


# =========================================================
# PAGE SETUP
# =========================================================
st.set_page_config(
    page_title="Environmental Change Dashboard",
    layout="wide",
    initial_sidebar_state="expanded",
)

# =========================================================
# COLORS / STYLE
# =========================================================
BG = "#0d1117"
PANEL = "#161b22"
BORDER = "#30363d"
TEXT = "#e6edf3"
MUTED = "#8b949e"

ACCENT_1 = "#58a6ff"
ACCENT_2 = "#3fb950"
ACCENT_3 = "#f78166"
ACCENT_4 = "#a371f7"
ACCENT_5 = "#ffd700"

LAND_COLORS = {
    "Forest": ACCENT_2,
    "Agriculture": ACCENT_5,
    "Urban": ACCENT_3,
    "Water": ACCENT_1,
    "Barren": "#c9d1d9",
}

EVENT_COLORS = {
    "None": MUTED,
    "Flood": ACCENT_1,
    "Wildfire": ACCENT_3,
    "Drought": ACCENT_5,
}

st.markdown(
    f"""
    <style>
        .stApp {{
            background-color: {BG};
            color: {TEXT};
        }}

        .block-container {{
            max-width: 1450px;
            padding-top: 1.2rem;
            padding-bottom: 2rem;
        }}

        section[data-testid="stSidebar"] {{
            background-color: {PANEL};
            border-right: 1px solid {BORDER};
        }}

        h1, h2, h3, h4, h5, h6, p, div, span, label {{
            color: {TEXT};
        }}

        .top-card {{
            background: linear-gradient(135deg, {PANEL} 0%, #11161d 100%);
            border: 1px solid {BORDER};
            border-radius: 16px;
            padding: 22px 24px;
            margin-bottom: 16px;
        }}

        .info-card {{
            background-color: {PANEL};
            border: 1px solid {BORDER};
            border-radius: 14px;
            padding: 16px 18px;
            height: 100%;
        }}

        .metric-card {{
            background-color: {PANEL};
            border: 1px solid {BORDER};
            border-radius: 14px;
            padding: 16px 18px;
            min-height: 112px;
        }}

        .metric-label {{
            color: {MUTED};
            font-size: 0.92rem;
            margin-bottom: 6px;
        }}

        .metric-value {{
            color: {TEXT};
            font-size: 1.7rem;
            font-weight: 700;
            line-height: 1.2;
        }}

        .metric-note {{
            color: {MUTED};
            font-size: 0.8rem;
            margin-top: 6px;
        }}

        .section-card {{
            background-color: {PANEL};
            border: 1px solid {BORDER};
            border-radius: 14px;
            padding: 16px 18px;
            margin-bottom: 16px;
        }}

        .small-note {{
            color: {MUTED};
            font-size: 0.88rem;
        }}

        .finding {{
            background-color: #11161d;
            border-left: 4px solid {ACCENT_1};
            padding: 12px 14px;
            margin-bottom: 10px;
            border-radius: 10px;
            color: {TEXT};
        }}

        .stDownloadButton button,
        .stButton button {{
            background-color: {PANEL};
            color: {TEXT};
            border: 1px solid {BORDER};
            border-radius: 10px;
        }}

        .stDataFrame {{
            border: 1px solid {BORDER};
            border-radius: 12px;
        }}
    </style>
    """,
    unsafe_allow_html=True,
)


# =========================================================
# HELPERS
# =========================================================
REQUIRED_COLUMNS = [
    "land_use",
    "event",
    "ndvi",
    "ndwi",
    "lst_celsius",
    "change_index",
]

OPTIONAL_COLUMNS = [
    "region",
    "date",
    "latitude",
    "longitude",
]

DISPLAY_NAMES = {
    "ndvi": "Green Cover Score",
    "ndwi": "Water Presence Score",
    "lst_celsius": "Surface Heat",
    "change_index": "Change Score",
    "land_use": "Area Type",
    "event": "Situation",
    "region": "Place",
    "date": "Date",
    "latitude": "Latitude",
    "longitude": "Longitude",
}

def nice_label(col: str) -> str:
    return DISPLAY_NAMES.get(col, col.replace("_", " ").title())


def style_axis(ax, title: str, xlabel: str, ylabel: str, title_color: str = ACCENT_1):
    ax.set_facecolor(PANEL)
    ax.set_title(title, color=title_color, fontsize=12, fontweight="bold", pad=10)
    ax.set_xlabel(xlabel, color=TEXT, fontsize=10)
    ax.set_ylabel(ylabel, color=TEXT, fontsize=10)
    ax.tick_params(colors=TEXT, labelsize=9)
    for spine in ax.spines.values():
        spine.set_edgecolor(BORDER)
    ax.grid(True, linestyle="--", alpha=0.12, color=MUTED)


@st.cache_data
def generate_demo_data(n: int = 900) -> pd.DataFrame:
    np.random.seed(42)

    land_use = np.random.choice(
        ["Forest", "Agriculture", "Urban", "Water", "Barren"],
        size=n,
        p=[0.24, 0.25, 0.22, 0.14, 0.15],
    )

    event = np.random.choice(
        ["None", "Flood", "Wildfire", "Drought"],
        size=n,
        p=[0.48, 0.18, 0.16, 0.18],
    )

    region = np.random.choice(
        ["North Zone", "South Zone", "East Zone", "West Zone", "Central Zone"],
        size=n,
    )

    base_green = np.random.normal(0.45, 0.18, n)
    base_water = np.random.normal(0.08, 0.16, n)
    base_heat = np.random.normal(31, 5, n)
    base_change = np.abs(np.random.normal(0.25, 0.12, n))

    land_adjust_green = pd.Series(land_use).map({
        "Forest": 0.22,
        "Agriculture": 0.08,
        "Urban": -0.18,
        "Water": -0.10,
        "Barren": -0.22,
    }).to_numpy()

    land_adjust_water = pd.Series(land_use).map({
        "Forest": 0.03,
        "Agriculture": -0.03,
        "Urban": -0.07,
        "Water": 0.35,
        "Barren": -0.08,
    }).to_numpy()

    land_adjust_heat = pd.Series(land_use).map({
        "Forest": -4.5,
        "Agriculture": -1.0,
        "Urban": 5.0,
        "Water": -3.5,
        "Barren": 3.5,
    }).to_numpy()

    event_adjust_green = pd.Series(event).map({
        "None": 0.00,
        "Flood": -0.05,
        "Wildfire": -0.18,
        "Drought": -0.15,
    }).to_numpy()

    event_adjust_water = pd.Series(event).map({
        "None": 0.00,
        "Flood": 0.22,
        "Wildfire": -0.06,
        "Drought": -0.12,
    }).to_numpy()

    event_adjust_heat = pd.Series(event).map({
        "None": 0.00,
        "Flood": -1.2,
        "Wildfire": 5.2,
        "Drought": 3.4,
    }).to_numpy()

    event_adjust_change = pd.Series(event).map({
        "None": 0.02,
        "Flood": 0.20,
        "Wildfire": 0.28,
        "Drought": 0.18,
    }).to_numpy()

    ndvi = np.clip(base_green + land_adjust_green + event_adjust_green, -0.2, 0.95)
    ndwi = np.clip(base_water + land_adjust_water + event_adjust_water, -0.6, 0.9)
    lst_celsius = np.clip(base_heat + land_adjust_heat + event_adjust_heat, 10, 50)
    change_index = np.clip(base_change + event_adjust_change, 0.01, 1.5)

    start_date = pd.Timestamp("2024-01-01")
    dates = start_date + pd.to_timedelta(np.random.randint(0, 365, n), unit="D")

    lat = np.random.uniform(12.0, 28.0, size=n)
    lon = np.random.uniform(72.0, 89.0, size=n)

    df = pd.DataFrame({
        "region": region,
        "date": dates,
        "land_use": land_use,
        "event": event,
        "ndvi": ndvi,
        "ndwi": ndwi,
        "lst_celsius": lst_celsius,
        "change_index": change_index,
        "latitude": lat,
        "longitude": lon,
    })

    return df


def clean_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # standardize column names
    df.columns = [c.strip().lower() for c in df.columns]

    # type conversions
    for col in ["ndvi", "ndwi", "lst_celsius", "change_index", "latitude", "longitude"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"], errors="coerce")

    for col in ["land_use", "event", "region"]:
        if col in df.columns:
            df[col] = df[col].astype(str).str.strip()
            df.loc[df[col].isin(["nan", "None", ""]), col] = np.nan

    # fill missing labels
    if "land_use" in df.columns:
        df["land_use"] = df["land_use"].fillna("Unknown")
    if "event" in df.columns:
        df["event"] = df["event"].fillna("None")
    if "region" in df.columns:
        df["region"] = df["region"].fillna("Unknown Area")

    # remove rows missing core measures
    keep_cols = [c for c in REQUIRED_COLUMNS if c in df.columns]
    df = df.dropna(subset=keep_cols)

    # clip obvious outliers
    if "ndvi" in df.columns:
        df["ndvi"] = df["ndvi"].clip(-1, 1)
    if "ndwi" in df.columns:
        df["ndwi"] = df["ndwi"].clip(-1, 1)
    if "lst_celsius" in df.columns:
        df["lst_celsius"] = df["lst_celsius"].clip(-20, 80)
    if "change_index" in df.columns:
        df["change_index"] = df["change_index"].clip(0, 10)

    # friendly categories
    if "ndvi" in df.columns:
        df["green_level"] = pd.cut(
            df["ndvi"],
            bins=[-1, 0.15, 0.35, 0.6, 1],
            labels=["Very Low", "Low", "Moderate", "High"],
            include_lowest=True,
        )

    if "lst_celsius" in df.columns:
        df["heat_level"] = pd.cut(
            df["lst_celsius"],
            bins=[-50, 22, 30, 37, 100],
            labels=["Cool", "Mild", "Warm", "Hot"],
            include_lowest=True,
        )

    return df


def validate_columns(df: pd.DataFrame) -> Tuple[bool, List[str]]:
    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    return len(missing) == 0, missing


def load_data(uploaded_file) -> pd.DataFrame:
    if uploaded_file is None:
        return generate_demo_data()

    file_name = uploaded_file.name.lower()

    if file_name.endswith(".csv"):
        df = pd.read_csv(uploaded_file)
    elif file_name.endswith(".xlsx"):
        df = pd.read_excel(uploaded_file)
    else:
        raise ValueError("Only CSV and Excel files are supported.")

    return df


def build_summary(df: pd.DataFrame) -> Dict[str, float]:
    return {
        "rows": len(df),
        "avg_green": df["ndvi"].mean(),
        "avg_water": df["ndwi"].mean(),
        "avg_heat": df["lst_celsius"].mean(),
        "avg_change": df["change_index"].mean(),
    }


def generate_findings(df: pd.DataFrame) -> List[str]:
    findings = []

    avg_green = df["ndvi"].mean()
    avg_heat = df["lst_celsius"].mean()
    avg_change = df["change_index"].mean()

    if avg_green < 0.25:
        findings.append("Overall green cover is low across the selected records.")

    if avg_heat > 34:
        findings.append("Surface heat is high in the selected view.")

    if avg_change > 0.45:
        findings.append("The overall change level is strong, which may indicate stressed or disturbed areas.")

    if "event" in df.columns and "Flood" in df["event"].unique():
        flood_df = df[df["event"] == "Flood"]
        if not flood_df.empty and flood_df["ndwi"].mean() > df["ndwi"].mean():
            findings.append("Flood-related records show stronger water presence than the overall average.")

    if "event" in df.columns and "Wildfire" in df["event"].unique():
        fire_df = df[df["event"] == "Wildfire"]
        if not fire_df.empty and fire_df["lst_celsius"].mean() > df["lst_celsius"].mean():
            findings.append("Wildfire-related records are linked with higher surface heat.")

    if "land_use" in df.columns and "Urban" in df["land_use"].unique():
        urban_df = df[df["land_use"] == "Urban"]
        if not urban_df.empty and urban_df["lst_celsius"].mean() > df["lst_celsius"].mean():
            findings.append("Built-up areas are hotter than the overall average.")

    if "land_use" in df.columns and "Forest" in df["land_use"].unique():
        forest_df = df[df["land_use"] == "Forest"]
        if not forest_df.empty and forest_df["ndvi"].mean() > df["ndvi"].mean():
            findings.append("Forest areas show stronger green cover than the overall average.")

    if "region" in df.columns:
        worst_region = (
            df.groupby("region")["change_index"]
            .mean()
            .sort_values(ascending=False)
            .head(1)
        )
        if not worst_region.empty:
            findings.append(
                f"The highest average change is observed in {worst_region.index[0]}."
            )

    if not findings:
        findings.append("The selected data looks stable with no standout pattern in the current view.")

    return findings


def region_summary(df: pd.DataFrame) -> pd.DataFrame:
    if "region" not in df.columns:
        return pd.DataFrame()

    summary = (
        df.groupby("region", dropna=False)
        .agg(
            records=("ndvi", "size"),
            green_cover=("ndvi", "mean"),
            water_presence=("ndwi", "mean"),
            surface_heat=("lst_celsius", "mean"),
            change_score=("change_index", "mean"),
        )
        .reset_index()
        .sort_values("change_score", ascending=False)
    )
    return summary


def event_summary(df: pd.DataFrame) -> pd.DataFrame:
    summary = (
        df.groupby("event", dropna=False)
        .agg(
            records=("ndvi", "size"),
            green_cover=("ndvi", "mean"),
            water_presence=("ndwi", "mean"),
            surface_heat=("lst_celsius", "mean"),
            change_score=("change_index", "mean"),
        )
        .reset_index()
        .sort_values("records", ascending=False)
    )
    return summary


def downloadable_csv(df: pd.DataFrame) -> bytes:
    return df.to_csv(index=False).encode("utf-8")


# =========================================================
# CHARTS
# =========================================================
def plot_main_relationships(df: pd.DataFrame):
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    fig.patch.set_facecolor(BG)
    fig.suptitle(
        "Area Comparison Overview",
        fontsize=14,
        color=ACCENT_1,
        fontweight="bold",
        y=0.98
    )

    # 1
    ax = axes[0]
    for area, grp in df.groupby("land_use"):
        ax.scatter(
            grp["ndvi"],
            grp["lst_celsius"],
            s=24,
            alpha=0.58,
            label=area,
            color=LAND_COLORS.get(area, "#c9d1d9"),
        )
    style_axis(ax, "Green Cover vs Surface Heat", "Green Cover Score", "Surface Heat")
    leg = ax.legend(fontsize=8, framealpha=0.2)
    if leg:
        leg.get_frame().set_facecolor(PANEL)
        leg.get_frame().set_edgecolor(BORDER)
        for t in leg.get_texts():
            t.set_color(TEXT)

    # 2
    ax = axes[1]
    for situation, grp in df.groupby("event"):
        ax.scatter(
            grp["ndvi"],
            grp["ndwi"],
            s=24,
            alpha=0.60,
            label=situation,
            color=EVENT_COLORS.get(situation, "#c9d1d9"),
        )
    style_axis(ax, "Green Cover vs Water Presence", "Green Cover Score", "Water Presence Score", ACCENT_2)
    leg = ax.legend(fontsize=8, framealpha=0.2)
    if leg:
        leg.get_frame().set_facecolor(PANEL)
        leg.get_frame().set_edgecolor(BORDER)
        for t in leg.get_texts():
            t.set_color(TEXT)

    # 3
    ax = axes[2]
    order = [e for e in ["None", "Flood", "Wildfire", "Drought"] if e in df["event"].unique()]
    positions = np.arange(len(order))
    data = [df[df["event"] == e]["change_index"].values for e in order]

    box = ax.boxplot(
        data,
        labels=order,
        patch_artist=True,
        widths=0.55,
        medianprops=dict(color="white", linewidth=1.5),
        whiskerprops=dict(color="#c9d1d9"),
        capprops=dict(color="#c9d1d9"),
    )

    for patch, e in zip(box["boxes"], order):
        patch.set_facecolor(EVENT_COLORS.get(e, "#c9d1d9"))
        patch.set_alpha(0.70)
        patch.set_edgecolor(BORDER)

    style_axis(ax, "Change Score by Situation", "Situation", "Change Score", ACCENT_3)

    plt.tight_layout()
    st.pyplot(fig, use_container_width=True)


def plot_level_breakdown(df: pd.DataFrame):
    c1, c2 = st.columns(2)

    with c1:
        fig, ax = plt.subplots(figsize=(7, 4.2))
        fig.patch.set_facecolor(BG)

        green_counts = df["green_level"].value_counts(dropna=False).reindex(
            ["Very Low", "Low", "Moderate", "High"]
        )

        ax.bar(
            green_counts.index.astype(str),
            green_counts.values,
            color=[ACCENT_3, ACCENT_5, ACCENT_1, ACCENT_2]
        )
        style_axis(ax, "Green Cover Level", "Group", "Count", ACCENT_2)
        st.pyplot(fig, use_container_width=True)

    with c2:
        fig, ax = plt.subplots(figsize=(7, 4.2))
        fig.patch.set_facecolor(BG)

        heat_counts = df["heat_level"].value_counts(dropna=False).reindex(
            ["Cool", "Mild", "Warm", "Hot"]
        )

        ax.bar(
            heat_counts.index.astype(str),
            heat_counts.values,
            color=[ACCENT_1, ACCENT_2, ACCENT_5, ACCENT_3]
        )
        style_axis(ax, "Surface Heat Level", "Group", "Count", ACCENT_3)
        st.pyplot(fig, use_container_width=True)


def plot_region_comparison(df: pd.DataFrame):
    if "region" not in df.columns:
        st.info("Place-wise comparison is not available because the file does not include a place column.")
        return

    summary = region_summary(df)
    if summary.empty:
        st.info("Place-wise comparison is not available for the current selection.")
        return

    top_n = min(10, len(summary))
    summary = summary.head(top_n)

    fig, ax = plt.subplots(figsize=(10, 5))
    fig.patch.set_facecolor(BG)

    bars = ax.barh(
        summary["region"],
        summary["change_score"],
        color=ACCENT_1,
        alpha=0.75,
    )
    ax.invert_yaxis()
    style_axis(ax, "Places with Highest Change", "Change Score", "Place", ACCENT_1)

    for bar, value in zip(bars, summary["change_score"]):
        ax.text(
            bar.get_width() + 0.01,
            bar.get_y() + bar.get_height() / 2,
            f"{value:.2f}",
            va="center",
            ha="left",
            color=TEXT,
            fontsize=9,
        )

    st.pyplot(fig, use_container_width=True)


def plot_timeline(df: pd.DataFrame):
    if "date" not in df.columns or df["date"].isna().all():
        st.info("Time view is not available because the file does not include usable dates.")
        return

    temp = df.dropna(subset=["date"]).copy()
    if temp.empty:
        st.info("Time view is not available for the current selection.")
        return

    temp["month"] = temp["date"].dt.to_period("M").astype(str)
    monthly = (
        temp.groupby("month")
        .agg(
            green_cover=("ndvi", "mean"),
            water_presence=("ndwi", "mean"),
            surface_heat=("lst_celsius", "mean"),
            change_score=("change_index", "mean"),
        )
        .reset_index()
    )

    fig, ax = plt.subplots(figsize=(11, 5))
    fig.patch.set_facecolor(BG)

    ax.plot(monthly["month"], monthly["green_cover"], marker="o", linewidth=2, label="Green Cover Score")
    ax.plot(monthly["month"], monthly["water_presence"], marker="o", linewidth=2, label="Water Presence Score")
    ax.plot(monthly["month"], monthly["change_score"], marker="o", linewidth=2, label="Change Score")

    style_axis(ax, "Monthly Change View", "Month", "Average Value", ACCENT_4)
    ax.tick_params(axis="x", rotation=45)

    leg = ax.legend(fontsize=8, framealpha=0.2)
    if leg:
        leg.get_frame().set_facecolor(PANEL)
        leg.get_frame().set_edgecolor(BORDER)
        for t in leg.get_texts():
            t.set_color(TEXT)

    st.pyplot(fig, use_container_width=True)

    fig2, ax2 = plt.subplots(figsize=(11, 4.2))
    fig2.patch.set_facecolor(BG)

    ax2.plot(monthly["month"], monthly["surface_heat"], marker="o", linewidth=2)
    style_axis(ax2, "Monthly Heat View", "Month", "Surface Heat", ACCENT_3)
    ax2.tick_params(axis="x", rotation=45)

    st.pyplot(fig2, use_container_width=True)


def show_map(df: pd.DataFrame):
    if not PYDECK_AVAILABLE:
        st.info("Map view is unavailable because the map library is not installed.")
        return

    if "latitude" not in df.columns or "longitude" not in df.columns:
        st.info("Map view is not available because the file does not include latitude and longitude.")
        return

    temp = df.dropna(subset=["latitude", "longitude"]).copy()
    if temp.empty:
        st.info("Map view is not available for the current selection.")
        return

    temp = temp.head(2000).copy()

    def point_color(row):
        ev = row.get("event", "None")
        if ev == "Flood":
            return [88, 166, 255, 180]
        if ev == "Wildfire":
            return [247, 129, 102, 180]
        if ev == "Drought":
            return [255, 215, 0, 180]
        return [139, 148, 158, 150]

    temp["color"] = temp.apply(point_color, axis=1)
    temp["size"] = (temp["change_index"].fillna(0.2) * 18000).clip(3000, 40000)

    view_state = pdk.ViewState(
        latitude=float(temp["latitude"].mean()),
        longitude=float(temp["longitude"].mean()),
        zoom=4.2,
        pitch=30,
    )

    layer = pdk.Layer(
        "ScatterplotLayer",
        data=temp,
        get_position="[longitude, latitude]",
        get_fill_color="color",
        get_radius="size",
        pickable=True,
        opacity=0.75,
    )

    tooltip = {
        "html": """
        <b>Place:</b> {region}<br/>
        <b>Area Type:</b> {land_use}<br/>
        <b>Situation:</b> {event}<br/>
        <b>Green Cover:</b> {ndvi}<br/>
        <b>Water Presence:</b> {ndwi}<br/>
        <b>Heat:</b> {lst_celsius}<br/>
        <b>Change:</b> {change_index}
        """,
        "style": {
            "backgroundColor": "#11161d",
            "color": "white"
        },
    }

    st.pydeck_chart(
        pdk.Deck(
            layers=[layer],
            initial_view_state=view_state,
            tooltip=tooltip,
            map_style="mapbox://styles/mapbox/dark-v10",
        ),
        use_container_width=True,
    )


# =========================================================
# UI: HEADER
# =========================================================
st.markdown(
    """
    <div class="top-card">
        <h1 style="margin:0; font-size:2rem;">Environmental Change Dashboard</h1>
        <div class="small-note" style="margin-top:8px;">
            A project-style dashboard for studying green cover, water presence, surface heat, and overall change across different places and situations.
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

# =========================================================
# SIDEBAR
# =========================================================
with st.sidebar:
    st.markdown("## Data")
    uploaded_file = st.file_uploader("Upload CSV or Excel file", type=["csv", "xlsx"])

    use_demo = st.checkbox("Use demo data", value=(uploaded_file is None))

    st.markdown("---")
    st.markdown("## About this app")
    st.markdown(
        """
        This dashboard is designed to look like a project, not a classroom chart sheet.

        The screen uses simple labels such as:
        - Green Cover Score
        - Water Presence Score
        - Surface Heat
        - Change Score
        """
    )

# =========================================================
# LOAD / VALIDATE
# =========================================================
try:
    if use_demo and uploaded_file is None:
        raw_df = generate_demo_data()
    else:
        raw_df = load_data(uploaded_file)
except Exception as e:
    st.error(f"Could not load the file: {e}")
    st.stop()

df = clean_dataframe(raw_df)
valid, missing_cols = validate_columns(df)

if not valid:
    st.error(
        "The file is missing required columns: "
        + ", ".join(missing_cols)
    )
    st.markdown(
        """
        Required columns:
        - land_use
        - event
        - ndvi
        - ndwi
        - lst_celsius
        - change_index
        """
    )
    st.stop()

# =========================================================
# SIDEBAR FILTERS
# =========================================================
with st.sidebar:
    st.markdown("## Filters")

    land_options = sorted(df["land_use"].dropna().unique().tolist())
    event_options = sorted(df["event"].dropna().unique().tolist())

    selected_land = st.multiselect("Area Type", land_options, default=land_options)
    selected_event = st.multiselect("Situation", event_options, default=event_options)

    ndvi_min, ndvi_max = float(df["ndvi"].min()), float(df["ndvi"].max())
    ndwi_min, ndwi_max = float(df["ndwi"].min()), float(df["ndwi"].max())
    heat_min, heat_max = float(df["lst_celsius"].min()), float(df["lst_celsius"].max())
    change_min, change_max = float(df["change_index"].min()), float(df["change_index"].max())

    selected_green = st.slider("Green Cover Score", ndvi_min, ndvi_max, (ndvi_min, ndvi_max))
    selected_water = st.slider("Water Presence Score", ndwi_min, ndwi_max, (ndwi_min, ndwi_max))
    selected_heat = st.slider("Surface Heat", heat_min, heat_max, (heat_min, heat_max))
    selected_change = st.slider("Change Score", change_min, change_max, (change_min, change_max))

    selected_regions = None
    if "region" in df.columns:
        region_options = sorted(df["region"].dropna().unique().tolist())
        selected_regions = st.multiselect("Place", region_options, default=region_options)

filtered = df[
    df["land_use"].isin(selected_land)
    & df["event"].isin(selected_event)
    & df["ndvi"].between(selected_green[0], selected_green[1])
    & df["ndwi"].between(selected_water[0], selected_water[1])
    & df["lst_celsius"].between(selected_heat[0], selected_heat[1])
    & df["change_index"].between(selected_change[0], selected_change[1])
].copy()

if selected_regions is not None:
    filtered = filtered[filtered["region"].isin(selected_regions)].copy()

if filtered.empty:
    st.warning("No records match the selected filters.")
    st.stop()

# =========================================================
# TOP METRICS
# =========================================================
summary = build_summary(filtered)

m1, m2, m3, m4, m5 = st.columns(5)

metric_values = [
    ("Records", f"{summary['rows']:,}", "Rows currently visible"),
    ("Green Cover", f"{summary['avg_green']:.3f}", "Average score"),
    ("Water Presence", f"{summary['avg_water']:.3f}", "Average score"),
    ("Surface Heat", f"{summary['avg_heat']:.2f}", "Average temperature"),
    ("Change Score", f"{summary['avg_change']:.3f}", "Average level"),
]

for col, (label, value, note) in zip([m1, m2, m3, m4, m5], metric_values):
    with col:
        st.markdown(
            f"""
            <div class="metric-card">
                <div class="metric-label">{label}</div>
                <div class="metric-value">{value}</div>
                <div class="metric-note">{note}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

st.markdown("")

# =========================================================
# INFO STRIP
# =========================================================
c1, c2, c3 = st.columns(3)

with c1:
    st.markdown(
        f"""
        <div class="info-card">
            <h4 style="margin-top:0;">Data Health</h4>
            <div class="small-note">
                Loaded rows: {len(raw_df):,}<br/>
                Clean rows used: {len(filtered):,}<br/>
                Available columns: {len(df.columns)}
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

with c2:
    region_text = "Available" if "region" in df.columns else "Not available"
    date_text = "Available" if "date" in df.columns and not df["date"].isna().all() else "Not available"
    st.markdown(
        f"""
        <div class="info-card">
            <h4 style="margin-top:0;">Extra Views</h4>
            <div class="small-note">
                Place view: {region_text}<br/>
                Time view: {date_text}<br/>
                Map view: {"Available" if ("latitude" in df.columns and "longitude" in df.columns) else "Not available"}
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

with c3:
    st.markdown(
        f"""
        <div class="info-card">
            <h4 style="margin-top:0;">Current Selection</h4>
            <div class="small-note">
                Area types selected: {len(selected_land)}<br/>
                Situations selected: {len(selected_event)}<br/>
                Filtered records shown: {len(filtered):,}
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

# =========================================================
# TABS
# =========================================================
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "Overview",
    "Place Comparison",
    "Time View",
    "Map View",
    "Data Table",
])

with tab1:
    st.markdown('<div class="section-card">', unsafe_allow_html=True)
    st.subheader("Main Comparison")
    plot_main_relationships(filtered)
    st.markdown("</div>", unsafe_allow_html=True)

    st.markdown('<div class="section-card">', unsafe_allow_html=True)
    st.subheader("Level Breakdown")
    plot_level_breakdown(filtered)
    st.markdown("</div>", unsafe_allow_html=True)

    st.markdown('<div class="section-card">', unsafe_allow_html=True)
    st.subheader("Key Findings")
    findings = generate_findings(filtered)
    for finding in findings:
        st.markdown(f'<div class="finding">{finding}</div>', unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)

with tab2:
    st.markdown('<div class="section-card">', unsafe_allow_html=True)
    st.subheader("Place Comparison")
    plot_region_comparison(filtered)
    st.markdown("</div>", unsafe_allow_html=True)

    st.markdown('<div class="section-card">', unsafe_allow_html=True)
    st.subheader("Place Summary")
    reg_sum = region_summary(filtered)
    if reg_sum.empty:
        st.info("Place summary is not available.")
    else:
        st.dataframe(
            reg_sum.style.format({
                "green_cover": "{:.3f}",
                "water_presence": "{:.3f}",
                "surface_heat": "{:.2f}",
                "change_score": "{:.3f}",
            }),
            use_container_width=True,
            height=420,
        )
    st.markdown("</div>", unsafe_allow_html=True)

with tab3:
    st.markdown('<div class="section-card">', unsafe_allow_html=True)
    st.subheader("Time View")
    plot_timeline(filtered)
    st.markdown("</div>", unsafe_allow_html=True)

with tab4:
    st.markdown('<div class="section-card">', unsafe_allow_html=True)
    st.subheader("Map View")
    show_map(filtered)
    st.markdown("</div>", unsafe_allow_html=True)

with tab5:
    st.markdown('<div class="section-card">', unsafe_allow_html=True)
    st.subheader("Filtered Data")
    st.dataframe(filtered, use_container_width=True, height=420)

    st.markdown("### Situation Summary")
    evt_sum = event_summary(filtered)
    st.dataframe(
        evt_sum.style.format({
            "green_cover": "{:.3f}",
            "water_presence": "{:.3f}",
            "surface_heat": "{:.2f}",
            "change_score": "{:.3f}",
        }),
        use_container_width=True,
        height=260,
    )

    d1, d2 = st.columns(2)
    with d1:
        st.download_button(
            label="Download filtered data",
            data=downloadable_csv(filtered),
            file_name="filtered_environmental_data.csv",
            mime="text/csv",
        )
    with d2:
        st.download_button(
            label="Download place summary",
            data=downloadable_csv(region_summary(filtered) if "region" in filtered.columns else pd.DataFrame()),
            file_name="place_summary.csv",
            mime="text/csv",
        )

    st.markdown("</div>", unsafe_allow_html=True)