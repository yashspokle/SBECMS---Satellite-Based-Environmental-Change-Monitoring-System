import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import pydeck as pdk

# =========================================================
# PAGE CONFIG
# =========================================================

st.set_page_config(
    page_title="AI Environmental Monitoring System",
    page_icon="🌍",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# =========================================================
# MODERN UI
# =========================================================

st.markdown("""
<style>

@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;700;800&display=swap');

html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
}

.stApp {
    background: linear-gradient(135deg, #050816 0%, #0d1326 40%, #111827 100%);
    color: white;
}

section[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #111827 0%, #0b1220 100%);
    border-right: 1px solid rgba(255,255,255,0.08);
}

.block-container {
    padding-top: 1rem;
    max-width: 1500px;
}

.main-title {
    font-size: 3rem;
    font-weight: 800;
    background: linear-gradient(90deg,#60a5fa,#34d399,#f59e0b);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
}

.subtitle {
    color: #9ca3af;
    font-size: 1rem;
}

.glass {
    background: rgba(255,255,255,0.05);
    border: 1px solid rgba(255,255,255,0.08);
    backdrop-filter: blur(12px);
    border-radius: 20px;
    padding: 20px;
    box-shadow: 0 8px 32px rgba(0,0,0,0.25);
}

.metric-card {
    background: linear-gradient(
        135deg,
        rgba(17,24,39,0.95),
        rgba(31,41,55,0.95)
    );

    border-radius: 24px;
    padding: 22px;

    border: 1px solid rgba(255,255,255,0.08);

    transition: all 0.35s ease;

    box-shadow:
        0 10px 30px rgba(0,0,0,0.25);
}

.metric-card:hover {

    transform:
        translateY(-8px)
        scale(1.02);

    box-shadow:
        0 20px 40px rgba(0,0,0,0.4);
}

.metric-title {
    color: #9ca3af;
    font-size: 0.9rem;
}

.metric-value {
    font-size: 2rem;
    font-weight: 800;
    color: white;
}

.metric-sub {
    color: #60a5fa;
    font-size: 0.8rem;
}

.ai-box {
    background: linear-gradient(135deg,#1e3a8a,#111827);
    border-left: 5px solid #60a5fa;
    padding: 16px;
    border-radius: 16px;
    margin-bottom: 10px;
}

.recommend-box {
    background: linear-gradient(135deg,#064e3b,#111827);
    border-left: 5px solid #34d399;
    padding: 14px;
    border-radius: 14px;
    margin-bottom: 10px;
}

</style>
""", unsafe_allow_html=True)

# =========================================================
# HEADER
# =========================================================

st.markdown("""
<div style="
display:flex;
justify-content:space-between;
align-items:center;
padding:14px 24px;
margin-top:15px;
margin-bottom:20px;
background:rgba(255,255,255,0.04);
border-radius:18px;
border:1px solid rgba(255,255,255,0.05);
">

<div style="font-size:1.1rem;font-weight:700;">
🌍 Live Monitoring Dashboard
</div>

<div style="display:flex;gap:12px;">

<div style="
padding:8px 14px;
background:#1d4ed8;
border-radius:12px;
">
Analytics
</div>

<div style="
padding:8px 14px;
background:#059669;
border-radius:12px;
">
AI Insights
</div>

<div style="
padding:8px 14px;
background:#7c3aed;
border-radius:12px;
">
Satellite Map
</div>

</div>

</div>
""", unsafe_allow_html=True)

# =========================================================
# COLUMN ALIASES
# =========================================================

COLUMN_ALIASES = {

    "land_use": [
        "land_use",
        "land cover",
        "landcover",
        "class",
        "terrain",
        "lulc",
        "label"
    ],

    "event": [
        "event",
        "disaster",
        "hazard",
        "condition",
        "risk"
    ],

    "ndvi": [
        "ndvi",
        "vegetation",
        "veg_index",
        "greenness"
    ],

    "ndwi": [
        "ndwi",
        "water_index",
        "moisture"
    ],

    "lst_celsius": [
        "lst",
        "temperature",
        "temp",
        "surface_temp",
        "surface_temperature",
        "heat",
        "thermal"
    ],

    "change_index": [
        "change",
        "change_index",
        "severity",
        "impact",
        "risk_score",
        "damage"
    ],

    "latitude": [
        "latitude",
        "lat",
        "y"
    ],

    "longitude": [
        "longitude",
        "lon",
        "lng",
        "x"
    ],

    "region": [
        "region",
        "location",
        "district",
        "city",
        "state"
    ],

    "date": [
        "date",
        "time",
        "timestamp"
    ]
}

# =========================================================
# AUTO COLUMN MAPPING
# =========================================================

def auto_map_columns(df):

    renamed = {}

    lower_cols = {c.lower(): c for c in df.columns}

    for standard_col, aliases in COLUMN_ALIASES.items():

        for alias in aliases:

            if alias.lower() in lower_cols:

                renamed[lower_cols[alias.lower()]] = standard_col
                break

    df = df.rename(columns=renamed)

    return df

# =========================================================
# DEMO DATA
# =========================================================

def generate_demo_data(n=1000):

    np.random.seed(42)

    return pd.DataFrame({

        "land_use": np.random.choice(
            ["Forest","Urban","Water","Agriculture"],
            n
        ),

        "event": np.random.choice(
            ["None","Flood","Wildfire","Drought"],
            n
        ),

        "ndvi": np.random.uniform(-0.1,0.9,n),

        "ndwi": np.random.uniform(-0.2,0.7,n),

        "lst_celsius": np.random.uniform(18,45,n),

        "change_index": np.random.uniform(0,1,n),

        "latitude": np.random.uniform(8,35,n),

        "longitude": np.random.uniform(68,90,n),

        "region": np.random.choice(
            ["North","South","East","West"],
            n
        ),

        "date": pd.date_range(
            start="2024-01-01",
            periods=n,
            freq="D"
        )
    })

# =========================================================
# SIDEBAR
# =========================================================

with st.sidebar:

    st.header("⚙️ Controls")

    uploaded_file = st.file_uploader(
        "Upload CSV or Excel",
        type=["csv","xlsx"]
    )

    use_demo = st.checkbox(
        "Use Demo Dataset",
        value=True
    )

# =========================================================
# LOAD DATA
# =========================================================

if uploaded_file:

    try:

        if uploaded_file.name.endswith(".csv"):
            raw_df = pd.read_csv(uploaded_file)

        else:
            raw_df = pd.read_excel(uploaded_file)

    except Exception as e:

        st.error(f"Error loading file: {e}")
        st.stop()

else:

    raw_df = generate_demo_data()

# =========================================================
# CLEAN DATA
# =========================================================

df = auto_map_columns(raw_df)

df.columns = [c.lower().strip() for c in df.columns]

# =========================================================
# DEBUG
# =========================================================

st.sidebar.write("Detected Columns")
st.sidebar.write(df.columns.tolist())

# =========================================================
# HANDLE MISSING COLUMNS SAFELY
# =========================================================

required_defaults = {

    "land_use": "Unknown",

    "event": "None",

    "ndvi": np.random.uniform(0.2,0.8,len(df)),

    "ndwi": np.random.uniform(-0.2,0.5,len(df)),

    "lst_celsius": np.random.uniform(20,40,len(df)),

    "change_index": np.random.uniform(0.1,0.6,len(df))
}

missing = []

for col, default_value in required_defaults.items():

    if col not in df.columns:

        missing.append(col)

        df[col] = default_value

if missing:

    st.warning(f"""
    Missing columns auto-generated:
    {', '.join(missing)}
    """)

# =========================================================
# NUMERIC CONVERSION
# =========================================================

numeric_cols = [
    "ndvi",
    "ndwi",
    "lst_celsius",
    "change_index",
    "latitude",
    "longitude"
]

for col in numeric_cols:

    if col in df.columns:

        df[col] = pd.to_numeric(
            df[col],
            errors="coerce"
        )

# =========================================================
# FILTERS
# =========================================================

with st.sidebar:

    st.header("📊 Filters")

    land_options = df["land_use"].dropna().unique()

    selected_land = st.multiselect(
        "Land Use",
        land_options,
        default=land_options
    )

    event_options = df["event"].dropna().unique()

    selected_event = st.multiselect(
        "Event",
        event_options,
        default=event_options
    )

filtered = df[
    (df["land_use"].isin(selected_land)) &
    (df["event"].isin(selected_event))
].copy()

# =========================================================
# AI MODEL
# =========================================================

filtered["risk"] = np.where(
    (
        (filtered["change_index"] > 0.5) |
        (filtered["lst_celsius"] > 35)
    ),
    1,
    0
)

features = filtered[[
    "ndvi",
    "ndwi",
    "lst_celsius",
    "change_index"
]].fillna(0)

target = filtered["risk"]

X_train, X_test, y_train, y_test = train_test_split(
    features,
    target,
    test_size=0.2,
    random_state=42
)

model = RandomForestClassifier()

model.fit(X_train, y_train)

preds = model.predict(X_test)

acc = accuracy_score(y_test, preds)

# =========================================================
# METRICS
# =========================================================

c1,c2,c3,c4 = st.columns(4)

metrics = [

    (
        "Records",
        len(filtered),
        "Dataset Rows"
    ),

    (
        "Avg Temp",
        round(filtered["lst_celsius"].mean(),2),
        "Surface Heat"
    ),

    (
        "Avg NDVI",
        round(filtered["ndvi"].mean(),2),
        "Vegetation"
    ),

    (
        "AI Accuracy",
        f"{acc:.2%}",
        "Prediction Model"
    )
]

for col, metric in zip([c1,c2,c3,c4], metrics):

    with col:

        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-title">{metric[0]}</div>
            <div class="metric-value">{metric[1]}</div>
            <div class="metric-sub">{metric[2]}</div>
        </div>
        """, unsafe_allow_html=True)

# =========================================================
# TABS
# =========================================================

tab1,tab2,tab3,tab4,tab5 = st.tabs([
    "📈 Analytics",
    "🧠 AI Insights",
    "🗺️ Map",
    "📊 Data",
    "📉 Trends"
])

# =========================================================
# ANALYTICS
# =========================================================

# =========================================================
# ADVANCED ANALYTICS
# =========================================================

with tab1:

    st.subheader("📈 Environmental Analytics")

    # ============================================
    # CHART 1
    # ============================================

    col1, col2 = st.columns(2)

    with col1:

        land_summary = filtered.groupby(
            "land_use"
        ).agg({

            "ndvi":"mean",
            "lst_celsius":"mean",
            "change_index":"mean"

        }).reset_index()

        fig = px.bar(
            land_summary,
            x="land_use",
            y="ndvi",
            color="change_index",
            title="Vegetation by Land Type",
            text_auto=".2f"
        )

        fig.update_layout(
            template="plotly_dark",
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            height=450
        )

        st.plotly_chart(
            fig,
            use_container_width=True
        )

    # ============================================
    # CHART 2
    # ============================================

    with col2:

        event_summary = filtered.groupby(
            "event"
        ).agg({

            "lst_celsius":"mean",
            "change_index":"mean"

        }).reset_index()

        fig2 = px.line(
            event_summary,
            x="event",
            y="lst_celsius",
            markers=True,
            line_shape="spline",
            title="Temperature by Event"
        )

        fig2.update_traces(
            line=dict(width=4)
        )

        fig2.update_layout(
            template="plotly_dark",
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            height=450
        )

        st.plotly_chart(
            fig2,
            use_container_width=True
        )

    # ============================================
    # HEATMAP
    # ============================================

    st.subheader("🔥 Correlation Heatmap")

    corr_cols = [
        "ndvi",
        "ndwi",
        "lst_celsius",
        "change_index"
    ]

    corr = filtered[corr_cols].corr()

    heatmap = go.Figure(
        data=go.Heatmap(
            z=corr.values,
            x=corr.columns,
            y=corr.columns,
            text=np.round(corr.values,2),
            texttemplate="%{text}",
            colorscale="Turbo"
        )
    )

    heatmap.update_layout(
        template="plotly_dark",
        height=500,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)"
    )

    st.plotly_chart(
        heatmap,
        use_container_width=True
    )

    # ============================================
    # REGION ANALYSIS
    # ============================================

    if "region" in filtered.columns:

        st.subheader("🌍 Regional Analysis")

        region_df = filtered.groupby(
            "region"
        ).agg({

            "change_index":"mean"

        }).reset_index()

        fig3 = px.area(
            region_df,
            x="region",
            y="change_index",
            title="Environmental Change by Region"
        )

        fig3.update_layout(
            template="plotly_dark",
            height=500,
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)"
        )

        st.plotly_chart(
            fig3,
            use_container_width=True
        )

with tab2:

    st.subheader("🧠 AI Environmental Intelligence")

    insights = []

    if filtered["lst_celsius"].mean() > 35:

        insights.append(
            "High surface temperature detected."
        )

    if filtered["ndvi"].mean() < 0.3:

        insights.append(
            "Vegetation health is critically low."
        )

    if filtered["ndwi"].mean() < 0:

        insights.append(
            "Water stress detected."
        )

    if filtered["change_index"].mean() > 0.5:

        insights.append(
            "Rapid environmental changes observed."
        )

    for insight in insights:

        st.markdown(f"""
        <div class="ai-box">
            🚨 {insight}
        </div>
        """, unsafe_allow_html=True)

    st.subheader("🌱 AI Recommendations")

    recommendations = [

        "Increase vegetation in urban zones.",

        "Deploy wildfire monitoring systems.",

        "Improve flood detection infrastructure.",

        "Use satellite thermal monitoring."
    ]

    for rec in recommendations:

        st.markdown(f"""
        <div class="recommend-box">
            ✅ {rec}
        </div>
        """, unsafe_allow_html=True)

# =========================================================
# MAP
# =========================================================

with tab3:

    if (
        "latitude" in filtered.columns and
        "longitude" in filtered.columns
    ):

        st.subheader("🌍 Satellite Risk Map")

        filtered["risk_size"] = (
            filtered["change_index"] * 50000
        )

        layer = pdk.Layer(
            "ScatterplotLayer",
            data=filtered,
            get_position='[longitude, latitude]',
            get_radius='risk_size',
            get_fill_color='[255,100,100,180]',
            pickable=True
        )

        view_state = pdk.ViewState(
            latitude=float(filtered["latitude"].mean()),
            longitude=float(filtered["longitude"].mean()),
            zoom=4,
            pitch=40
        )

        deck = pdk.Deck(
            layers=[layer],
            initial_view_state=view_state,
            map_style='mapbox://styles/mapbox/dark-v11'
        )

        st.pydeck_chart(deck)

    else:

        st.warning(
            "Latitude/Longitude columns not found."
        )

# =========================================================
# DATA
# =========================================================

with tab4:

    st.dataframe(
        filtered,
        use_container_width=True,
        height=600
    )

    csv = filtered.to_csv(index=False).encode('utf-8')

    st.download_button(
        "⬇ Download CSV",
        csv,
        "environmental_data.csv",
        "text/csv"
    )

# =========================================================
# TRENDS
# =========================================================

with tab5:

    if "date" in filtered.columns:

        filtered["date"] = pd.to_datetime(
            filtered["date"],
            errors="coerce"
        )

        monthly = filtered.groupby(
            filtered["date"].dt.month
        ).agg({

            "ndvi":"mean",

            "lst_celsius":"mean",

            "change_index":"mean"

        }).reset_index()

        fig = make_subplots(
            rows=3,
            cols=1,
            subplot_titles=(
                "Vegetation",
                "Surface Heat",
                "Environmental Change"
            )
        )

        fig.add_trace(
            go.Scatter(
                x=monthly["date"],
                y=monthly["ndvi"],
                mode='lines+markers'
            ),
            row=1,
            col=1
        )

        fig.add_trace(
            go.Scatter(
                x=monthly["date"],
                y=monthly["lst_celsius"],
                mode='lines+markers'
            ),
            row=2,
            col=1
        )

        fig.add_trace(
            go.Scatter(
                x=monthly["date"],
                y=monthly["change_index"],
                mode='lines+markers'
            ),
            row=3,
            col=1
        )

        fig.update_layout(
            height=800,
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            font_color='white'
        )

        st.plotly_chart(
            fig,
            use_container_width=True
        )

# =========================================================
# FOOTER
# =========================================================

st.markdown("""
<br><br>

<div style='text-align:center;color:gray'>

AI Powered Environmental Intelligence Dashboard

</div>
""", unsafe_allow_html=True)