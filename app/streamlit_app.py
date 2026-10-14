import streamlit as st
import folium
from streamlit_folium import st_folium
import geopandas as gpd
import pandas as pd
import numpy as np
import osmnx as ox
import networkx as nx
import joblib
from pathlib import Path
import json
import math
import re
import matplotlib.pyplot as plt
import seaborn as sns
from shapely.geometry import Point

# ─────────────────────────────────────────
# PAGE CONFIGURATION
# ─────────────────────────────────────────
st.set_page_config(
    page_title="CARS — Chennai Accessibility & Risk System",
    layout="wide",
    page_icon=None,
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────
# GLOBAL HIGH-END DESIGN SYSTEM
# ─────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@300;400;500;600;700;800&family=JetBrains+Mono:wght@400;500;600;700&display=swap');

:root {
    --bg-base: #070d18;
    --bg-surface: #0c1424;
    --bg-card: rgba(13, 22, 39, 0.75);
    --bg-card-hover: rgba(20, 32, 56, 0.9);
    --border-subtle: rgba(59, 130, 246, 0.12);
    --border-glow: rgba(56, 189, 248, 0.35);
    --text-primary: #f8fafc;
    --text-secondary: #94a3b8;
    --text-muted: #64748b;
    --accent-blue: #3b82f6;
    --accent-cyan: #38bdf8;
    --accent-emerald: #10b981;
    --accent-rose: #f43f5e;
}

html, body, [data-testid="stAppViewContainer"], [data-testid="stApp"] {
    font-family: 'Plus Jakarta Sans', -apple-system, sans-serif !important;
    background-color: var(--bg-base) !important;
    background-image: 
        radial-gradient(at 15% 15%, rgba(14, 165, 233, 0.08) 0px, transparent 50%),
        radial-gradient(at 85% 85%, rgba(59, 130, 246, 0.06) 0px, transparent 50%);
    background-attachment: fixed !important;
    color: var(--text-primary) !important;
}

/* ── Custom Sleek Scrollbar ── */
::-webkit-scrollbar {
    width: 6px;
    height: 6px;
}
::-webkit-scrollbar-track {
    background: #070d18;
}
::-webkit-scrollbar-thumb {
    background: #1e293b;
    border-radius: 4px;
}
::-webkit-scrollbar-thumb:hover {
    background: #334155;
}

/* ── Map Container Frame ── */
iframe {
    border-radius: 14px !important;
    border: 1px solid var(--border-subtle) !important;
    box-shadow: 0 16px 36px -10px rgba(0, 0, 0, 0.75) !important;
    transition: border-color 0.3s ease;
}
iframe:hover {
    border-color: var(--border-glow) !important;
}

/* ── Sidebar Styling ── */
[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #091120 0%, #060b14 100%) !important;
    border-right: 1px solid rgba(255, 255, 255, 0.06);
    box-shadow: 4px 0 24px rgba(0, 0, 0, 0.4);
}
[data-testid="stSidebar"] * {
    color: #cbd5e1 !important;
}
[data-testid="stSidebar"] hr {
    border-color: rgba(255, 255, 255, 0.08) !important;
    margin: 18px 0 !important;
}
[data-testid="stSidebar"] h2 {
    font-size: 1.15rem !important;
    font-weight: 700 !important;
    letter-spacing: -0.3px;
    color: #f1f5f9 !important;
}
[data-testid="stSidebar"] h3 {
    font-size: 0.95rem !important;
    font-weight: 700 !important;
    letter-spacing: -0.2px;
    color: #e2e8f0 !important;
}

/* ── Brand Header ── */
.app-header {
    background: linear-gradient(135deg, rgba(13, 25, 48, 0.85) 0%, rgba(15, 32, 64, 0.7) 100%);
    backdrop-filter: blur(12px);
    -webkit-backdrop-filter: blur(12px);
    border: 1px solid var(--border-subtle);
    border-radius: 14px;
    padding: 24px 32px;
    margin-bottom: 24px;
    display: flex;
    justify-content: space-between;
    align-items: center;
    box-shadow: 0 12px 32px -8px rgba(0, 0, 0, 0.6);
    position: relative;
    overflow: hidden;
}
.app-header::before {
    content: '';
    position: absolute;
    top: 0;
    left: 0;
    width: 100%;
    height: 2px;
    background: linear-gradient(90deg, #0284c7, #38bdf8, #818cf8, transparent);
}
.app-header-left h1 {
    margin: 0 !important;
    font-size: 1.65rem !important;
    font-weight: 800 !important;
    letter-spacing: -0.6px;
    background: linear-gradient(135deg, #ffffff 0%, #cbd5e1 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
}
.app-header-left p {
    margin: 6px 0 0 0;
    font-size: 0.84rem;
    color: #94a3b8;
    letter-spacing: 0.2px;
}
.badge-live {
    display: inline-flex;
    align-items: center;
    gap: 8px;
    background: rgba(16, 185, 129, 0.1);
    border: 1px solid rgba(16, 185, 129, 0.3);
    color: #34d399;
    padding: 6px 14px;
    border-radius: 999px;
    font-size: 0.75rem;
    font-weight: 700;
    letter-spacing: 0.8px;
    text-transform: uppercase;
}
.badge-dot {
    width: 8px;
    height: 8px;
    background-color: #10b981;
    border-radius: 50%;
    box-shadow: 0 0 10px #10b981;
    animation: pulse-dot 2s infinite;
}
@keyframes pulse-dot {
    0% { transform: scale(0.95); opacity: 0.8; }
    50% { transform: scale(1.15); opacity: 1; }
    100% { transform: scale(0.95); opacity: 0.8; }
}

/* ── Section Titles ── */
.section-heading {
    font-size: 0.88rem;
    font-weight: 700;
    color: #64748b;
    letter-spacing: 1.5px;
    text-transform: uppercase;
    display: flex;
    align-items: center;
    gap: 12px;
    margin-bottom: 18px;
    margin-top: 14px;
}
.section-heading::after {
    content: '';
    flex: 1;
    height: 1px;
    background: linear-gradient(90deg, rgba(59, 130, 246, 0.25), transparent);
}

/* ── Metric Cards ── */
[data-testid="stMetric"] {
    background: var(--bg-card) !important;
    backdrop-filter: blur(8px);
    -webkit-backdrop-filter: blur(8px);
    border: 1px solid var(--border-subtle) !important;
    border-radius: 12px !important;
    padding: 18px 22px !important;
    box-shadow: 0 8px 20px -6px rgba(0, 0, 0, 0.5) !important;
    transition: transform 0.2s ease, border-color 0.2s ease, box-shadow 0.2s ease;
}
[data-testid="stMetric"]:hover {
    transform: translateY(-2px);
    border-color: var(--border-glow) !important;
    box-shadow: 0 12px 28px -6px rgba(0, 0, 0, 0.6) !important;
}
[data-testid="stMetricLabel"] {
    color: var(--text-muted) !important;
    font-size: 0.72rem !important;
    font-weight: 600 !important;
    text-transform: uppercase !important;
    letter-spacing: 0.8px !important;
}
[data-testid="stMetricValue"] {
    color: #38bdf8 !important;
    font-size: 1.75rem !important;
    font-weight: 800 !important;
    font-family: 'JetBrains Mono', monospace !important;
    letter-spacing: -0.5px;
}

/* ── Form Inputs ── */
[data-testid="stTextInput"] input,
[data-testid="stSelectbox"] > div > div {
    background: #091222 !important;
    border: 1px solid var(--border-subtle) !important;
    color: #f1f5f9 !important;
    border-radius: 8px !important;
    font-size: 0.88rem !important;
    transition: all 0.2s ease;
}
[data-testid="stTextInput"] input:focus,
[data-testid="stSelectbox"] > div > div:focus-within {
    border-color: var(--accent-cyan) !important;
    box-shadow: 0 0 0 2px rgba(56, 189, 248, 0.2) !important;
}

/* ── Radio Buttons ── */
[data-testid="stRadio"] > div {
    background: #081120;
    border: 1px solid var(--border-subtle);
    border-radius: 10px;
    padding: 10px 14px;
}

/* ── Buttons ── */
[data-testid="stFormSubmitButton"] > button,
[data-testid="stButton"] > button {
    background: linear-gradient(135deg, #0284c7 0%, #2563eb 100%) !important;
    color: #ffffff !important;
    border: 1px solid rgba(56, 189, 248, 0.3) !important;
    border-radius: 8px !important;
    font-weight: 600 !important;
    font-size: 0.86rem !important;
    letter-spacing: 0.4px !important;
    padding: 10px 20px !important;
    transition: all 0.25s ease !important;
    box-shadow: 0 4px 14px rgba(2, 132, 199, 0.3) !important;
}
[data-testid="stFormSubmitButton"] > button:hover,
[data-testid="stButton"] > button:hover {
    background: linear-gradient(135deg, #0ea5e9 0%, #3b82f6 100%) !important;
    box-shadow: 0 6px 20px rgba(14, 165, 233, 0.45) !important;
    transform: translateY(-1px);
}
[data-testid="stFormSubmitButton"] > button:active,
[data-testid="stButton"] > button:active {
    transform: translateY(0);
}

/* ── Alerts & Expanders ── */
[data-testid="stAlert"] {
    background: rgba(15, 23, 42, 0.85) !important;
    border-radius: 10px !important;
    border: 1px solid var(--border-subtle) !important;
    backdrop-filter: blur(8px);
}
[data-testid="stExpander"] {
    background: var(--bg-card) !important;
    border: 1px solid var(--border-subtle) !important;
    border-radius: 10px !important;
    backdrop-filter: blur(8px);
}
[data-testid="stExpander"]:hover {
    border-color: rgba(59, 130, 246, 0.25) !important;
}
hr {
    border-color: var(--border-subtle) !important;
    margin: 24px 0 !important;
}
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────
# SESSION STATE INITIALISATION
# ─────────────────────────────────────────
if 'blocked_blocks' not in st.session_state:
    st.session_state.blocked_blocks = set()

if 'modified_edges' not in st.session_state:
    st.session_state.modified_edges = []

if 'blocked_nodes' not in st.session_state:
    st.session_state.blocked_nodes = set()


# ─────────────────────────────────────────
# CACHED DATA LOADING
# ─────────────────────────────────────────
@st.cache_resource
def load_model():
    return joblib.load(Path("models/random_forest.pkl"))

@st.cache_data
def load_grid():
    grid = gpd.read_file(Path("data/processed/grid/chennai_grid_with_features.geojson"), engine="pyogrio")
    grid['block_no'] = grid.index + 1
    return grid

@st.cache_data
def load_hotspots():
    path = Path("data/processed/flood_history/flood_hotspots.geojson")
    if path.exists():
        return gpd.read_file(path, engine="pyogrio")
    return None

@st.cache_resource
def load_road_graph():
    G = ox.load_graphml(Path("data/processed/roads/chennai_roads_risk.graphml"))
    for u, v, k, data in G.edges(keys=True, data=True):
        length = float(data.get('length', 100.0))
        prob   = float(data.get('flood_prob', 0.0))
        if prob > 0.01:
            base_risk_len = length * (1.0 + prob * 50.0)
        else:
            base_risk_len = length
        data['base_risk_adjusted_length'] = base_risk_len
        data['risk_adjusted_length']      = base_risk_len
        data['length']                    = length
        data['flood_prob']                = prob
        data['base_flood_prob']           = prob
        data['is_user_blocked']           = False
    return G

@st.cache_resource
def get_graph_spatial_index(_G):
    node_ids = np.array(list(_G.nodes))
    xs = np.array([_G.nodes[n]['x'] for n in node_ids], dtype=np.float64)
    ys = np.array([_G.nodes[n]['y'] for n in node_ids], dtype=np.float64)
    return node_ids, xs, ys

@st.cache_data
def load_safe_institutions(_grid_gdf):
    inst_path = Path("data/processed/institutions/chennai_institutions_cleaned.geojson")
    if not inst_path.exists():
        return gpd.GeoDataFrame()
    pois = gpd.read_file(inst_path, engine="pyogrio")
    safe_cells = _grid_gdf[_grid_gdf['flood_prob'] <= 0.01].copy()
    if pois.empty or safe_cells.empty:
        return gpd.GeoDataFrame()
    if pois.crs != safe_cells.crs:
        pois = pois.to_crs(safe_cells.crs)
    safe_pois = gpd.sjoin(pois, safe_cells[['cell_id', 'block_no', 'flood_prob', 'geometry']], how='inner')
    safe_pois = safe_pois.reset_index(drop=True)
    return safe_pois[['name', 'block_no', 'flood_prob', 'geometry']]

@st.cache_data
def load_all_institutions(_grid_gdf):
    inst_path = Path("data/processed/institutions/chennai_institutions_cleaned.geojson")
    if not inst_path.exists():
        return gpd.GeoDataFrame()
    pois = gpd.read_file(inst_path, engine="pyogrio")
    if pois.empty:
        return gpd.GeoDataFrame()
    if pois.crs != _grid_gdf.crs:
        pois = pois.to_crs(_grid_gdf.crs)
    joined = gpd.sjoin(pois, _grid_gdf[['cell_id', 'block_no', 'flood_prob', 'geometry']], how='left')
    joined = joined[~joined.index.duplicated(keep='first')].copy()
    joined['flood_prob'] = joined['flood_prob'].fillna(0.0)
    joined['block_no'] = joined['block_no'].fillna(-1).astype(int)
    return joined


# ── Clean, unobtrusive loading spinner ──
with st.spinner("Initialising geospatial graph network and AI models..."):
    rf_model = load_model()
    grid_gdf = load_grid()
    hotspots_gdf = load_hotspots()
    G_base = load_road_graph()
    node_ids, node_xs, node_ys = get_graph_spatial_index(G_base)

    FEATURES = [
        'elevation_mean', 'slope_mean', 'rainfall_24h', 'rainfall_72h',
        'rainfall_7day', 'distance_to_water_m', 'distance_to_drain_m',
        'drainage_density', 'built_up_percentage', 'water_percentage'
    ]
    X = grid_gdf[FEATURES].fillna(0)
    grid_gdf['flood_prob'] = rf_model.predict_proba(X)[:, 1]
    safe_institutions_gdf = load_safe_institutions(grid_gdf)
    all_institutions_gdf = load_all_institutions(grid_gdf)


# ─────────────────────────────────────────
# TOP APPLICATION HEADER BANNER
# ─────────────────────────────────────────
st.markdown("""
<div class="app-header">
    <div class="app-header-left">
        <h1>CARS &mdash; Chennai Accessibility &amp; Risk System</h1>
        <p>Geospatial Flood Intelligence &nbsp;&middot;&nbsp; Risk-Aware Route Optimization &nbsp;&middot;&nbsp; Real-Time Hazard Reporting</p>
    </div>
    <div class="badge-live">
        <span class="badge-dot"></span>
        Operational Live
    </div>
</div>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────
# LOCATIONS & LANDMARK MAPPING
# ─────────────────────────────────────────
LOCATIONS = {
    "Adyar": (13.0012, 80.2565), "Alwarpet": (13.0335, 80.2524), "Ambattur": (13.1143, 80.1481),
    "Aminjikarai": (13.0732, 80.2210), "Ashok Nagar": (13.0371, 80.2122), "Avadi": (13.1143, 80.1022),
    "Ayanavaram": (13.0970, 80.2366), "Besant Nagar": (13.0003, 80.2737), "Chetpet": (13.0714, 80.2417),
    "Choolaimedu": (13.0645, 80.2274), "Chromepet": (12.9516, 80.1406), "Egmore": (13.0732, 80.2609),
    "Ekkatuthangal": (13.0237, 80.2037), "Ennore": (13.2144, 80.3243), "George Town": (13.0978, 80.2862),
    "Gopalapuram": (13.0478, 80.2562), "Guindy": (13.0067, 80.2206), "K.K. Nagar": (13.0400, 80.1973),
    "Kilpauk": (13.0818, 80.2405), "Kodambakkam": (13.0521, 80.2255), "Korattur": (13.1118, 80.1834),
    "Korukkupet": (13.1172, 80.2785), "Kotturpuram": (13.0245, 80.2444), "Koyambedu": (13.0667, 80.1914),
    "Madhavaram": (13.1488, 80.2306), "Madipakkam": (12.9647, 80.1961), "Mandaveli": (13.0270, 80.2657),
    "Medavakkam": (12.9171, 80.1923), "Meenambakkam (Airport)": (12.9822, 80.1636), "Mogappair": (13.0838, 80.1747),
    "Mount Road (Anna Salai)": (13.0592, 80.2642), "Mylapore": (13.0368, 80.2676), "Nandanam": (13.0305, 80.2405),
    "Nanganallur": (12.9774, 80.1898), "Navallur": (12.8465, 80.2265), "Neelankarai": (12.9490, 80.2555),
    "Nungambakkam": (13.0569, 80.2425), "Pallavaram": (12.9675, 80.1491), "Pammal": (12.9749, 80.1338),
    "Parrys": (13.0886, 80.2860), "Pattabiram": (13.1218, 80.0596), "Perambur": (13.1112, 80.2458),
    "Perungudi": (12.9654, 80.2461), "Poonamallee": (13.0473, 80.0945), "Porur": (13.0354, 80.1562),
    "Purasawalkam": (13.0885, 80.2526), "Ramapuram": (13.0313, 80.1818), "Royapettah": (13.0559, 80.2635),
    "Royapuram": (13.1118, 80.2941), "Saidapet": (13.0213, 80.2231), "Santhome": (13.0323, 80.2783),
    "Selaiyur": (12.9099, 80.1432), "Shenoy Nagar": (13.0788, 80.2227), "Sholinganallur": (12.8996, 80.2279),
    "Siruseri": (12.8273, 80.2209), "T Nagar": (13.0418, 80.2341), "Tambaram": (12.9249, 80.1000),
    "Taramani": (12.9863, 80.2432), "Teynampet": (13.0405, 80.2503), "Thiruvanmiyur": (12.9863, 80.2589),
    "Thoraipakkam": (12.9404, 80.2376), "Tondiarpet": (13.1278, 80.2785), "Triplicane": (13.0588, 80.2759),
    "Vadapalani": (13.0500, 80.2121), "Valasaravakkam": (13.0396, 80.1738), "Velachery": (12.9815, 80.2180),
    "Vepery": (13.0841, 80.2631), "Villivakkam": (13.1075, 80.2078), "Virugambakkam": (13.0487, 80.1901),
    "Washermanpet": (13.1091, 80.2828), "West Mambalam": (13.0384, 80.2241)
}

@st.cache_data
def get_landmark_block_map(_grid_gdf):
    loc_to_block  = {}
    block_to_locs = {}
    for name, (lat, lon) in LOCATIONS.items():
        pt = Point(lon, lat)
        matches = _grid_gdf[_grid_gdf.geometry.contains(pt)]
        if not matches.empty:
            b_num = int(matches.iloc[0]['block_no'])
        else:
            dists = ((_grid_gdf['centroid_latitude'] - lat)**2 + (_grid_gdf['centroid_longitude'] - lon)**2)
            b_num = int(_grid_gdf.loc[dists.idxmin(), 'block_no'])
        loc_to_block[name] = b_num
        block_to_locs.setdefault(b_num, []).append(name)
    return loc_to_block, block_to_locs

loc_to_block, block_to_locs = get_landmark_block_map(grid_gdf)


def parse_block_numbers(input_str, max_block=4860):
    if not input_str or not input_str.strip():
        return set(), []
    valid_blocks   = set()
    invalid_tokens = []
    cleaned_str    = input_str

    for m in re.finditer(r'\b(\d+)\s*[-–—]\s*(\d+)\b', input_str):
        s_b, e_b = int(m.group(1)), int(m.group(2))
        if s_b > e_b:
            s_b, e_b = e_b, s_b
        if 1 <= s_b <= max_block and 1 <= e_b <= max_block:
            for b in range(s_b, min(e_b + 1, s_b + 50)):
                valid_blocks.add(b)
        else:
            invalid_tokens.append(m.group(0))
        cleaned_str = cleaned_str.replace(m.group(0), ' ')

    for m in re.finditer(r'cell_?(\d+)', cleaned_str, re.IGNORECASE):
        b = int(m.group(1)) + 1
        if 1 <= b <= max_block:
            valid_blocks.add(b)
        else:
            invalid_tokens.append(m.group(0))
        cleaned_str = cleaned_str.replace(m.group(0), ' ')

    for m in re.finditer(r'#?(\d+)', cleaned_str):
        b = int(m.group(1))
        if 1 <= b <= max_block:
            valid_blocks.add(b)
        else:
            invalid_tokens.append(m.group(0))

    return valid_blocks, invalid_tokens


# ─────────────────────────────────────────
# REAL-TIME GRAPH MODIFICATION
# ─────────────────────────────────────────
def apply_blocked_blocks_to_graph(G, blocked_blocks, grid_gdf, node_ids, xs, ys):
    if st.session_state.modified_edges:
        for u, v, k in st.session_state.modified_edges:
            if G.has_edge(u, v, k):
                G[u][v][k]['risk_adjusted_length'] = G[u][v][k]['base_risk_adjusted_length']
                G[u][v][k]['flood_prob']            = G[u][v][k]['base_flood_prob']
                G[u][v][k]['is_user_blocked']       = False
        st.session_state.modified_edges = []

    st.session_state.blocked_nodes = set()
    if not blocked_blocks:
        return

    modified          = []
    blocked_nodes_set = set()

    for block_num in blocked_blocks:
        cell_idx = block_num - 1
        if 0 <= cell_idx < len(grid_gdf):
            cell = grid_gdf.iloc[cell_idx]
            minx, miny, maxx, maxy = cell.geometry.bounds
            mask          = (xs >= minx) & (xs <= maxx) & (ys >= miny) & (ys <= maxy)
            matched_nodes = node_ids[mask]

            if len(matched_nodes) == 0:
                c_lat = float(cell['centroid_latitude'])
                c_lon = float(cell['centroid_longitude'])
                nearest = ox.distance.nearest_nodes(G, c_lon, c_lat)
                matched_nodes = [nearest]

            for n in matched_nodes:
                blocked_nodes_set.add(n)
                for u, v, k, data in G.edges(n, data=True, keys=True):
                    data['risk_adjusted_length'] = 99999999.0
                    data['flood_prob']            = 1.0
                    data['is_user_blocked']       = True
                    modified.append((u, v, k))
                for u, v, k, data in G.in_edges(n, data=True, keys=True):
                    data['risk_adjusted_length'] = 99999999.0
                    data['flood_prob']            = 1.0
                    data['is_user_blocked']       = True
                    modified.append((u, v, k))

    st.session_state.modified_edges = modified
    st.session_state.blocked_nodes  = blocked_nodes_set


apply_blocked_blocks_to_graph(G_base, st.session_state.blocked_blocks, grid_gdf, node_ids, node_xs, node_ys)
G = G_base


# ─────────────────────────────────────────
# GEOGRAPHIC UTILITIES & ROUTING
# ─────────────────────────────────────────
def haversine_dist_km(lat1, lon1, lat2, lon2):
    R    = 6371.0
    dlat = math.radians(lat2 - lat1)
    dlon = math.radians(lon2 - lon1)
    a    = (math.sin(dlat / 2) ** 2 +
            math.cos(math.radians(lat1)) * math.cos(math.radians(lat2)) * math.sin(dlon / 2) ** 2)
    return R * 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))


def find_best_evacuation_center(G_active, orig_node, start_lat, start_lon, max_radius_km=7.0):
    if safe_institutions_gdf is None or safe_institutions_gdf.empty:
        candidates = grid_gdf[grid_gdf['flood_prob'] <= 0.01].copy().reset_index(drop=True)
        if candidates.empty:
            candidates = grid_gdf[grid_gdf['flood_prob'] == grid_gdf['flood_prob'].min()].copy().reset_index(drop=True)
        candidates['target_lat']   = candidates.geometry.centroid.y
        candidates['target_lon']   = candidates.geometry.centroid.x
        candidates['display_name'] = "Designated Safe Sector"
    else:
        candidates = safe_institutions_gdf.copy()
        if 'block_no' in candidates.columns and st.session_state.blocked_blocks:
            candidates = candidates[~candidates['block_no'].isin(st.session_state.blocked_blocks)].copy().reset_index(drop=True)
        else:
            candidates = candidates.reset_index(drop=True)
        candidates['target_lat']   = candidates.geometry.y
        candidates['target_lon']   = candidates.geometry.x
        candidates['display_name'] = candidates['name']

    if candidates.empty:
        return None

    candidates['dist_km'] = candidates.apply(
        lambda r: haversine_dist_km(start_lat, start_lon, r['target_lat'], r['target_lon']),
        axis=1
    )

    lengths, paths    = nx.single_source_dijkstra(G_active, orig_node, weight='risk_adjusted_length')
    nearby_candidates = candidates[candidates['dist_km'] <= max_radius_km].copy()

    best_center = None
    lowest_cost = float('inf')

    for _, row in nearby_candidates.iterrows():
        cand_lat  = float(row['target_lat'])
        cand_lon  = float(row['target_lon'])
        dest_node = ox.distance.nearest_nodes(G_active, cand_lon, cand_lat)
        if dest_node in lengths and lengths[dest_node] < 50000000.0 and lengths[dest_node] < lowest_cost:
            lowest_cost = lengths[dest_node]
            best_center = {
                "lat": cand_lat, "lon": cand_lon, "name": str(row['display_name']),
                "prob": float(row.get('flood_prob', 0.0)), "dest_node": dest_node,
                "route": paths[dest_node], "crow_dist_km": float(row['dist_km']),
                "fallback_used": False
            }

    if best_center is None:
        for _, row in candidates.iterrows():
            cand_lat  = float(row['target_lat'])
            cand_lon  = float(row['target_lon'])
            dest_node = ox.distance.nearest_nodes(G_active, cand_lon, cand_lat)
            if dest_node in lengths and lengths[dest_node] < 50000000.0 and lengths[dest_node] < lowest_cost:
                lowest_cost = lengths[dest_node]
                best_center = {
                    "lat": cand_lat, "lon": cand_lon, "name": str(row['display_name']),
                    "prob": float(row.get('flood_prob', 0.0)), "dest_node": dest_node,
                    "route": paths[dest_node], "crow_dist_km": float(row['dist_km']),
                    "fallback_used": True
                }

    return best_center


# ─────────────────────────────────────────
# SIDEBAR NAVIGATION & INCIDENT REPORTING
# ─────────────────────────────────────────
st.sidebar.markdown("## Route Navigation")
st.sidebar.markdown("---")

with st.sidebar.form("routing_form"):
    start_loc    = st.selectbox("Origin", list(LOCATIONS.keys()), index=list(LOCATIONS.keys()).index("T Nagar"))
    routing_mode = st.radio("Navigation Objective", [
        "Navigate to Destination",
        "Find Nearest Safe Evacuation Zone (5–7 km)"
    ])
    end_loc      = st.selectbox("Destination", list(LOCATIONS.keys()),
                                index=list(LOCATIONS.keys()).index("Velachery"))
    route_type   = st.radio("Route Criteria", ["Lower-Risk Path", "Shortest Distance"])
    calculate_btn = st.form_submit_button("Compute Route", use_container_width=True)

st.sidebar.markdown("---")
st.sidebar.markdown("### Real-Time Incident Reporting")
st.sidebar.caption(
    "Flag sectors by block number to mark as flooded. "
    "Traversing road segments will be penalized and avoided automatically."
)

with st.sidebar.form("block_report_form"):
    typed_blocks   = st.text_input(
        "Block / Sector ID(s)",
        placeholder="e.g. 2955 | 2955, 2700 | 2950-2953",
        help="Enter individual block numbers, comma-separated lists, or ranges."
    )
    add_blocks_btn = st.form_submit_button("Report Flooded Sector(s)", use_container_width=True)

if add_blocks_btn and typed_blocks:
    parsed_blocks, invalid_tokens = parse_block_numbers(typed_blocks, max_block=len(grid_gdf))
    if parsed_blocks:
        st.session_state.blocked_blocks.update(parsed_blocks)
        st.sidebar.success(f"Reported {len(parsed_blocks)} sector(s) as impassable.")
        st.rerun()
    if invalid_tokens:
        st.sidebar.warning(f"Unrecognized inputs: {', '.join(invalid_tokens)}")

quick_landmark = st.sidebar.selectbox(
    "Quick Select Landmark",
    ["— Select Landmark —"] + sorted(list(LOCATIONS.keys()))
)
if st.sidebar.button("Report Selected Landmark", use_container_width=True):
    if quick_landmark != "— Select Landmark —":
        target_b = loc_to_block[quick_landmark]
        st.session_state.blocked_blocks.add(target_b)
        st.sidebar.success(f"{quick_landmark} (Block {target_b}) marked as flooded.")
        st.rerun()

if st.session_state.blocked_blocks:
    st.sidebar.markdown(f"**Reported Inundations ({len(st.session_state.blocked_blocks)} active)**")
    for b in sorted(list(st.session_state.blocked_blocks)):
        loc_desc    = f" — {block_to_locs[b][0]}" if b in block_to_locs else ""
        col_n, col_b = st.sidebar.columns([3, 1])
        col_n.markdown(
            f"<span style='color:#f87171;font-weight:600;font-size:0.83rem;'>Block {b}{loc_desc}</span>",
            unsafe_allow_html=True
        )
        if col_b.button("Remove", key=f"del_sidebar_{b}"):
            st.session_state.blocked_blocks.remove(b)
            st.rerun()
    if st.sidebar.button("Clear All Incidents", use_container_width=True):
        st.session_state.blocked_blocks = set()
        st.rerun()
else:
    st.sidebar.info("No active road closures reported.")

st.sidebar.markdown("---")
st.sidebar.markdown("### 🗺️ Map Display Layers")
st.sidebar.caption("Toggle layers directly above the map canvas to show or hide AI risk zones, landmark nodes, institutions/shelters, hazard closures, and historical hotspots.")


# ─────────────────────────────────────────
# LIVE HAZARD BANNER
# ─────────────────────────────────────────
if st.session_state.blocked_blocks:
    active_str = ", ".join([
        f"Block {b}" + (f" ({block_to_locs[b][0]})" if b in block_to_locs else "")
        for b in sorted(list(st.session_state.blocked_blocks))
    ])
    st.error(
        f"**ACTIVE FLOOD HAZARDS**: {len(st.session_state.blocked_blocks)} sector(s) flagged: {active_str}.\n\n"
        f"These zones are highlighted in high-contrast red on the geospatial map. Roads passing through are closed and rerouted."
    )


# Reference Directory Expander
with st.expander("Geospatial Sector Directory — Neighborhood to Block Number", expanded=False):
    st.markdown(
        "The study area is partitioned into standardized 550m x 550m analysis grids. "
        "Reference directory for key municipal sectors:"
    )
    dir_col1, dir_col2, dir_col3 = st.columns(3)
    loc_items  = sorted(LOCATIONS.keys())
    split_size = math.ceil(len(loc_items) / 3)
    for idx, col in enumerate([dir_col1, dir_col2, dir_col3]):
        with col:
            tbl = "| Sector / Locality | Block ID |\n| :--- | :--- |\n"
            for loc in loc_items[idx * split_size : (idx + 1) * split_size]:
                b_num       = loc_to_block.get(loc, 'N/A')
                status_mark = " (HAZARD)" if b_num in st.session_state.blocked_blocks else ""
                tbl += f"| {loc}{status_mark} | **{b_num}** |\n"
            st.markdown(tbl)


# ─────────────────────────────────────────
# SECTION 1 — FLOOD-RISK MAP & ROUTING
# ─────────────────────────────────────────
st.markdown("---")
st.markdown('<div class="section-heading">Geospatial Inundation Risk &amp; Route Guidance</div>', unsafe_allow_html=True)

start_b = loc_to_block.get(start_loc)
dest_b  = loc_to_block.get(end_loc)

if start_b in st.session_state.blocked_blocks:
    st.error(
        f"**Departure Warning**: Starting location **{start_loc}** is inside active flood sector **Block {start_b}**. "
        f"Access routes may be impassable."
    )
if "Evacuation" not in routing_mode and dest_b in st.session_state.blocked_blocks:
    st.error(
        f"**Destination Warning**: Selected destination **{end_loc}** is inside active flood sector **Block {dest_b}**. "
        f"Rerouting advised."
    )

route_coords   = []
route_stats    = {
    "distance_km": 0.0,
    "high_risk_segments": 0,
    "blocked_zones_avoided": len(st.session_state.blocked_blocks)
}
actual_end_lat, actual_end_lon = None, None
evac_name      = None
evac_crow_dist = None

if calculate_btn:
    start_lat, start_lon = LOCATIONS[start_loc]
    orig_node = ox.distance.nearest_nodes(G, start_lon, start_lat)

    if "Evacuation" in routing_mode:
        with st.spinner("Evaluating lowest-risk corridors to verified safe centers..."):
            best_evac = find_best_evacuation_center(G, orig_node, start_lat, start_lon, max_radius_km=7.0)

        if best_evac:
            if best_evac["fallback_used"]:
                st.warning(
                    f"No reachable facility within 7 km. Routing to closest accessible shelter: "
                    f"**{best_evac['name']}**."
                )
            actual_end_lat, actual_end_lon = best_evac["lat"], best_evac["lon"]
            evac_name, evac_crow_dist      = best_evac["name"], best_evac["crow_dist_km"]
            route, dest_node               = best_evac["route"], best_evac["dest_node"]
        else:
            route = None
            st.error(
                "Network Cutoff: All corridors to safe shelters are severed by reported flood incidents."
            )
    else:
        actual_end_lat, actual_end_lon = LOCATIONS[end_loc]
        dest_node = ox.distance.nearest_nodes(G, actual_end_lon, actual_end_lat)
        weight    = 'risk_adjusted_length' if route_type == "Lower-Risk Path" else 'length'
        try:
            with st.spinner("Computing optimal route trajectory..."):
                route = nx.shortest_path(G, orig_node, dest_node, weight=weight)
            total_cost = nx.path_weight(G, route, weight=weight)
            if total_cost >= 50000000.0:
                route = None
                st.error(
                    f"Route Disconnected: All corridors to **{end_loc}** are impassable due to active flood sectors."
                )
        except nx.NetworkXNoPath:
            route = None
            st.error(
                f"No Route Available: Destination **{end_loc}** is unreachable via current road network."
            )

    if actual_end_lat is not None and start_lat == actual_end_lat and start_lon == actual_end_lon:
        st.warning("Origin and destination coordinates are identical.")
    elif route:
        dist_m = 0
        for u, v in zip(route[:-1], route[1:]):
            edge_dict = G[u][v]
            if isinstance(edge_dict, dict) and 0 in edge_dict:
                edge_data = min(edge_dict.values(), key=lambda d: d.get('risk_adjusted_length', 100))
            else:
                edge_data = edge_dict
            dist_m += float(edge_data.get('length', 100))
            if float(edge_data.get('flood_prob', 0)) > 0.01:
                route_stats["high_risk_segments"] += 1
            route_coords.append((G.nodes[u]['y'], G.nodes[u]['x']))
        route_coords.append((G.nodes[dest_node]['y'], G.nodes[dest_node]['x']))
        route_stats["distance_km"] = dist_m / 1000

        st.success("Optimal route path computed successfully. High-risk zones avoided.")
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Total Route Distance", f"{route_stats['distance_km']:.2f} km")
        with col2:
            st.metric("High-Risk Segments", route_stats['high_risk_segments'],
                      delta="None Detected" if route_stats['high_risk_segments'] == 0 else "Elevated Risk",
                      delta_color="inverse")
        with col3:
            st.metric("Hazards Avoided", route_stats["blocked_zones_avoided"],
                      delta="Normal" if route_stats["blocked_zones_avoided"] == 0
                            else f"{route_stats['blocked_zones_avoided']} Rerouted",
                      delta_color="normal")
        with col4:
            if "Evacuation" in routing_mode:
                st.metric("Assigned Safe Center", evac_name,
                          delta=f"{evac_crow_dist:.1f} km Euclidean", delta_color="normal")
            else:
                st.metric("Target Destination", end_loc)
else:
    st.info(
        "Select your journey endpoints in the sidebar and click 'Compute Route'. "
        "Dynamic sector flooding can be configured from the incident reporting section."
    )


# ─────────────────────────────────────────
# MAP DISPLAY TOGGLES & VISIBILITY CONTROLS
# ─────────────────────────────────────────
st.markdown("---")
st.markdown("""
<div style="display: flex; align-items: center; justify-content: space-between; margin-bottom: 12px;">
    <div style="font-size: 1.15rem; font-weight: 700; color: #f1f5f9; display: flex; align-items: center; gap: 8px;">
        <span>🗺️</span> Map Display Layers &amp; Overlays
    </div>
    <div style="font-size: 0.80rem; color: #94a3b8; font-family: 'JetBrains Mono', monospace;">
        Toggle active layers on the geospatial map
    </div>
</div>
""", unsafe_allow_html=True)

col_t1, col_t2, col_t3, col_t4, col_t5 = st.columns(5)
with col_t1:
    show_risk_zones = st.toggle("🌊 Risk Zones", value=True, key="toggle_risk_zones",
                                help="AI predicted flood inundation probability choropleth heatmap")
with col_t2:
    show_landmarks = st.toggle("📍 Node Points", value=True, key="toggle_landmarks",
                               help="Landmark points, numbered road network blocks, and localities")
with col_t3:
    show_institutions = st.toggle("🏫 Institutions", value=True, key="toggle_institutions",
                                  help="Colleges, universities, schools, and emergency safe shelters")
with col_t4:
    show_blocked = st.toggle("⛔ Hazard Closures", value=True, key="toggle_blocked",
                             help="Real-time user-reported flooded sectors and impassable roads")
with col_t5:
    show_hotspots = st.toggle("⚠️ Historical Hotspots", value=False, key="toggle_hotspots",
                              help="Recorded flood hotspot locations across Chennai")


# ─────────────────────────────────────────
# MAP VISUALISATION
# ─────────────────────────────────────────
m = folium.Map(location=[13.05, 80.23], zoom_start=11, tiles=None)

# Add map base tile layers (OpenStreetMap set as default)
folium.TileLayer('OpenStreetMap', name='OpenStreetMap', control=True, show=True).add_to(m)
folium.TileLayer('CartoDB positron', name='Clean Light', control=True, show=False).add_to(m)
folium.TileLayer('CartoDB dark_matter', name='Dark Mode', control=True, show=False).add_to(m)

# 1. AI Flood Risk Zones Overlay
if show_risk_zones:
    grid_map              = grid_gdf[grid_gdf['flood_prob'] > 0.15].copy()
    grid_map['geometry']  = grid_map.geometry.simplify(0.0008)
    grid_map['risk_pct']  = (grid_map['flood_prob'] * 100).round(1).astype(str) + "%"
    grid_map['area_name'] = grid_map['block_no'].apply(
        lambda b: ", ".join(block_to_locs.get(b, [f"Grid Block {b}"]))
    )
    grid_map['cell_status'] = grid_map['block_no'].apply(
        lambda b: "TEMPORARILY FLOODED (User Reported)"
        if b in st.session_state.blocked_blocks else "Normal AI Assessment"
    )

    folium.Choropleth(
        geo_data=json.loads(grid_map[['cell_id', 'geometry']].to_json()),
        data=grid_map, columns=['cell_id', 'flood_prob'],
        key_on='feature.properties.cell_id', fill_color='YlOrRd',
        fill_opacity=0.40, line_opacity=0.10, legend_name='Predicted Flood Probability',
        name='🌊 AI Flood Risk Zones', show=True
    ).add_to(m)

    folium.GeoJson(
        json.loads(grid_map[['cell_id', 'block_no', 'area_name', 'risk_pct', 'cell_status', 'geometry']].to_json()),
        style_function=lambda x: {'fillColor': 'transparent', 'color': 'transparent', 'weight': 0},
        tooltip=folium.GeoJsonTooltip(
            fields=['block_no', 'area_name', 'risk_pct', 'cell_status'],
            aliases=['Point / Block #:', 'Area / Landmark:', 'Predicted Flood Risk:', 'Status:']
        ),
        control=False
    ).add_to(m)

# 2. Blocked / Hazard Sectors Overlay
if show_blocked and st.session_state.blocked_blocks:
    blocked_cells = grid_gdf[grid_gdf['block_no'].isin(st.session_state.blocked_blocks)].copy()
    if not blocked_cells.empty:
        blocked_fg = folium.FeatureGroup(name="⛔ Reported Hazard Blocks", show=True)
        folium.GeoJson(
            json.loads(blocked_cells[['cell_id', 'block_no', 'geometry']].to_json()),
            style_function=lambda x: {
                'fillColor': '#FF0000', 'color': '#7F1D1D',
                'fillOpacity': 0.75, 'weight': 3.5
            },
            tooltip=folium.GeoJsonTooltip(
                fields=['block_no'],
                aliases=['TEMPORARILY FLOODED BLOCK #:']
            )
        ).add_to(blocked_fg)

        for _, row in blocked_cells.iterrows():
            b_id      = int(row['block_no'])
            c_lat     = float(row['centroid_latitude'])
            c_lon     = float(row['centroid_longitude'])
            loc_label = f"<br>({block_to_locs[b_id][0]})" if b_id in block_to_locs else ""
            folium.Marker(
                [c_lat, c_lon],
                icon=folium.DivIcon(html=f'''
                    <div style="background-color: #DC2626; color: white; padding: 4px 8px; border-radius: 6px; font-weight: 800; font-size: 11px; border: 2px solid #FFFFFF; box-shadow: 0 2px 8px rgba(0,0,0,0.6); text-align: center; white-space: nowrap; transform: translate(-50%, -50%);">
                        BLOCKED<br>#{b_id}{loc_label}
                    </div>
                '''),
                tooltip=f"Real-Time Flooded Square #{b_id} (All Roads Impassable)"
            ).add_to(blocked_fg)
        blocked_fg.add_to(m)

# 3. Node Points / Landmarks Overlay
if show_landmarks:
    landmark_fg = folium.FeatureGroup(name="📍 Landmark Node Points", show=True)
    for loc_name, (lat, lon) in LOCATIONS.items():
        b_num = loc_to_block.get(loc_name)
        if b_num and b_num not in st.session_state.blocked_blocks:
            folium.Marker(
                [lat, lon],
                icon=folium.DivIcon(html=f'''
                    <div style="background: rgba(15, 23, 42, 0.85); color: #F8FAFC; padding: 2px 5px; border-radius: 4px; font-size: 10px; font-weight: 600; border: 1px solid #475569; text-align: center; white-space: nowrap; transform: translate(-50%, -50%);">
                        {loc_name} <span style="color: #38BDF8; font-weight: 700;">#{b_num}</span>
                    </div>
                '''),
                tooltip=f"{loc_name} (Point / Block #{b_num})"
            ).add_to(landmark_fg)
    landmark_fg.add_to(m)

# 4. Institutions & Shelters Overlay
if show_institutions and all_institutions_gdf is not None and not all_institutions_gdf.empty:
    institutions_fg = folium.FeatureGroup(name="🏫 Key Institutions & Shelters", show=True)
    for _, row in all_institutions_gdf.iterrows():
        lat = float(row.geometry.y)
        lon = float(row.geometry.x)
        b_num = int(row.get('block_no', -1))
        f_prob = float(row.get('flood_prob', 0.0))
        is_blocked = b_num in st.session_state.blocked_blocks
        is_safe = (f_prob <= 0.05) and not is_blocked

        color = "#10b981" if is_safe else ("#ef4444" if is_blocked else "#f59e0b")
        status_txt = "Designated Safe Shelter" if is_safe else ("Flooded Sector" if is_blocked else f"Risk: {f_prob*100:.1f}%")
        
        folium.CircleMarker(
            location=[lat, lon],
            radius=6,
            color=color,
            fill=True,
            fill_color=color,
            fill_opacity=0.85,
            weight=1.5,
            tooltip=f"🏫 {row['name']} ({str(row.get('amenity', 'Facility')).title()}) — Block #{b_num} — {status_txt}"
        ).add_to(institutions_fg)
    institutions_fg.add_to(m)

# 5. Historical Flood Hotspots Overlay
if show_hotspots and hotspots_gdf is not None and not hotspots_gdf.empty:
    hotspots_fg = folium.FeatureGroup(name="⚠️ Historical Flood Hotspots", show=True)
    valid_hotspots = hotspots_gdf[['Name', 'geometry']].dropna(subset=['geometry'])
    folium.GeoJson(
        json.loads(valid_hotspots.to_json()),
        marker=folium.CircleMarker(
            radius=3.5, color='#d97706', fill=True, fill_color='#fbbf24',
            fill_opacity=0.65, weight=1
        ),
        tooltip=folium.GeoJsonTooltip(fields=['Name'], aliases=['Recorded Hotspot:'])
    ).add_to(hotspots_fg)
    hotspots_fg.add_to(m)

# 6. Computed Route & Waypoints Overlay
if route_coords:
    route_fg = folium.FeatureGroup(name="🛣️ Navigation Route", show=True)
    route_color = "green" if (route_type == "Lower-Risk Path" or "Evacuation" in routing_mode) else "blue"
    folium.PolyLine(route_coords, color=route_color, weight=6, opacity=0.85).add_to(route_fg)
    if start_loc in LOCATIONS:
        folium.Marker(
            LOCATIONS[start_loc],
            tooltip=f"Start: {start_loc} (Block #{loc_to_block.get(start_loc)})",
            icon=folium.Icon(color="gray", icon="play")
        ).add_to(route_fg)
    if actual_end_lat is not None:
        icon_color  = "green" if "Evacuation" in routing_mode else "red"
        icon_type   = "info-sign" if "Evacuation" in routing_mode else "flag"
        tooltip_txt = (
            f"Evacuation: {evac_name}"
            if "Evacuation" in routing_mode
            else f"Destination: {end_loc} (Block #{loc_to_block.get(end_loc)})"
        )
        folium.Marker([actual_end_lat, actual_end_lon], tooltip=tooltip_txt,
                      icon=folium.Icon(color=icon_color, icon=icon_type)).add_to(route_fg)
    route_fg.add_to(m)

folium.LayerControl(position="topright", collapsed=True).add_to(m)
st_folium(m, width="100%", height=560, returned_objects=[])


# ─────────────────────────────────────────
# SECTION 2 — GEOSPATIAL & MODEL ANALYTICS
# ─────────────────────────────────────────
st.markdown("---")
st.markdown('<div class="section-heading">Model Diagnostics &amp; Environmental Analytics</div>', unsafe_allow_html=True)

if st.session_state.blocked_blocks:
    col_stat1, col_stat2, col_stat3 = st.columns(3)
    with col_stat1:
        st.metric("Impassable Grid Cells", len(st.session_state.blocked_blocks))
    with col_stat2:
        st.metric("Submerged Intersections", len(st.session_state.blocked_nodes))
    with col_stat3:
        st.metric("Penalized Road Segments", len(st.session_state.modified_edges))

CHART_BG  = "#091222"
AXIS_CLR  = "#1e293b"
GRID_CLR  = "#13213c"
LABEL_CLR = "#94a3b8"

plt.rcParams['font.sans-serif'] = 'DejaVu Sans'

col_graph1, col_graph2 = st.columns(2)

with col_graph1:
    st.markdown("**Feature Importance — Random Forest Model**")
    importances = pd.Series(rf_model.feature_importances_, index=FEATURES).sort_values(ascending=True)
    fig1, ax1   = plt.subplots(figsize=(8, 4.8), dpi=100)
    fig1.patch.set_facecolor(CHART_BG)
    ax1.set_facecolor(CHART_BG)
    bars = ax1.barh(importances.index, importances.values, color='#0284c7', edgecolor='#38bdf8', height=0.65, alpha=0.9)
    ax1.set_xlabel("Feature Relative Weight", color=LABEL_CLR, fontsize=9.5, labelpad=8)
    ax1.tick_params(colors=LABEL_CLR, labelsize=8.5)
    ax1.grid(axis='x', color=GRID_CLR, linestyle='--', linewidth=0.7, alpha=0.8)
    for spine in ax1.spines.values():
        spine.set_color(AXIS_CLR)
    fig1.tight_layout()
    st.pyplot(fig1)

with col_graph2:
    st.markdown("**Predicted Flood Probability Distribution**")
    fig2, ax2 = plt.subplots(figsize=(8, 4.8), dpi=100)
    fig2.patch.set_facecolor(CHART_BG)
    ax2.set_facecolor(CHART_BG)
    sns.histplot(grid_gdf['flood_prob'], bins=28, kde=True, color='#f43f5e', ax=ax2, 
                 line_kws={'linewidth': 2.2}, alpha=0.55)
    ax2.set_xlabel("Calculated Probability Score", color=LABEL_CLR, fontsize=9.5, labelpad=8)
    ax2.set_ylabel("Grid Cell Density", color=LABEL_CLR, fontsize=9.5, labelpad=8)
    ax2.tick_params(colors=LABEL_CLR, labelsize=8.5)
    ax2.grid(color=GRID_CLR, linestyle='--', linewidth=0.7, alpha=0.8)
    for spine in ax2.spines.values():
        spine.set_color(AXIS_CLR)
    fig2.tight_layout()
    st.pyplot(fig2)

st.markdown("<br>", unsafe_allow_html=True)
col_graph3, col_graph4 = st.columns(2)

with col_graph3:
    st.markdown("**Water Proximity vs. Inundation Probability**")
    fig3, ax3 = plt.subplots(figsize=(8, 4.8), dpi=100)
    fig3.patch.set_facecolor(CHART_BG)
    ax3.set_facecolor(CHART_BG)
    ax3.scatter(grid_gdf['distance_to_water_m'], grid_gdf['flood_prob'], 
                alpha=0.35, color="#06b6d4", s=22, edgecolors='none')
    ax3.set_xlabel("Distance to Water Body (meters)", color=LABEL_CLR, fontsize=9.5, labelpad=8)
    ax3.set_ylabel("Predicted Risk", color=LABEL_CLR, fontsize=9.5, labelpad=8)
    ax3.tick_params(colors=LABEL_CLR, labelsize=8.5)
    ax3.grid(color=GRID_CLR, linestyle='--', linewidth=0.7, alpha=0.8)
    for spine in ax3.spines.values():
        spine.set_color(AXIS_CLR)
    fig3.tight_layout()
    st.pyplot(fig3)

with col_graph4:
    st.markdown("**Mean Elevation vs. Inundation Probability**")
    fig4, ax4 = plt.subplots(figsize=(8, 4.8), dpi=100)
    fig4.patch.set_facecolor(CHART_BG)
    ax4.set_facecolor(CHART_BG)
    ax4.scatter(grid_gdf['elevation_mean'], grid_gdf['flood_prob'], 
                alpha=0.35, color="#f59e0b", s=22, edgecolors='none')
    ax4.set_xlabel("Mean Surface Elevation (meters)", color=LABEL_CLR, fontsize=9.5, labelpad=8)
    ax4.set_ylabel("Predicted Risk", color=LABEL_CLR, fontsize=9.5, labelpad=8)
    ax4.tick_params(colors=LABEL_CLR, labelsize=8.5)
    ax4.grid(color=GRID_CLR, linestyle='--', linewidth=0.7, alpha=0.8)
    for spine in ax4.spines.values():
        spine.set_color(AXIS_CLR)
    fig4.tight_layout()
    st.pyplot(fig4)