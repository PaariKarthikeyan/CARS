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

# ==========================================
# PAGE CONFIGURATION & STATE
# ==========================================
st.set_page_config(page_title="CARS: Chennai Accessibility & Risk System", layout="wide")
st.title("CARS: Chennai Accessibility & Risk System")

# Initialize session state for real-time flood reports by block numbers
if 'blocked_blocks' not in st.session_state:
    st.session_state.blocked_blocks = set()

if 'modified_edges' not in st.session_state:
    st.session_state.modified_edges = []

if 'blocked_nodes' not in st.session_state:
    st.session_state.blocked_nodes = set()

# ==========================================
# CACHED DATA LOADING
# ==========================================
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
    
    # Store baseline length and risk attributes for fast in-place restoration
    for u, v, k, data in G.edges(keys=True, data=True):
        length = float(data.get('length', 100.0))
        prob = float(data.get('flood_prob', 0.0))
        
        if prob > 0.01:
            multiplier = 1.0 + (prob * 50.0)
            base_risk_len = length * multiplier
        else:
            base_risk_len = length
            
        data['base_risk_adjusted_length'] = base_risk_len
        data['risk_adjusted_length'] = base_risk_len
        data['length'] = length
        data['flood_prob'] = prob
        data['base_flood_prob'] = prob
        data['is_user_blocked'] = False
        
    return G

@st.cache_resource
def get_graph_spatial_index(_G):
    """Precomputes coordinate arrays for fast spatial indexing of road nodes."""
    node_ids = np.array(list(_G.nodes))
    xs = np.array([_G.nodes[n]['x'] for n in node_ids], dtype=np.float64)
    ys = np.array([_G.nodes[n]['y'] for n in node_ids], dtype=np.float64)
    return node_ids, xs, ys

@st.cache_data
def load_safe_institutions(_grid_gdf):
    """Loads cleaned offline institutions and filters for zero-risk zones."""
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

# Load models and spatial layers
with st.spinner("Loading geospatial network, AI models, and Safe Institutions..."):
    rf_model = load_model()
    grid_gdf = load_grid()
    hotspots_gdf = load_hotspots()
    G_base = load_road_graph()
    node_ids, node_xs, node_ys = get_graph_spatial_index(G_base)
    
    features = [
        'elevation_mean', 'slope_mean', 'rainfall_24h', 'rainfall_72h', 
        'rainfall_7day', 'distance_to_water_m', 'distance_to_drain_m', 
        'drainage_density', 'built_up_percentage', 'water_percentage'
    ]
    X = grid_gdf[features].fillna(0)
    grid_gdf['flood_prob'] = rf_model.predict_proba(X)[:, 1]
    safe_institutions_gdf = load_safe_institutions(grid_gdf)

# ==========================================
# SIDEBAR: LOCATIONS & LANDMARK MAPPING
# ==========================================
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
    """Maps all landmark locations to their corresponding 1-indexed block numbers."""
    loc_to_block = {}
    block_to_locs = {}
    for name, (lat, lon) in LOCATIONS.items():
        pt = Point(lon, lat)
        matches = _grid_gdf[_grid_gdf.geometry.contains(pt)]
        if not matches.empty:
            b_num = int(matches.iloc[0]['block_no'])
        else:
            # Nearest cell fallback
            dists = ((_grid_gdf['centroid_latitude'] - lat)**2 + (_grid_gdf['centroid_longitude'] - lon)**2)
            b_num = int(_grid_gdf.loc[dists.idxmin(), 'block_no'])
        loc_to_block[name] = b_num
        if b_num not in block_to_locs:
            block_to_locs[b_num] = []
        block_to_locs[b_num].append(name)
    return loc_to_block, block_to_locs

loc_to_block, block_to_locs = get_landmark_block_map(grid_gdf)

# Helper function to parse user block numbers
def parse_block_numbers(input_str, max_block=4860):
    """
    Parses single or multiple block numbers from user input string.
    Supports formats:
      - Single: "2955"
      - Comma/space separated: "2955, 2700, 3352"
      - Range: "2950-2953"
      - Prefixed: "cell_02954", "#2955", "Block 2955"
    """
    if not input_str or not input_str.strip():
        return set(), []
    
    valid_blocks = set()
    invalid_tokens = []
    cleaned_str = input_str

    # 1. Extract ranges like 2950-2953
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

    # 2. Extract cell_XXXXX pattern
    for m in re.finditer(r'cell_?(\d+)', cleaned_str, re.IGNORECASE):
        b = int(m.group(1)) + 1
        if 1 <= b <= max_block:
            valid_blocks.add(b)
        else:
            invalid_tokens.append(m.group(0))
        cleaned_str = cleaned_str.replace(m.group(0), ' ')

    # 3. Extract standalone numbers
    for m in re.finditer(r'#?(\d+)', cleaned_str):
        b = int(m.group(1))
        if 1 <= b <= max_block:
            valid_blocks.add(b)
        else:
            invalid_tokens.append(m.group(0))

    return valid_blocks, invalid_tokens

# ==========================================
# REAL-TIME GRAPH MODIFICATION
# ==========================================
def apply_blocked_blocks_to_graph(G, blocked_blocks, grid_gdf, node_ids, xs, ys):
    """
    Applies real-time infinite penalties to all road intersections and edges
    falling inside the user-specified block numbers. Restores previous edges cleanly.
    """
    # Restore any previously modified edges
    if st.session_state.modified_edges:
        for u, v, k in st.session_state.modified_edges:
            if G.has_edge(u, v, k):
                G[u][v][k]['risk_adjusted_length'] = G[u][v][k]['base_risk_adjusted_length']
                G[u][v][k]['flood_prob'] = G[u][v][k]['base_flood_prob']
                G[u][v][k]['is_user_blocked'] = False
        st.session_state.modified_edges = []

    st.session_state.blocked_nodes = set()
    if not blocked_blocks:
        return

    modified = []
    blocked_nodes_set = set()

    for block_num in blocked_blocks:
        cell_idx = block_num - 1
        if 0 <= cell_idx < len(grid_gdf):
            cell = grid_gdf.iloc[cell_idx]
            minx, miny, maxx, maxy = cell.geometry.bounds
            mask = (xs >= minx) & (xs <= maxx) & (ys >= miny) & (ys <= maxy)
            matched_nodes = node_ids[mask]
            
            # Fallback if cell has no road nodes strictly inside
            if len(matched_nodes) == 0:
                c_lat = float(cell['centroid_latitude'])
                c_lon = float(cell['centroid_longitude'])
                nearest = ox.distance.nearest_nodes(G, c_lon, c_lat)
                matched_nodes = [nearest]
                
            for n in matched_nodes:
                blocked_nodes_set.add(n)
                for u, v, k, data in G.edges(n, data=True, keys=True):
                    data['risk_adjusted_length'] = 99999999.0
                    data['flood_prob'] = 1.0
                    data['is_user_blocked'] = True
                    modified.append((u, v, k))
                for u, v, k, data in G.in_edges(n, data=True, keys=True):
                    data['risk_adjusted_length'] = 99999999.0
                    data['flood_prob'] = 1.0
                    data['is_user_blocked'] = True
                    modified.append((u, v, k))

    st.session_state.modified_edges = modified
    st.session_state.blocked_nodes = blocked_nodes_set

# Apply current real-time block reports to the road graph
apply_blocked_blocks_to_graph(G_base, st.session_state.blocked_blocks, grid_gdf, node_ids, node_xs, node_ys)
G = G_base

# ==========================================
# GEOGRAPHIC UTILITIES & ROUTING
# ==========================================
def haversine_dist_km(lat1, lon1, lat2, lon2):
    R = 6371.0
    dlat = math.radians(lat2 - lat1)
    dlon = math.radians(lon2 - lon1)
    a = (math.sin(dlat / 2) ** 2 + 
         math.cos(math.radians(lat1)) * math.cos(math.radians(lat2)) * math.sin(dlon / 2) ** 2)
    return R * 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))

def find_best_evacuation_center(G_active, orig_node, start_lat, start_lon, max_radius_km=7.0):
    if safe_institutions_gdf is None or safe_institutions_gdf.empty:
        candidates = grid_gdf[grid_gdf['flood_prob'] <= 0.01].copy().reset_index(drop=True)
        if candidates.empty:
            candidates = grid_gdf[grid_gdf['flood_prob'] == grid_gdf['flood_prob'].min()].copy().reset_index(drop=True)
        candidates['target_lat'] = candidates.geometry.centroid.y
        candidates['target_lon'] = candidates.geometry.centroid.x
        candidates['display_name'] = "Designated Safe Zone Cell"
    else:
        # Exclude safe centers that fall inside real-time reported flooded blocks
        candidates = safe_institutions_gdf.copy()
        if 'block_no' in candidates.columns and st.session_state.blocked_blocks:
            candidates = candidates[~candidates['block_no'].isin(st.session_state.blocked_blocks)].copy().reset_index(drop=True)
        else:
            candidates = candidates.reset_index(drop=True)
            
        candidates['target_lat'] = candidates.geometry.y
        candidates['target_lon'] = candidates.geometry.x
        candidates['display_name'] = candidates['name']

    if candidates.empty:
        return None

    candidates['dist_km'] = candidates.apply(
        lambda r: haversine_dist_km(start_lat, start_lon, r['target_lat'], r['target_lon']),
        axis=1
    )

    lengths, paths = nx.single_source_dijkstra(G_active, orig_node, weight='risk_adjusted_length')
    nearby_candidates = candidates[candidates['dist_km'] <= max_radius_km].copy()
    
    best_center = None
    lowest_cost = float('inf')

    # Priority 1: Centers within 7km
    for _, row in nearby_candidates.iterrows():
        cand_lat = float(row['target_lat'])
        cand_lon = float(row['target_lon'])
        dest_node = ox.distance.nearest_nodes(G_active, cand_lon, cand_lat)

        if dest_node in lengths and lengths[dest_node] < 50000000.0 and lengths[dest_node] < lowest_cost:
            lowest_cost = lengths[dest_node]
            best_center = {
                "lat": cand_lat, "lon": cand_lon, "name": str(row['display_name']),
                "prob": float(row.get('flood_prob', 0.0)), "dest_node": dest_node,
                "route": paths[dest_node], "crow_dist_km": float(row['dist_km']),
                "fallback_used": False
            }

    # Priority 2: Closest reachable safe center anywhere if 7km radius is blocked
    if best_center is None:
        for _, row in candidates.iterrows():
            cand_lat = float(row['target_lat'])
            cand_lon = float(row['target_lon'])
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

# ==========================================
# SIDEBAR: NAVIGATION & REAL-TIME FLOOD INPUT
# ==========================================
st.sidebar.header("Navigation Panel")

with st.sidebar.form("routing_form"):
    start_loc = st.selectbox("Start Location", list(LOCATIONS.keys()), index=list(LOCATIONS.keys()).index("T Nagar"))
    routing_mode = st.radio("Routing Goal", ["Navigate to Specific Destination", "Find Nearest Safe Evacuation Center (Within 5-7 km)"])
    end_loc = st.selectbox("Destination (Used for Specific Destination Mode)", list(LOCATIONS.keys()), index=list(LOCATIONS.keys()).index("Velachery"))
    route_type = st.radio("Route Preference", ["Lower-Risk Route (Safer)", "Shortest Distance"])
    calculate_btn = st.form_submit_button("Calculate Route")

st.sidebar.markdown("---")
st.sidebar.subheader("🚨 Real-Time Flood Reporting")
st.sidebar.caption("Type point/block number(s) to immediately mark them **flooded & red**. All traversing roads will be declared unsafe.")

# Fast block reporting form
with st.sidebar.form("block_report_form"):
    typed_blocks = st.text_input(
        "Enter Block / Point Number(s):",
        placeholder="e.g. 2955 or 2955, 2700 or 2950-2953",
        help="Type single or multiple block numbers (1 to 4860). Ranges (e.g. 2950-2953) are also accepted."
    )
    add_blocks_btn = st.form_submit_button("🔴 Mark Block(s) as Flooded")

if add_blocks_btn and typed_blocks:
    parsed_blocks, invalid_tokens = parse_block_numbers(typed_blocks, max_block=len(grid_gdf))
    if parsed_blocks:
        st.session_state.blocked_blocks.update(parsed_blocks)
        st.sidebar.success(f"Reported {len(parsed_blocks)} block(s) as flooded!")
        st.rerun()
    if invalid_tokens:
        st.sidebar.warning(f"Invalid / out-of-bounds: {', '.join(invalid_tokens)}")

# Quick Landmark Flood dropdown
quick_landmark = st.sidebar.selectbox(
    "Quick-Select Landmark to Flood:", 
    ["-- Choose Landmark --"] + sorted(list(LOCATIONS.keys()))
)
if st.sidebar.button("➕ Mark Selected Landmark Flooded"):
    if quick_landmark != "-- Choose Landmark --":
        target_b = loc_to_block[quick_landmark]
        st.session_state.blocked_blocks.add(target_b)
        st.sidebar.success(f"Marked {quick_landmark} (Block #{target_b}) as flooded!")
        st.rerun()

# Manage active reports in sidebar
if st.session_state.blocked_blocks:
    st.sidebar.markdown(f"**Temporarily Flooded Blocks ({len(st.session_state.blocked_blocks)}):**")
    for b in sorted(list(st.session_state.blocked_blocks)):
        loc_desc = f" ({block_to_locs[b][0]})" if b in block_to_locs else ""
        col_name, col_btn = st.sidebar.columns([3, 1])
        col_name.markdown(f"<span style='color:#DC2626; font-weight:700;'>🚨 Block #{b}{loc_desc}</span>", unsafe_allow_html=True)
        if col_btn.button("✖", key=f"del_sidebar_{b}", help=f"Remove Block #{b}"):
            st.session_state.blocked_blocks.remove(b)
            st.rerun()

    if st.sidebar.button("🧹 Clear All Real-Time Flood Reports"):
        st.session_state.blocked_blocks = set()
        st.rerun()
else:
    st.sidebar.info("No active flood reports. All grid blocks normal.")

# ==========================================
# REAL-TIME EMERGENCY STATUS BANNER
# ==========================================
if st.session_state.blocked_blocks:
    active_str = ", ".join([f"#{b}" + (f" ({block_to_locs[b][0]})" if b in block_to_locs else "") for b in sorted(list(st.session_state.blocked_blocks))])
    st.error(
        f"🚨 **LIVE FLOOD HAZARD ACTIVE**: {len(st.session_state.blocked_blocks)} block(s) marked flooded: **{active_str}**.\n\n"
        f"These squares are highlighted in **RED** on the map. All roads through them are closed, and routing will automatically find safe alternate corridors."
    )

# Quick Directory / Point Numbers Reference
with st.expander("📍 Chennai Landmark to Point / Block Number Directory (Quick Reference)", expanded=False):
    st.markdown("Each square on the map represents a 550m x 550m city sector. Use this guide to find block numbers for key Chennai neighborhoods:")
    dir_col1, dir_col2, dir_col3 = st.columns(3)
    loc_items = sorted(LOCATIONS.keys())
    split_size = math.ceil(len(loc_items) / 3)
    
    for idx, col in enumerate([dir_col1, dir_col2, dir_col3]):
        with col:
            tbl = "| Landmark | Point / Block # |\n| :--- | :--- |\n"
            for loc in loc_items[idx * split_size : (idx + 1) * split_size]:
                b_num = loc_to_block.get(loc, 'N/A')
                status_icon = "🔴" if b_num in st.session_state.blocked_blocks else "🟩"
                tbl += f"| {status_icon} {loc} | **#{b_num}** |\n"
            st.markdown(tbl)

# ==========================================
# SECTION 1: ROUTING LOGIC EXECUTION
# ==========================================
st.header("1. Interactive Map & Live Routing Results")

# Check if selected Start or Destination is inside a flooded block
start_b = loc_to_block.get(start_loc)
dest_b = loc_to_block.get(end_loc)

if start_b in st.session_state.blocked_blocks:
    st.error(f"⚠️ **HAZARD WARNING**: Your starting location **{start_loc}** is located directly inside **Flooded Block #{start_b}**! Roads are submerged. Evacuate to higher ground immediately.")
if "Evacuation" not in routing_mode and dest_b in st.session_state.blocked_blocks:
    st.error(f"⚠️ **HAZARD WARNING**: Your chosen destination **{end_loc}** is located directly inside **Flooded Block #{dest_b}**! The destination area is inundated and impassable.")

route_coords = []
route_stats = {"distance_km": 0.0, "high_risk_segments": 0, "blocked_zones_avoided": len(st.session_state.blocked_blocks)}
actual_end_lat, actual_end_lon = None, None
evac_name = None
evac_crow_dist = None

if calculate_btn:
    start_lat, start_lon = LOCATIONS[start_loc]
    orig_node = ox.distance.nearest_nodes(G, start_lon, start_lat)

    if "Evacuation Center" in routing_mode:
        st.info("Locating verified safe evacuation zones and computing path avoiding real-time flood blockages...")
        best_evac = find_best_evacuation_center(G, orig_node, start_lat, start_lon, max_radius_km=7.0)

        if best_evac:
            if best_evac["fallback_used"]:
                st.warning(f"No safe route exists within 7km. Routing to closest accessible safe haven: {best_evac['name']}.")
            actual_end_lat, actual_end_lon = best_evac["lat"], best_evac["lon"]
            evac_name, evac_crow_dist = best_evac["name"], best_evac["crow_dist_km"]
            route, dest_node = best_evac["route"], best_evac["dest_node"]
        else:
            route = None
            st.error("Severe Network Isolation: No safe evacuation centers are reachable due to real-time flood blockages.")
    else:
        actual_end_lat, actual_end_lon = LOCATIONS[end_loc]
        dest_node = ox.distance.nearest_nodes(G, actual_end_lon, actual_end_lat)
        weight = 'risk_adjusted_length' if route_type == "Lower-Risk Route (Safer)" else 'length'
        
        try:
            route = nx.shortest_path(G, orig_node, dest_node, weight=weight)
            # Verify if path was forced across blocked edges
            total_cost = nx.path_weight(G, route, weight=weight)
            if total_cost >= 50000000.0:
                route = None
                st.error(f"🚫 Path Blocked: All corridors to {end_loc} are completely severed by reported flood blockages in Block(s): {sorted(list(st.session_state.blocked_blocks))}.")
        except nx.NetworkXNoPath:
            route = None
            st.error(f"Cannot find a path to {end_loc}. The destination is entirely cut off by real-time flood blockages.")

    # Calculate final route statistics
    if actual_end_lat is not None and start_lat == actual_end_lat and start_lon == actual_end_lon:
        st.warning("Start location and destination are identical.")
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

        st.success("Optimal route calculated! The navigation engine has safely navigated around all flooded squares.")
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Total Road Distance", f"{route_stats['distance_km']:.2f} km")
        with col2:
            st.metric("Risky Segments Crossed", route_stats['high_risk_segments'],
                      delta="Avoided!" if route_stats['high_risk_segments'] == 0 else "High Risk", delta_color="inverse")
        with col3:
            st.metric("Flooded Blocks Avoided", route_stats["blocked_zones_avoided"],
                      delta="All Clear" if route_stats["blocked_zones_avoided"] == 0 else f"{route_stats['blocked_zones_avoided']} Active Blocked", delta_color="normal")
        with col4:
            if "Evacuation Center" in routing_mode:
                st.metric("Safe Haven Destination", f"{evac_name}", delta=f"{evac_crow_dist:.1f} km direct", delta_color="normal")
            else:
                st.metric("Destination", f"{end_loc}")
else:
    st.info("Select your starting point in the sidebar and click 'Calculate Route'. Type block number(s) in the sidebar to report live flood blockages.")

# ==========================================
# MAP GENERATION & VISUALIZATION
# ==========================================
m = folium.Map(location=[13.05, 80.23], zoom_start=11, tiles="OpenStreetMap")

# 1. AI Flood Prediction Grid Layer
grid_map = grid_gdf[grid_gdf['flood_prob'] > 0.15].copy()
grid_map['geometry'] = grid_map.geometry.simplify(0.0008)
grid_map['risk_pct'] = (grid_map['flood_prob'] * 100).round(1).astype(str) + "%"
grid_map['area_name'] = grid_map['block_no'].apply(
    lambda b: ", ".join(block_to_locs.get(b, [f"Grid Block #{b}"]))
)
grid_map['cell_status'] = grid_map['block_no'].apply(
    lambda b: "🚨 TEMPORARILY FLOODED (User Reported)" if b in st.session_state.blocked_blocks else "Normal AI Assessment"
)

folium.Choropleth(
    geo_data=json.loads(grid_map[['cell_id', 'geometry']].to_json()),
    data=grid_map, columns=['cell_id', 'flood_prob'],
    key_on='feature.properties.cell_id', fill_color='YlOrRd',
    fill_opacity=0.40, line_opacity=0.1, legend_name='Predicted Flood Probability'
).add_to(m)

# Tooltip layer providing Point Numbers for every visible square
folium.GeoJson(
    json.loads(grid_map[['cell_id', 'block_no', 'area_name', 'risk_pct', 'cell_status', 'geometry']].to_json()),
    style_function=lambda x: {'fillColor': 'transparent', 'color': 'transparent', 'weight': 0},
    tooltip=folium.GeoJsonTooltip(
        fields=['block_no', 'area_name', 'risk_pct', 'cell_status'],
        aliases=['🏷️ Point / Block #:', '📍 Area / Landmark:', '🌊 Predicted Flood Risk:', '⚠️ Status:']
    )
).add_to(m)

# 2. Render Real-Time Blocked Blocks in Bright RED
if st.session_state.blocked_blocks:
    blocked_cells = grid_gdf[grid_gdf['block_no'].isin(st.session_state.blocked_blocks)].copy()
    if not blocked_cells.empty:
        # High-visibility red polygon fill with thick red border
        folium.GeoJson(
            json.loads(blocked_cells[['cell_id', 'block_no', 'geometry']].to_json()),
            style_function=lambda x: {
                'fillColor': '#FF0000',
                'color': '#7F1D1D',
                'fillOpacity': 0.75,
                'weight': 3.5
            },
            tooltip=folium.GeoJsonTooltip(
                fields=['block_no'],
                aliases=['🚨 TEMPORARILY FLOODED BLOCK #:']
            )
        ).add_to(m)
        
        # Prominent hazard labels at the centroid of each blocked square
        for _, row in blocked_cells.iterrows():
            b_id = int(row['block_no'])
            c_lat = float(row['centroid_latitude'])
            c_lon = float(row['centroid_longitude'])
            loc_label = f"<br>({block_to_locs[b_id][0]})" if b_id in block_to_locs else ""
            
            folium.Marker(
                [c_lat, c_lon],
                icon=folium.DivIcon(html=f'''
                    <div style="background-color: #DC2626; color: white; padding: 4px 8px; border-radius: 6px; font-weight: 800; font-size: 11px; border: 2px solid #FFFFFF; box-shadow: 0 2px 8px rgba(0,0,0,0.6); text-align: center; white-space: nowrap; transform: translate(-50%, -50%);">
                        🚨 BLOCKED<br>#{b_id}{loc_label}
                    </div>
                '''),
                tooltip=f"Real-Time Flooded Square #{b_id} (All Roads Impassable)"
            ).add_to(m)

# 3. Landmark Point Number Markers (so users can easily identify square numbers)
landmark_fg = folium.FeatureGroup(name="Landmark Point Numbers", show=True)
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

# 4. Render Calculated Route and Origin/Destination Waypoints
if route_coords:
    route_color = "green" if (route_type == "Lower-Risk Route (Safer)" or "Evacuation Center" in routing_mode) else "blue"
    folium.PolyLine(route_coords, color=route_color, weight=6, opacity=0.85).add_to(m)
    if start_loc in LOCATIONS:
        folium.Marker(LOCATIONS[start_loc], tooltip=f"Start: {start_loc} (Block #{loc_to_block.get(start_loc)})", icon=folium.Icon(color="gray", icon="play")).add_to(m)
    if actual_end_lat is not None:
        icon_color = "green" if "Evacuation Center" in routing_mode else "red"
        icon_type = "info-sign" if "Evacuation Center" in routing_mode else "flag"
        tooltip_txt = f"Evacuation: {evac_name}" if "Evacuation Center" in routing_mode else f"Destination: {end_loc} (Block #{loc_to_block.get(end_loc)})"
        folium.Marker([actual_end_lat, actual_end_lon], tooltip=tooltip_txt, icon=folium.Icon(color=icon_color, icon=icon_type)).add_to(m)

folium.LayerControl().add_to(m)

# Render map smoothly without slow click listener overhead
st_folium(m, width="100%", height=540, returned_objects=[])

# ==========================================
# SECTION 2: DATA VISUALIZATION GRAPHS
# ==========================================
st.markdown("---")
st.header("2. Geospatial & Model Data Analytics")

# Real-time incident analytics summary
if st.session_state.blocked_blocks:
    col_stat1, col_stat2, col_stat3 = st.columns(3)
    with col_stat1:
        st.metric("Active Flooded Squares", len(st.session_state.blocked_blocks))
    with col_stat2:
        st.metric("Submerged Road Intersections", len(st.session_state.blocked_nodes))
    with col_stat3:
        st.metric("Closed Road Edges", len(st.session_state.modified_edges))

col_graph1, col_graph2 = st.columns(2)
with col_graph1:
    st.subheader("Random Forest Feature Importance")
    importances = pd.Series(rf_model.feature_importances_, index=features).sort_values(ascending=True)
    fig1, ax1 = plt.subplots(figsize=(8, 4.8))
    importances.plot(kind='barh', color='teal', ax=ax1)
    ax1.set_xlabel("Importance Metric")
    st.pyplot(fig1)

with col_graph2:
    st.subheader("Flood Probability Distribution")
    fig2, ax2 = plt.subplots(figsize=(8, 4.8))
    sns.histplot(grid_gdf['flood_prob'], bins=25, kde=True, color='crimson', ax=ax2)
    ax2.set_xlabel("Predicted Probability")
    st.pyplot(fig2)

st.markdown("<br>", unsafe_allow_html=True)
col_graph3, col_graph4 = st.columns(2)
with col_graph3:
    st.subheader("Distance to Water vs. Flood Risk")
    fig3, ax3 = plt.subplots(figsize=(8, 4.8))
    sns.scatterplot(data=grid_gdf, x='distance_to_water_m', y='flood_prob', alpha=0.25, color="navy", ax=ax3)
    ax3.set_xlabel("Distance to Water (m)")
    st.pyplot(fig3)

with col_graph4:
    st.subheader("Elevation vs. Flood Risk")
    fig4, ax4 = plt.subplots(figsize=(8, 4.8))
    sns.scatterplot(data=grid_gdf, x='elevation_mean', y='flood_prob', alpha=0.25, color="darkorange", ax=ax4)
    ax4.set_xlabel("Mean Elevation (m)")
    st.pyplot(fig4)