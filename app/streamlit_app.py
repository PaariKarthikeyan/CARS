import streamlit as st
import folium
from streamlit_folium import st_folium
import geopandas as gpd
import pandas as pd
import osmnx as ox
import networkx as nx
import joblib
from pathlib import Path
import json
import math
import matplotlib.pyplot as plt
import seaborn as sns

# ==========================================
# PAGE CONFIGURATION & STATE
# ==========================================
st.set_page_config(page_title="CARS: Chennai Accessibility & Risk System", layout="wide")
st.title("CARS: Chennai Accessibility & Risk System")

# Initialize session state for real-time crowdsourced flood reports
if 'user_reports' not in st.session_state:
    st.session_state.user_reports = []

# ==========================================
# CACHED DATA LOADING
# ==========================================
@st.cache_resource
def load_model():
    return joblib.load(Path("models/random_forest.pkl"))

@st.cache_data
def load_grid():
    grid = gpd.read_file(Path("data/processed/grid/chennai_grid_with_features.geojson"), engine="pyogrio")
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
    
    # PROPORTIONAL PENALTY FIX: 
    # Instead of a flat addition, we multiply the length by the risk severity.
    # This ensures a 10m puddle isn't penalized as harshly as a 5km flooded highway.
    for u, v, k, data in G.edges(keys=True, data=True):
        length = float(data.get('length', 100.0))
        prob = float(data.get('flood_prob', 0.0))
        
        if prob > 0.01:
            # A road with 50% flood risk will 'feel' 26x longer to the algorithm
            multiplier = 1 + (prob * 50.0) 
            data['risk_adjusted_length'] = length * multiplier
        else:
            data['risk_adjusted_length'] = length
            
        data['length'] = length
        data['flood_prob'] = prob
        
    return G

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
        
    safe_pois = gpd.sjoin(pois, safe_cells[['cell_id', 'flood_prob', 'geometry']], how='inner')
    safe_pois = safe_pois.reset_index(drop=True)
    return safe_pois[['name', 'flood_prob', 'geometry']]

# Load models and spatial layers
with st.spinner("Loading geospatial network, AI models, and Safe Institutions..."):
    rf_model = load_model()
    grid_gdf = load_grid()
    hotspots_gdf = load_hotspots()
    G_base = load_road_graph()
    
    features = [
        'elevation_mean', 'slope_mean', 'rainfall_24h', 'rainfall_72h', 
        'rainfall_7day', 'distance_to_water_m', 'distance_to_drain_m', 
        'drainage_density', 'built_up_percentage', 'water_percentage'
    ]
    X = grid_gdf[features].fillna(0)
    grid_gdf['flood_prob'] = rf_model.predict_proba(X)[:, 1]
    safe_institutions_gdf = load_safe_institutions(grid_gdf)

# ==========================================
# REAL-TIME GRAPH MODIFICATION
# ==========================================
def apply_user_reports_to_graph(G, reports):
    """Creates a temporary graph where user-reported nodes are impassable."""
    if not reports:
        return G
    
    G_active = G.copy()
    for lat, lon in reports:
        # Snap the user's click to the nearest road intersection
        blocked_node = ox.distance.nearest_nodes(G_active, lon, lat)
        
        # Apply an infinite penalty to all roads connecting to this intersection
        for u, v, k, data in G_active.edges(blocked_node, data=True, keys=True):
            data['risk_adjusted_length'] = 99999999.0
        for u, v, k, data in G_active.in_edges(blocked_node, data=True, keys=True):
            data['risk_adjusted_length'] = 99999999.0
            
    return G_active

# Generate the active graph for this session
G = apply_user_reports_to_graph(G_base, st.session_state.user_reports)

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
        candidates = safe_institutions_gdf.copy().reset_index(drop=True)
        candidates['target_lat'] = candidates.geometry.y
        candidates['target_lon'] = candidates.geometry.x
        candidates['display_name'] = candidates['name']

    candidates['dist_km'] = candidates.apply(
        lambda r: haversine_dist_km(start_lat, start_lon, r['target_lat'], r['target_lon']),
        axis=1
    )

    # Calculate actual road distance to all reachable nodes avoiding user-reported blockages
    lengths, paths = nx.single_source_dijkstra(G_active, orig_node, weight='risk_adjusted_length')
    nearby_candidates = candidates[candidates['dist_km'] <= max_radius_km].copy()
    
    best_center = None
    lowest_cost = float('inf')

    # Priority 1: Centers within 7km
    for _, row in nearby_candidates.iterrows():
        cand_lat = float(row['target_lat'])
        cand_lon = float(row['target_lon'])
        dest_node = ox.distance.nearest_nodes(G_active, cand_lon, cand_lat)

        if dest_node in lengths and lengths[dest_node] < lowest_cost:
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

            if dest_node in lengths and lengths[dest_node] < lowest_cost:
                lowest_cost = lengths[dest_node]
                best_center = {
                    "lat": cand_lat, "lon": cand_lon, "name": str(row['display_name']),
                    "prob": float(row.get('flood_prob', 0.0)), "dest_node": dest_node,
                    "route": paths[dest_node], "crow_dist_km": float(row['dist_km']),
                    "fallback_used": True
                }

    return best_center

# ==========================================
# SIDEBAR: ROUTING FORM & LOCATIONS
# ==========================================
LOCATIONS = {
    "Adyar": (13.0012, 80.2565), "Alwarpet": (13.0335, 80.2524), "Ambattur": (13.1143, 80.1481),
    "Aminjikarai": (13.0732, 80.2210), "Anna Nagar": (13.0850, 80.2101), "Ashok Nagar": (13.0371, 80.2122),
    "Avadi": (13.1143, 80.1022), "Ayanavaram": (13.0970, 80.2366), "Besant Nagar": (13.0003, 80.2737),
    "Chetpet": (13.0714, 80.2417), "Choolaimedu": (13.0645, 80.2274), "Chromepet": (12.9516, 80.1406),
    "Egmore": (13.0732, 80.2609), "Ekkatuthangal": (13.0237, 80.2037), "Ennore": (13.2144, 80.3243),
    "George Town": (13.0978, 80.2862), "Gopalapuram": (13.0478, 80.2562), "Guindy": (13.0067, 80.2206),
    "K.K. Nagar": (13.0400, 80.1973), "Kilpauk": (13.0818, 80.2405), "Kodambakkam": (13.0521, 80.2255),
    "Korattur": (13.1118, 80.1834), "Korukkupet": (13.1172, 80.2785), "Kotturpuram": (13.0245, 80.2444),
    "Koyambedu": (13.0667, 80.1914), "Madhavaram": (13.1488, 80.2306), "Madipakkam": (12.9647, 80.1961),
    "Mandaveli": (13.0270, 80.2657), "Medavakkam": (12.9171, 80.1923), "Meenambakkam (Airport)": (12.9822, 80.1636),
    "Mogappair": (13.0838, 80.1747), "Mount Road (Anna Salai)": (13.0592, 80.2642), "Mylapore": (13.0368, 80.2676),
    "Nandanam": (13.0305, 80.2405), "Nanganallur": (12.9774, 80.1898), "Navallur": (12.8465, 80.2265),
    "Neelankarai": (12.9490, 80.2555), "Nungambakkam": (13.0569, 80.2425), "Pallavaram": (12.9675, 80.1491),
    "Pammal": (12.9749, 80.1338), "Parrys": (13.0886, 80.2860), "Pattabiram": (13.1218, 80.0596),
    "Perambur": (13.1112, 80.2458), "Perungudi": (12.9654, 80.2461), "Poonamallee": (13.0473, 80.0945),
    "Porur": (13.0354, 80.1562), "Purasawalkam": (13.0885, 80.2526), "Ramapuram": (13.0313, 80.1818),
    "Royapettah": (13.0559, 80.2635), "Royapuram": (13.1118, 80.2941), "Saidapet": (13.0213, 80.2231),
    "Santhome": (13.0323, 80.2783), "Selaiyur": (12.9099, 80.1432), "Shenoy Nagar": (13.0788, 80.2227),
    "Sholinganallur": (12.8996, 80.2279), "Siruseri": (12.8273, 80.2209), "T Nagar": (13.0418, 80.2341),
    "Tambaram": (12.9249, 80.1000), "Taramani": (12.9863, 80.2432), "Teynampet": (13.0405, 80.2503),
    "Thiruvanmiyur": (12.9863, 80.2589), "Thoraipakkam": (12.9404, 80.2376), "Tondiarpet": (13.1278, 80.2785),
    "Triplicane": (13.0588, 80.2759), "Vadapalani": (13.0500, 80.2121), "Valasaravakkam": (13.0396, 80.1738),
    "Velachery": (12.9815, 80.2180), "Vepery": (13.0841, 80.2631), "Villivakkam": (13.1075, 80.2078),
    "Virugambakkam": (13.0487, 80.1901), "Washermanpet": (13.1091, 80.2828), "West Mambalam": (13.0384, 80.2241)
}

st.sidebar.header("Navigation Panel")

with st.sidebar.form("routing_form"):
    start_loc = st.selectbox("Start Location", list(LOCATIONS.keys()), index=list(LOCATIONS.keys()).index("T Nagar"))
    routing_mode = st.radio("Routing Goal", ["Navigate to Specific Destination", "Find Nearest Safe Evacuation Center (Within 5-7 km)"])
    end_loc = st.selectbox("Destination (Used for Specific Destination Mode)", list(LOCATIONS.keys()), index=list(LOCATIONS.keys()).index("Velachery"))
    route_type = st.radio("Route Preference", ["Lower-Risk Route (Safer)", "Shortest Distance"])
    calculate_btn = st.form_submit_button("Calculate Route")

st.sidebar.markdown("---")
st.sidebar.subheader("Real-Time Incident Reporting")
st.sidebar.info("Click anywhere on the map to report a live flood. The routing engine will instantly block that road and reroute you.")
if st.sidebar.button("Clear All User Flood Reports"):
    st.session_state.user_reports = []
    st.rerun()

# ==========================================
# SECTION 1: ROUTING LOGIC EXECUTION
# ==========================================
st.header("1. Interactive Map & Live Routing Results")

route_coords = []
route_stats = {"distance_km": 0.0, "high_risk_segments": 0}
actual_end_lat, actual_end_lon = None, None
evac_name = None
evac_crow_dist = None

if calculate_btn:
    start_lat, start_lon = LOCATIONS[start_loc]
    orig_node = ox.distance.nearest_nodes(G, start_lon, start_lat)

    if "Evacuation Center" in routing_mode:
        st.info("Locating 0% risk safe zones and computing path avoiding user-reported blockages...")
        best_evac = find_best_evacuation_center(G, orig_node, start_lat, start_lon, max_radius_km=7.0)

        if best_evac:
            if best_evac["fallback_used"]:
                st.warning(f"No safe route exists within 7km. Routing to closest accessible safe zone: {best_evac['name']}.")
            actual_end_lat, actual_end_lon = best_evac["lat"], best_evac["lon"]
            evac_name, evac_crow_dist = best_evac["name"], best_evac["crow_dist_km"]
            route, dest_node = best_evac["route"], best_evac["dest_node"]
        else:
            route = None
            st.error("Severe Network Issue: No safe zones are accessible from this starting location.")
    else:
        actual_end_lat, actual_end_lon = LOCATIONS[end_loc]
        dest_node = ox.distance.nearest_nodes(G, actual_end_lon, actual_end_lat)
        weight = 'risk_adjusted_length' if route_type == "Lower-Risk Route (Safer)" else 'length'
        
        try:
            route = nx.shortest_path(G, orig_node, dest_node, weight=weight)
        except nx.NetworkXNoPath:
            route = None
            st.error(f"Cannot find a path to {end_loc}. The destination is entirely blocked by user reports or disconnected.")

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

        st.success("Optimal route calculated successfully!")
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Total Road Distance", f"{route_stats['distance_km']:.2f} km")
        with col2:
            st.metric("Risky Segments Crossed", route_stats['high_risk_segments'],
                      delta="Avoided!" if route_stats['high_risk_segments'] == 0 else "Avoid if Possible", delta_color="inverse")
        with col3:
            if "Evacuation Center" in routing_mode:
                st.metric("Safe Evacuation Center", f"{evac_name}", delta=f"{evac_crow_dist:.1f} km direct", delta_color="normal")
            else:
                st.metric("Destination", f"{end_loc}")
else:
    st.info("Select your starting point in the sidebar and click 'Calculate Route'. Click on the map below to report live blockages.")

# ==========================================
# MAP GENERATION & INTERACTION
# ==========================================
m = folium.Map(location=[13.05, 80.23], zoom_start=11, tiles="OpenStreetMap")

# Render AI Flood Prediction Grid
grid_map = grid_gdf[grid_gdf['flood_prob'] > 0.2].copy()
grid_map['geometry'] = grid_map.geometry.simplify(0.001)
folium.Choropleth(
    geo_data=json.loads(grid_map[['cell_id', 'geometry']].to_json()),
    data=grid_map, columns=['cell_id', 'flood_prob'],
    key_on='feature.properties.cell_id', fill_color='YlOrRd',
    fill_opacity=0.45, line_opacity=0.1, legend_name='Predicted Flood Probability'
).add_to(m)

# Render Path and Waypoints
if route_coords:
    route_color = "green" if (route_type == "Lower-Risk Route (Safer)" or "Evacuation Center" in routing_mode) else "red"
    folium.PolyLine(route_coords, color=route_color, weight=6, opacity=0.85).add_to(m)
    if start_loc in LOCATIONS:
        folium.Marker(LOCATIONS[start_loc], tooltip=f"Start: {start_loc}", icon=folium.Icon(color="gray", icon="play")).add_to(m)
    if actual_end_lat is not None:
        icon_color = "green" if "Evacuation Center" in routing_mode else "red"
        icon_type = "info-sign" if "Evacuation Center" in routing_mode else "flag"
        tooltip_txt = f"Evacuation: {evac_name}" if "Evacuation Center" in routing_mode else f"Destination: {end_loc}"
        folium.Marker([actual_end_lat, actual_end_lon], tooltip=tooltip_txt, icon=folium.Icon(color=icon_color, icon=icon_type)).add_to(m)

# Render User-Reported Real-Time Blockages
for lat, lon in st.session_state.user_reports:
    folium.Marker(
        [lat, lon], 
        icon=folium.Icon(color="black", icon="ban-circle"),
        tooltip="User Reported Live Flood (Road Blocked)"
    ).add_to(m)

# Render map and catch user clicks for real-time reporting
map_data = st_folium(m, width="100%", height=520, returned_objects=["last_clicked"])

# Process map clicks to add new flood reports
if map_data and map_data.get("last_clicked"):
    clicked_lat = map_data["last_clicked"]["lat"]
    clicked_lon = map_data["last_clicked"]["lng"]
    new_report = (clicked_lat, clicked_lon)
    
    # Prevent infinite loop of appending the same click
    if new_report not in st.session_state.user_reports:
        st.session_state.user_reports.append(new_report)
        st.rerun()

# ==========================================
# SECTION 2: DATA VISUALIZATION GRAPHS
# ==========================================
st.markdown("---")
st.header("2. Geospatial & Model Data Analytics")

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