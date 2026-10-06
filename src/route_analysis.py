import osmnx as ox
import networkx as nx
import geopandas as gpd
import pandas as pd
import numpy as np
import joblib
from pathlib import Path
import warnings

warnings.filterwarnings("ignore")

def analyze_routes():
    # Paths
    graph_path = Path("data/raw/roads/chennai_roads.graphml")
    grid_path = Path("data/processed/grid/chennai_grid_with_features.geojson")
    model_path = Path("models/random_forest.pkl")
    out_dir = Path("data/processed/roads")
    out_graph = out_dir / "chennai_roads_risk.graphml"

    if not graph_path.exists():
        print(f"ERROR: {graph_path} not found. Ensure OSMnx GraphML is in the raw data folder.")
        return
    if not grid_path.exists() or not model_path.exists():
        print("ERROR: Grid features or model not found. Run combine_features.py and train_models.py first.")
        return

    print("1. Loading trained model and master grid...")
    model = joblib.load(model_path)
    grid = gpd.read_file(grid_path, engine="pyogrio")

    # Predict flood probabilities for the grid
    features = [
        'elevation_mean', 'slope_mean', 'rainfall_24h', 'rainfall_72h', 
        'rainfall_7day', 'distance_to_water_m', 'distance_to_drain_m', 
        'drainage_density', 'built_up_percentage', 'water_percentage'
    ]
    X = grid[features].fillna(0)
    
    # Extract probability of class 1 (Flood)
    grid['flood_prob'] = model.predict_proba(X)[:, 1]

    print("2. Loading OpenStreetMap road network...")
    G = ox.load_graphml(graph_path)
    gdf_nodes, gdf_edges = ox.graph_to_gdfs(G)

    print("3. Assigning flood probabilities to road segments...")
    # Use edge centroids for a quick spatial join with the grid
    edges_centroids = gdf_edges.copy()
    edges_centroids['geometry'] = edges_centroids.geometry.centroid
    
    if edges_centroids.crs != grid.crs:
        edges_centroids = edges_centroids.to_crs(grid.crs)

    edges_with_risk = gpd.sjoin(edges_centroids, grid[['cell_id', 'flood_prob', 'geometry']], how='left', predicate='intersects')
    
    # Handle duplicates if an edge centroid touches a boundary exactly
    edges_with_risk = edges_with_risk[~edges_with_risk.index.duplicated(keep='first')]
    edges_with_risk['flood_prob'] = edges_with_risk['flood_prob'].fillna(0.0)

    print("4. Calculating Risk-Adjusted Edge Costs...")
    # Penalty factor: e.g., a 100% probability adds 5000 meters to the segment's perceived length
    RISK_PENALTY_FACTOR = 5000 
    
    for u, v, k, data in G.edges(keys=True, data=True):
        try:
            prob = float(edges_with_risk.loc[(u, v, k), 'flood_prob'])
        except KeyError:
            prob = 0.0
            
        # Ensure length is numeric
        length = data.get('length', 100.0)
        if isinstance(length, str):
            # Sometimes length comes as a list string in OSM data like "['100', '110']"
            try:
                length = float(length.strip("[]").replace("'", "").split(",")[0])
            except ValueError:
                length = 100.0
        
        # Calculate routing weights
        risk_penalty = prob * RISK_PENALTY_FACTOR
        risk_adjusted_cost = length + risk_penalty
        
        # Update graph with new edge attributes
        data['flood_prob'] = prob
        data['risk_adjusted_length'] = risk_adjusted_cost
        data['length'] = float(length)

    print(f"Saving risk-enriched road graph to {out_graph}...")
    ox.save_graphml(G, out_graph)

    print("\n5. Testing Routing Algorithms...")
    nodes = list(G.nodes())
    
    # Pick two nodes roughly far apart for a test
    start_node = nodes[100]
    end_node = nodes[len(nodes) - 100]
    
    try:
        # Standard Route (Shortest Distance)
        shortest_route = nx.shortest_path(G, start_node, end_node, weight='length')
        shortest_len = sum(G[u][v][0]['length'] for u, v in zip(shortest_route[:-1], shortest_route[1:]))
        
        # Get edges on shortest route with > 50% flood prob
        shortest_high_risk = sum(1 for u, v in zip(shortest_route[:-1], shortest_route[1:]) if G[u][v][0]['flood_prob'] > 0.5)

        # Safer Route (Avoids Floods)
        safer_route = nx.shortest_path(G, start_node, end_node, weight='risk_adjusted_length')
        safer_len = sum(G[u][v][0]['length'] for u, v in zip(safer_route[:-1], safer_route[1:]))
        
        # Get edges on safer route with > 50% flood prob
        safer_high_risk = sum(1 for u, v in zip(safer_route[:-1], safer_route[1:]) if G[u][v][0]['flood_prob'] > 0.5)

        print("\n--- Route Comparison Results ---")
        print("ROUTE 1: Shortest Distance")
        print(f"  - Distance: {shortest_len / 1000:.2f} km")
        print(f"  - High-Risk Road Segments: {shortest_high_risk}")
        
        print("\nROUTE 2: Lower-Risk Route")
        print(f"  - Distance: {safer_len / 1000:.2f} km")
        print(f"  - High-Risk Road Segments: {safer_high_risk}")
        print("--------------------------------\n")
        print("Note: The 'Safer' route is a lower PREDICTED-risk route, not a guarantee of physical safety.")

    except nx.NetworkXNoPath:
        print("Could not find a path between the test nodes (network might be disconnected).")

if __name__ == "__main__":
    analyze_routes()