import geopandas as gpd
import pandas as pd
from pathlib import Path
import warnings

# Suppress pandas FutureWarnings from geopandas overlay operations
warnings.filterwarnings('ignore', category=FutureWarning)

def process_roads():
    # Paths
    raw_edges = Path("data/raw/roads/chennai_roads.geojson")
    raw_nodes = Path("data/raw/roads/chennai_nodes.geojson")
    grid_path = Path("data/processed/grid/chennai_grid.geojson")
    
    out_dir = Path("data/processed/roads")
    out_dir.mkdir(parents=True, exist_ok=True)
    
    out_edges = out_dir / "chennai_roads_cleaned.geojson"
    out_nodes = out_dir / "chennai_nodes_cleaned.geojson"
    out_features = out_dir / "road_features.geojson"

    if not raw_edges.exists() or not raw_nodes.exists():
        print(f"ERROR: Missing raw road geojson files in data/raw/roads/. Cannot process.")
        return
    if not grid_path.exists():
        print(f"ERROR: {grid_path} not found. Run create_grid.py first.")
        return

    print("Loading raw roads and nodes...")
    edges = gpd.read_file(raw_edges, engine="pyogrio")
    nodes = gpd.read_file(raw_nodes, engine="pyogrio")
    grid = gpd.read_file(grid_path, engine="pyogrio")

    # 1. Clean geometries
    print("Cleaning geometries...")
    edges = edges[edges.geometry.is_valid & ~edges.geometry.is_empty].copy()
    nodes = nodes[nodes.geometry.is_valid & ~nodes.geometry.is_empty].copy()

    # Ensure essential columns exist
    if 'highway' not in edges.columns:
        edges['highway'] = 'unknown'
    if 'oneway' not in edges.columns:
        edges['oneway'] = False

    # 2. Reproject to EPSG:32644 (UTM 44N) for metric calculations
    print("Reprojecting to EPSG:32644 for metric calculations...")
    edges_utm = edges.to_crs(epsg=32644)
    nodes_utm = nodes.to_crs(epsg=32644)
    grid_utm = grid.to_crs(epsg=32644)

    # Calculate actual length in meters for individual road segments
    edges_utm['length_m'] = edges_utm.geometry.length

    # 3. Calculate Grid Features: Intersections (Nodes)
    print("Calculating intersection density per grid cell...")
    # Spatial join nodes to grid
    nodes_in_grid = gpd.sjoin(nodes_utm, grid_utm[['cell_id', 'geometry']], how="inner", predicate="intersects")
    node_counts = nodes_in_grid.groupby('cell_id').size().reset_index(name='intersection_count')

    # 4. Calculate Grid Features: Road Length & Density
    print("Intersecting roads with grid cells to calculate road density (this may take a minute)...")
    # Using overlay to precisely cut roads at grid boundaries
    roads_intersected = gpd.overlay(edges_utm, grid_utm[['cell_id', 'geometry']], how='intersection')
    
    # Calculate the length of the cut segments
    roads_intersected['segment_length_m'] = roads_intersected.geometry.length
    
    # Aggregate road length by cell_id
    road_lengths = roads_intersected.groupby('cell_id')['segment_length_m'].sum().reset_index(name='road_length_m')

    # 5. Merge features back to the grid dataframe
    print("Merging features...")
    grid_features = grid[['cell_id', 'area_sqm', 'geometry']].copy()
    
    # Merge road lengths
    grid_features = grid_features.merge(road_lengths, on='cell_id', how='left')
    grid_features['road_length_m'] = grid_features['road_length_m'].fillna(0)
    
    # Merge intersection counts
    grid_features = grid_features.merge(node_counts, on='cell_id', how='left')
    grid_features['intersection_count'] = grid_features['intersection_count'].fillna(0)
    
    # Calculate road density (meters of road per square meter of land)
    grid_features['road_density'] = grid_features['road_length_m'] / grid_features['area_sqm']

    # Round numerical features
    grid_features['road_length_m'] = grid_features['road_length_m'].round(2)
    grid_features['road_density'] = grid_features['road_density'].round(6)
    
    # Drop area_sqm for the final feature output to keep it clean
    grid_features = grid_features[['cell_id', 'road_length_m', 'road_density', 'intersection_count', 'geometry']]

    # 6. Save outputs
    print("Saving cleaned files to EPSG:4326...")
    
    # Keep only important columns for the edges to reduce file size
    cols_to_keep = ['highway', 'oneway', 'length_m', 'geometry']
    edges_clean = edges_utm[cols_to_keep].to_crs(epsg=4326)
    nodes_clean = nodes_utm[['geometry']].to_crs(epsg=4326)

    edges_clean.to_file(out_edges, driver="GeoJSON", engine="pyogrio")
    nodes_clean.to_file(out_nodes, driver="GeoJSON", engine="pyogrio")
    
    # Save the grid features
    grid_features.to_file(out_features, driver="GeoJSON", engine="pyogrio")

    print(f"Success! Saved cleaned roads and extracted features for {len(grid_features)} grid cells.")

if __name__ == "__main__":
    process_roads()