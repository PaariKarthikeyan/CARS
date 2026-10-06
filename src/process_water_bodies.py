import geopandas as gpd
import pandas as pd
import numpy as np
from pathlib import Path
import warnings

# Suppress pandas FutureWarnings from geopandas overlay and sjoin
warnings.filterwarnings('ignore', category=FutureWarning)

def classify_water_features(gdf):
    """
    Categorizes geometries into 'water' (lakes/ponds), 'river', or 'drain' (canals/drains).
    Handles missing columns safely since OSM data attributes can vary.
    """
    gdf['water_category'] = 'water' # Default
    
    # Check if 'waterway' column exists (standard OSM tag for rivers/drains)
    if 'waterway' in gdf.columns:
        gdf.loc[gdf['waterway'].isin(['river', 'stream']), 'water_category'] = 'river'
        gdf.loc[gdf['waterway'].isin(['canal', 'drain', 'ditch']), 'water_category'] = 'drain'
        
    return gdf

def process_water_bodies():
    # Paths
    raw_water_path = Path("data/raw/water_bodies/chennai_water.geojson")
    grid_path = Path("data/processed/grid/chennai_grid.geojson")
    
    out_dir = Path("data/processed/water_bodies")
    out_dir.mkdir(parents=True, exist_ok=True)
    
    out_water_cleaned = out_dir / "chennai_water_cleaned.geojson"
    out_features = out_dir / "water_distance_features.geojson"

    if not raw_water_path.exists():
        print(f"ERROR: {raw_water_path} not found.")
        return
    if not grid_path.exists():
        print(f"ERROR: {grid_path} not found. Run create_grid.py first.")
        return

    print("Loading raw water bodies and grid...")
    water_gdf = gpd.read_file(raw_water_path, engine="pyogrio")
    grid = gpd.read_file(grid_path, engine="pyogrio")

    # 1. Clean geometries
    print("Cleaning water geometries...")
    water_gdf = water_gdf[water_gdf.geometry.is_valid & ~water_gdf.geometry.is_empty].copy()
    
    # 2. Classify features
    water_gdf = classify_water_features(water_gdf)
    
    # 3. Reproject to EPSG:32644 (UTM 44N) for metric calculations
    print("Reprojecting to EPSG:32644...")
    water_utm = water_gdf.to_crs(epsg=32644)
    grid_utm = grid.to_crs(epsg=32644)
    
    # Create a separate geodataframe for grid centroids to measure distances
    grid_centroids = grid_utm.copy()
    grid_centroids['geometry'] = grid_centroids.geometry.centroid

    # Separate categories
    general_water = water_utm[water_utm['water_category'] == 'water']
    rivers = water_utm[water_utm['water_category'] == 'river']
    drains = water_utm[water_utm['water_category'] == 'drain']

    # Helper function to calculate nearest distances
    def get_nearest_distance(centroids, target_gdf, distance_col_name, default_dist=9999.0):
        if target_gdf.empty:
            print(f"  - Warning: No features found for {distance_col_name}. Using default {default_dist}m.")
            centroids[distance_col_name] = default_dist
            return centroids[[distance_col_name]]
            
        # sjoin_nearest calculates distances and joins indices
        nearest = gpd.sjoin_nearest(centroids, target_gdf, how='left', distance_col=distance_col_name)
        # Drop duplicates in case a centroid is equidistant to multiple features
        nearest = nearest[~nearest.index.duplicated(keep='first')]
        
        # Fill NaNs with default distance just in case
        nearest[distance_col_name] = nearest[distance_col_name].fillna(default_dist)
        return nearest[[distance_col_name]]

    # 4. Calculate Distances
    print("Calculating distances to nearest water bodies, rivers, and drains...")
    dist_w = get_nearest_distance(grid_centroids, general_water, 'distance_to_water_m')
    dist_r = get_nearest_distance(grid_centroids, rivers, 'distance_to_river_m')
    dist_d = get_nearest_distance(grid_centroids, drains, 'distance_to_drain_m')

    # Merge distances back to grid features
    grid_features = grid[['cell_id', 'area_sqm', 'geometry']].copy()
    grid_features = pd.concat([grid_features, dist_w, dist_r, dist_d], axis=1)

    # 5. Calculate Drainage/River Length and Density
    print("Intersecting drainage/rivers with grid cells to calculate density...")
    # Combine rivers and drains for the network
    drainage_network = water_utm[water_utm['water_category'].isin(['river', 'drain'])]
    
    if not drainage_network.empty:
        # Overlay network with grid
        drainage_intersected = gpd.overlay(drainage_network, grid_utm[['cell_id', 'geometry']], how='intersection')
        
        # Calculate length (for polygons this calculates perimeter, for lines it calculates length)
        drainage_intersected['length_m'] = drainage_intersected.geometry.length
        
        # Aggregate by cell
        drainage_lengths = drainage_intersected.groupby('cell_id')['length_m'].sum().reset_index(name='drainage_length_m')
        
        grid_features = grid_features.merge(drainage_lengths, on='cell_id', how='left')
    else:
        grid_features['drainage_length_m'] = 0.0

    # Fill NaNs with 0 for cells with no drainage
    grid_features['drainage_length_m'] = grid_features['drainage_length_m'].fillna(0)
    
    # Calculate density (meters of drainage per square meter of cell)
    grid_features['drainage_density'] = grid_features['drainage_length_m'] / grid_features['area_sqm']

    # Round numeric features
    cols_to_round = ['distance_to_water_m', 'distance_to_river_m', 'distance_to_drain_m', 'drainage_length_m']
    grid_features[cols_to_round] = grid_features[cols_to_round].round(2)
    grid_features['drainage_density'] = grid_features['drainage_density'].round(6)

    # Clean up output columns
    grid_features = grid_features.drop(columns=['area_sqm'])

    # 6. Save outputs
    print("Saving cleaned water data and grid features...")
    
    # Save cleaned water geometries in standard WGS84 for mapping
    cols_to_keep = ['water_category', 'geometry']
    # If other standard names exist in raw data, keep them too
    for col in ['name', 'waterway', 'natural', 'water']:
        if col in water_utm.columns:
            cols_to_keep.insert(0, col)
            
    water_clean = water_utm[cols_to_keep].to_crs(epsg=4326)
    water_clean.to_file(out_water_cleaned, driver="GeoJSON", engine="pyogrio")
    
    # Save features
    grid_features.to_file(out_features, driver="GeoJSON", engine="pyogrio")

    print(f"Success! Saved features for {len(grid_features)} grid cells.")
    print(f"Outputs saved to {out_dir}/")

if __name__ == "__main__":
    process_water_bodies()