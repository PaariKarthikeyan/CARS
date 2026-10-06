import geopandas as gpd
import pandas as pd
import numpy as np
from pathlib import Path
import warnings

# Suppress pandas FutureWarnings from geopandas sjoin
warnings.filterwarnings('ignore', category=FutureWarning)

def convert_kml_to_geojson(kml_path, geojson_path):
    """Reads a KML, forces 2D geometry, and saves as GeoJSON."""
    if not kml_path.exists():
        print(f"  - Warning: {kml_path.name} not found. Skipping conversion.")
        return None
        
    try:
        gdf = gpd.read_file(kml_path, engine="pyogrio", driver="KML")
        if not gdf.empty:
            gdf["geometry"] = gdf.geometry.force_2d()
            gdf.to_file(geojson_path, driver="GeoJSON", engine="pyogrio")
            print(f"  - Converted {kml_path.name} ({len(gdf)} features)")
            return gdf
    except Exception as e:
        print(f"  - Error reading {kml_path.name}: {e}")
    return None

def process_flood_history():
    # Directories
    raw_dir = Path("data/raw/flood_history")
    out_dir = Path("data/processed/flood_history")
    out_dir.mkdir(parents=True, exist_ok=True)
    
    grid_path = Path("data/processed/grid/chennai_grid.geojson")
    if not grid_path.exists():
        print(f"ERROR: {grid_path} not found. Run create_grid.py first.")
        return

    # Load Grid
    print("Loading grid...")
    grid = gpd.read_file(grid_path, engine="pyogrio")
    
    # Initialize base feature columns
    grid['historical_flood_count'] = 0
    grid['inundated_flag'] = 0
    grid['flood_depth_mean'] = 0.0
    grid['flood_risk_class'] = "Unknown"

    print("\nConverting KMLs and extracting spatial features...")
    
    # 1. Process Flood Points (2015)
    pts_kml = raw_dir / "chennai_flood_points_2015.kml"
    pts_geo = out_dir / "chennai_flood_points_2015.geojson"
    pts_gdf = convert_kml_to_geojson(pts_kml, pts_geo)
    
    if pts_gdf is not None and not pts_gdf.empty:
        # Spatial join to count points per cell
        pts_in_grid = gpd.sjoin(pts_gdf, grid[['cell_id', 'geometry']], how="inner", predicate="intersects")
        counts = pts_in_grid.groupby('cell_id').size().reset_index(name='count')
        grid = grid.merge(counts, on='cell_id', how='left')
        grid['historical_flood_count'] = grid['count'].fillna(0).astype(int)
        grid = grid.drop(columns=['count'])

    # 2. Process Inundation Depth Polygons
    inundation_kml = raw_dir / "chennai_inundation_depth.kml"
    inundation_geo = out_dir / "chennai_inundation_depth.geojson"
    inun_gdf = convert_kml_to_geojson(inundation_kml, inundation_geo)
    
    if inun_gdf is not None and not inun_gdf.empty:
        inun_in_grid = gpd.sjoin(grid[['cell_id', 'geometry']], inun_gdf, how="inner", predicate="intersects")
        inundated_cells = inun_in_grid['cell_id'].unique()
        grid.loc[grid['cell_id'].isin(inundated_cells), 'inundated_flag'] = 1
        
        # Attempt to parse depth if available (Often KMLs store this in 'Name' or 'Description')
        # Here we check for a 'depth' column; if none, it remains 0.0 but flagged as inundated.
        depth_col = next((c for c in inun_gdf.columns if 'depth' in c.lower()), None)
        if depth_col:
            inun_in_grid[depth_col] = pd.to_numeric(inun_in_grid[depth_col], errors='coerce')
            mean_depths = inun_in_grid.groupby('cell_id')[depth_col].mean().reset_index(name='mean_d')
            grid = grid.merge(mean_depths, on='cell_id', how='left')
            grid['flood_depth_mean'] = grid['mean_d'].fillna(0.0).round(2)
            grid = grid.drop(columns=['mean_d'])

    # 3. Process Flood Hazard Zones
    hazard_kml = raw_dir / "chennai_flood_hazard_zones.kml"
    hazard_geo = out_dir / "chennai_flood_hazard_zones.geojson"
    haz_gdf = convert_kml_to_geojson(hazard_kml, hazard_geo)
    
    if haz_gdf is not None and not haz_gdf.empty:
        haz_in_grid = gpd.sjoin(grid[['cell_id', 'geometry']], haz_gdf, how="inner", predicate="intersects")
        # Find 'name' or 'description' representing the hazard class (e.g., 'High', 'Low')
        class_col = next((c for c in haz_gdf.columns if c.lower() in ['name', 'description', 'class', 'hazard']), None)
        if class_col:
            # Drop duplicates to keep the first matched hazard zone per cell
            haz_in_grid = haz_in_grid.drop_duplicates(subset=['cell_id'])
            haz_map = haz_in_grid.set_index('cell_id')[class_col].to_dict()
            grid['flood_risk_class'] = grid['cell_id'].map(haz_map).fillna("Unknown")

    # 4. Generate the Target Label (flood_label)
    print("\nGenerating final flood_label...")
    # Assumption: If a cell had a historical flood point OR fell inside an inundation polygon, it is labeled 1.
    grid['flood_label'] = np.where(
        (grid['historical_flood_count'] > 0) | (grid['inundated_flag'] > 0), 1, 0
    )
    
    # Class balance check
    num_flooded = grid['flood_label'].sum()
    num_safe = len(grid) - num_flooded
    print(f"Class Balance: {num_flooded} Flooded (1) | {num_safe} Non-Flooded (0)")
    
    if num_flooded == 0:
        print("WARNING: No flooded cells detected! Check if flood points overlap the grid bounding box.")
    
    # Clean up and save
    grid = grid.drop(columns=['inundated_flag']) # Drop temporary flag
    
    features_gdf = grid[[
        'cell_id', 'geometry', 'historical_flood_count', 
        'flood_depth_mean', 'flood_risk_class', 'flood_label'
    ]]
    
    out_features = out_dir / "flood_labels.geojson"
    features_gdf.to_file(out_features, driver="GeoJSON", engine="pyogrio")
    print(f"Successfully saved flood labels to {out_features}")

if __name__ == "__main__":
    process_flood_history()