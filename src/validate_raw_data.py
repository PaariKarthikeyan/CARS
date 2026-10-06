import os
import pandas as pd
import geopandas as gpd
import rasterio
from pathlib import Path

def get_file_size(filepath):
    size_bytes = os.path.getsize(filepath)
    return f"{size_bytes / (1024 * 1024):.2f} MB"

def validate_vector(filepath, driver=None):
    try:
        gdf = gpd.read_file(filepath, engine="pyogrio", driver=driver) if driver else gpd.read_file(filepath, engine="pyogrio")
        print(f"  - CRS: {gdf.crs}")
        print(f"  - Rows: {len(gdf)}")
        print(f"  - Geometry types: {gdf.geometry.type.unique()}")
        print(f"  - Columns: {list(gdf.columns)}")
    except Exception as e:
        print(f"  - ERROR reading vector: {e}")

def validate_raster(filepath):
    try:
        with rasterio.open(filepath) as src:
            print(f"  - CRS: {src.crs}")
            print(f"  - Dimensions: {src.width}x{src.height} (WxH)")
            print(f"  - Bounds: {src.bounds}")
            print(f"  - Bands: {src.count}")
    except Exception as e:
        print(f"  - ERROR reading raster: {e}")

def validate_tabular(filepath):
    try:
        df = pd.read_csv(filepath)
        print(f"  - Rows: {len(df)}")
        print(f"  - Columns: {list(df.columns)}")
    except Exception as e:
        print(f"  - ERROR reading tabular: {e}")

def check_file(path_str, file_type, driver=None):
    filepath = Path(path_str)
    print(f"\nChecking: {filepath}")
    if filepath.exists():
        print(f"  - Status: EXISTS ({get_file_size(filepath)})")
        if file_type == 'vector':
            validate_vector(filepath, driver)
        elif file_type == 'raster':
            validate_raster(filepath)
        elif file_type == 'tabular':
            validate_tabular(filepath)
    else:
        print("  - Status: MISSING")
        print(f"  -> Action: If this file is missing, the corresponding process script will either skip its feature extraction (filling with 0/NaN) or use fallback data. Check data/raw/ folders.")

def main():
    print("==========================================")
    print("RAW DATA VALIDATION")
    print("==========================================")
    
    # Tabular
    check_file("data/raw/rainfall/chennai_rainfall.csv", "tabular")
    
    # Rasters
    check_file("data/raw/elevation/chennai_elevation.tif", "raster")
    check_file("data/raw/land_use/chennai_landcover.tif", "raster")
    
    # Vectors
    check_file("data/raw/roads/chennai_roads.geojson", "vector")
    check_file("data/raw/roads/chennai_nodes.geojson", "vector")
    check_file("data/raw/water_bodies/chennai_water.geojson", "vector")
    
    # KMLs (using pyogrio with KML driver)
    check_file("data/raw/flood_history/chennai_flood_points_2015.kml", "vector", driver="KML")
    check_file("data/raw/flood_history/chennai_inundation_depth.kml", "vector", driver="KML")
    check_file("data/raw/flood_history/chennai_flood_hazard_zones.kml", "vector", driver="KML")
    
    # GraphML (just check existence)
    graphml_path = Path("data/raw/roads/chennai_roads.graphml")
    print(f"\nChecking: {graphml_path}")
    if graphml_path.exists():
        print(f"  - Status: EXISTS ({get_file_size(graphml_path)})")
        print("  - Type: NetworkX GraphML (Validated during road processing)")
    else:
        print("  - Status: MISSING (Routing will fail without this)")

if __name__ == "__main__":
    main()