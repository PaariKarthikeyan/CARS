import numpy as np
import geopandas as gpd
from shapely.geometry import Polygon
from pathlib import Path

def create_grid():
    # Bounding Box for Chennai
    xmin, ymin = 80.05, 12.85
    xmax, ymax = 80.35, 13.25
    
    # Grid cell size (degrees)
    # Note: 0.005 degrees is approx 550m x 550m.
    # Degree-based cells are not exactly equal in area globally, 
    # but at a local city scale, the distortion is negligible.
    cell_size = 0.005
    
    # Create coordinate arrays
    x_coords = np.arange(xmin, xmax, cell_size)
    y_coords = np.arange(ymin, ymax, cell_size)
    
    polygons = []
    
    print(f"Generating grid cells (Size: {cell_size} degrees)...")
    for x in x_coords:
        for y in y_coords:
            polygons.append(Polygon([
                (x, y), 
                (x + cell_size, y), 
                (x + cell_size, y + cell_size), 
                (x, y + cell_size)
            ]))
            
    # Create GeoDataFrame in WGS84 (EPSG:4326)
    grid = gpd.GeoDataFrame({'geometry': polygons}, crs="EPSG:4326")
    
    # Assign unique cell IDs
    grid['cell_id'] = [f"cell_{i:05d}" for i in range(len(grid))]
    
    # Calculate Centroids
    grid['centroid_longitude'] = grid.geometry.centroid.x
    grid['centroid_latitude'] = grid.geometry.centroid.y
    
    # Reproject to UTM Zone 44N (EPSG:32644) to calculate area in square meters
    print("Reprojecting to EPSG:32644 for metric area calculation...")
    grid_utm = grid.to_crs(epsg=32644)
    grid['area_sqm'] = grid_utm.geometry.area
    
    # Reorder columns for readability
    grid = grid[['cell_id', 'centroid_longitude', 'centroid_latitude', 'area_sqm', 'geometry']]
    
    # Ensure output directory exists
    output_dir = Path("data/processed/grid")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    output_file = output_dir / "chennai_grid.geojson"
    print(f"Saving {len(grid)} grid cells to {output_file}...")
    
    # Save using pyogrio for speed
    grid.to_file(output_file, driver="GeoJSON", engine="pyogrio")
    print("Grid creation complete!")

if __name__ == "__main__":
    create_grid()