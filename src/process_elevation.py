import os
import numpy as np
import geopandas as gpd
import rasterio
from rasterio.mask import mask
from rasterio.transform import from_bounds
from shapely.geometry import box
from rasterstats import zonal_stats
from pathlib import Path

def calculate_slope(elevation_array, transform, lat):
    """
    Calculate slope in degrees from an elevation array.
    Assumes elevation is in meters and horizontal CRS is EPSG:4326.
    Converts horizontal degree resolution to meters based on latitude.
    """
    # 1 degree of latitude is approx 111,320 meters
    # 1 degree of longitude is approx 111,320 * cos(latitude) meters
    dy_deg = abs(transform[4])
    dx_deg = abs(transform[0])
    
    dy_m = dy_deg * 111320
    dx_m = dx_deg * 111320 * np.cos(np.radians(lat))
    
    # Calculate gradients (rate of change) in x and y directions
    y_grad, x_grad = np.gradient(elevation_array, dy_m, dx_m)
    
    # Calculate slope in degrees
    slope_rad = np.arctan(np.sqrt(x_grad**2 + y_grad**2))
    slope_deg = np.degrees(slope_rad)
    
    return slope_deg

def process_elevation():
    # File paths
    raw_dem_path = Path("data/raw/elevation/chennai_elevation.tif")
    grid_path = Path("data/processed/grid/chennai_grid.geojson")
    
    out_dir = Path("data/processed/elevation")
    out_dir.mkdir(parents=True, exist_ok=True)
    
    out_clipped_dem = out_dir / "chennai_elevation_clipped.tif"
    out_slope = out_dir / "chennai_slope.tif"
    out_features = out_dir / "terrain_features.geojson"

    if not raw_dem_path.exists():
        print(f"ERROR: {raw_dem_path} not found.")
        return
    if not grid_path.exists():
        print(f"ERROR: {grid_path} not found. Run create_grid.py first.")
        return

    # 1. Define Chennai Bounding Box (EPSG:4326)
    minx, miny, maxx, maxy = 80.05, 12.85, 80.35, 13.25
    bbox = box(minx, miny, maxx, maxy)
    bbox_gdf = gpd.GeoDataFrame({'geometry': [bbox]}, crs="EPSG:4326")

    print(f"Reading and clipping raw DEM: {raw_dem_path}...")
    
    # 2. Clip the DEM
    with rasterio.open(raw_dem_path) as src:
        if src.crs and src.crs.to_epsg() != 4326:
            bbox_gdf = bbox_gdf.to_crs(src.crs)
            
        out_image, out_transform = mask(src, bbox_gdf.geometry, crop=True)
        out_meta = src.meta.copy()
        
        # Handle nodata values explicitly - convert to float32 first!
        nodata_val = src.nodata if src.nodata is not None else -9999
        out_image = out_image.astype(np.float32)
        out_image[out_image == nodata_val] = np.nan
        
        # Update metadata for the clipped raster (Now storing as float32)
        out_meta.update({
            "driver": "GTiff",
            "height": out_image.shape[1],
            "width": out_image.shape[2],
            "transform": out_transform,
            "crs": src.crs,
            "dtype": "float32",
            "nodata": np.nan
        })
        
        # Save clipped DEM
        with rasterio.open(out_clipped_dem, "w", **out_meta) as dest:
            dest.write(out_image)

    print(f"Saved clipped DEM to {out_clipped_dem}")

    # 3. Calculate Slope
    print("Calculating slope...")
    dem_array = out_image[0]
    center_lat = (miny + maxy) / 2
    
    slope_array = calculate_slope(dem_array, out_transform, center_lat)
    
    # Save Slope Raster
    with rasterio.open(out_slope, "w", **out_meta) as dest:
        dest.write(slope_array.reshape(1, slope_array.shape[0], slope_array.shape[1]).astype(np.float32))
        
    print(f"Saved slope raster to {out_slope}")

    # 4. Extract Zonal Statistics for the Grid
    print(f"Extracting elevation and slope statistics for grid cells...")
    grid = gpd.read_file(grid_path, engine="pyogrio")
    
    # Elevation stats (Using np.nan as the nodata value now)
    dem_stats = zonal_stats(
        vectors=grid.geometry,
        raster=out_clipped_dem,
        stats=['min', 'max', 'mean'],
        nodata=np.nan,
        geojson_out=False
    )
    
    # Slope stats
    slope_stats = zonal_stats(
        vectors=grid.geometry,
        raster=out_slope,
        stats=['min', 'max', 'mean'],
        nodata=np.nan,
        geojson_out=False
    )
    
    # Append to grid
    grid['elevation_min'] = [stat['min'] for stat in dem_stats]
    grid['elevation_max'] = [stat['max'] for stat in dem_stats]
    grid['elevation_mean'] = [stat['mean'] for stat in dem_stats]
    
    grid['slope_min'] = [stat['min'] for stat in slope_stats]
    grid['slope_max'] = [stat['max'] for stat in slope_stats]
    grid['slope_mean'] = [stat['mean'] for stat in slope_stats]
    
    # Handle cells with no coverage (NaNs) -> fill with 0 or drop.
    cols_to_fill = ['elevation_min', 'elevation_max', 'elevation_mean', 'slope_min', 'slope_max', 'slope_mean']
    grid[cols_to_fill] = grid[cols_to_fill].fillna(0).round(2)
    
    # Keep only ID and features (we can join geometries later)
    features_gdf = grid[['cell_id', 'geometry'] + cols_to_fill]
    
    # Save features
    features_gdf.to_file(out_features, driver="GeoJSON", engine="pyogrio")
    print(f"Successfully saved terrain features for {len(grid)} cells to {out_features}")

if __name__ == "__main__":
    process_elevation()