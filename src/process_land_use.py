import numpy as np
import geopandas as gpd
import rasterio
from rasterio.mask import mask
from shapely.geometry import box
from rasterstats import zonal_stats
from pathlib import Path

def process_land_use():
    # Paths
    raw_lc_path = Path("data/raw/land_use/chennai_landcover.tif")
    grid_path = Path("data/processed/grid/chennai_grid.geojson")
    
    out_dir = Path("data/processed/land_use")
    out_dir.mkdir(parents=True, exist_ok=True)
    
    out_clipped_lc = out_dir / "chennai_landcover_clipped.tif"
    out_features = out_dir / "land_use_features.geojson"

    if not raw_lc_path.exists():
        print(f"ERROR: {raw_lc_path} not found.")
        return
    if not grid_path.exists():
        print(f"ERROR: {grid_path} not found. Run create_grid.py first.")
        return

    # 1. Define Chennai Bounding Box (EPSG:4326)
    minx, miny, maxx, maxy = 80.05, 12.85, 80.35, 13.25
    bbox = box(minx, miny, maxx, maxy)
    bbox_gdf = gpd.GeoDataFrame({'geometry': [bbox]}, crs="EPSG:4326")

    print(f"Reading and clipping raw Land Cover: {raw_lc_path}...")
    
    # 2. Clip the Raster
    with rasterio.open(raw_lc_path) as src:
        # Match CRS if needed
        if src.crs and src.crs.to_epsg() != 4326:
            bbox_gdf = bbox_gdf.to_crs(src.crs)
            
        out_image, out_transform = mask(src, bbox_gdf.geometry, crop=True)
        out_meta = src.meta.copy()
        
        # Determine nodata
        nodata_val = src.nodata if src.nodata is not None else 0
        
        out_meta.update({
            "driver": "GTiff",
            "height": out_image.shape[1],
            "width": out_image.shape[2],
            "transform": out_transform,
            "crs": src.crs
        })
        
        # Save clipped LC
        with rasterio.open(out_clipped_lc, "w", **out_meta) as dest:
            dest.write(out_image)

    print(f"Saved clipped land cover to {out_clipped_lc}")

    # 3. Extract Zonal Categorical Statistics
    print(f"Extracting land cover pixel counts for grid cells...")
    grid = gpd.read_file(grid_path, engine="pyogrio")
    
    # Calculate categorical stats (returns a dict of pixel counts for each class code)
    lc_stats = zonal_stats(
        vectors=grid.geometry,
        raster=out_clipped_lc,
        categorical=True,
        nodata=nodata_val,
        geojson_out=False
    )
    
    # 4. Process Categories into Target Features
    # ESA WorldCover Groups:
    # Vegetation: 10 (Trees), 20 (Shrubs), 30 (Grass), 90 (Wetland), 95 (Mangroves)
    # Cropland: 40
    # Built-up: 50
    # Bare land: 60
    # Water: 80
    
    built_up_pct = []
    water_pct = []
    veg_pct = []
    crop_pct = []
    bare_pct = []
    dominant_lc = []
    
    for cell_stats in lc_stats:
        # Safely get pixel counts (default to 0 if class is missing in the cell)
        built_up = cell_stats.get(50, 0)
        water = cell_stats.get(80, 0)
        veg = cell_stats.get(10, 0) + cell_stats.get(20, 0) + cell_stats.get(30, 0) + cell_stats.get(90, 0) + cell_stats.get(95, 0)
        crop = cell_stats.get(40, 0)
        bare = cell_stats.get(60, 0)
        
        total_pixels = built_up + water + veg + crop + bare
        
        if total_pixels == 0:
            built_up_pct.append(0.0)
            water_pct.append(0.0)
            veg_pct.append(0.0)
            crop_pct.append(0.0)
            bare_pct.append(0.0)
            dominant_lc.append("Unknown")
            continue
            
        # Calculate percentages
        built_up_pct.append((built_up / total_pixels) * 100)
        water_pct.append((water / total_pixels) * 100)
        veg_pct.append((veg / total_pixels) * 100)
        crop_pct.append((crop / total_pixels) * 100)
        bare_pct.append((bare / total_pixels) * 100)
        
        # Determine dominant land cover name
        counts = {
            "Built-up": built_up,
            "Water": water,
            "Vegetation": veg,
            "Cropland": crop,
            "Bare Land": bare
        }
        dominant_lc.append(max(counts, key=counts.get))
        
    # Append to grid
    grid['built_up_percentage'] = np.round(built_up_pct, 2)
    grid['water_percentage'] = np.round(water_pct, 2)
    grid['vegetation_percentage'] = np.round(veg_pct, 2)
    grid['cropland_percentage'] = np.round(crop_pct, 2)
    grid['bare_land_percentage'] = np.round(bare_pct, 2)
    grid['dominant_landcover'] = dominant_lc
    
    # Keep only ID and features
    features_gdf = grid[[
        'cell_id', 'geometry', 'dominant_landcover', 'built_up_percentage', 
        'water_percentage', 'vegetation_percentage', 'cropland_percentage', 'bare_land_percentage'
    ]]
    
    # Save features
    features_gdf.to_file(out_features, driver="GeoJSON", engine="pyogrio")
    print(f"Successfully saved land use features for {len(grid)} cells to {out_features}")

if __name__ == "__main__":
    process_land_use()