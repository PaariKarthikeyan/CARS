import geopandas as gpd
import pandas as pd
from sklearn.cluster import DBSCAN
from pathlib import Path

def run_dbscan_hotspots():
    # Paths
    pts_path = Path("data/processed/flood_history/chennai_flood_points_2015.geojson")
    out_dir = Path("data/processed/flood_history")
    out_hotspots = out_dir / "flood_hotspots.geojson"

    if not pts_path.exists():
        print(f"ERROR: {pts_path} not found. Ensure process_flood_history.py ran successfully.")
        return

    print(f"Loading historical flood points from {pts_path}...")
    flood_pts = gpd.read_file(pts_path, engine="pyogrio")
    
    if flood_pts.empty:
        print("ERROR: Flood points file is empty. Cannot run DBSCAN.")
        return

    # 1. Reproject to EPSG:32644 (UTM 44N) to calculate distances in meters
    print("Reprojecting to EPSG:32644 for metric clustering...")
    flood_pts_utm = flood_pts.to_crs(epsg=32644)

    # Extract X (Easting) and Y (Northing) coordinates
    coords = pd.DataFrame({
        'x': flood_pts_utm.geometry.x,
        'y': flood_pts_utm.geometry.y
    })

    # 2. Configure and Run DBSCAN
    # eps = 500 meters: points within 500m of each other can be grouped.
    # min_samples = 5: at least 5 points needed to declare a 'hotspot'.
    eps_meters = 500
    min_pts = 5
    
    print(f"Running DBSCAN clustering (eps={eps_meters}m, min_samples={min_pts})...")
    db = DBSCAN(eps=eps_meters, min_samples=min_pts)
    clusters = db.fit_predict(coords)

    # 3. Assign labels and summarize
    flood_pts_utm['cluster_id'] = clusters
    
    # Calculate statistics
    n_clusters = len(set(clusters)) - (1 if -1 in clusters else 0)
    n_noise = list(clusters).count(-1)
    
    print(f"\nDBSCAN Results:")
    print(f"  - Total Hotspots (Clusters) Found: {n_clusters}")
    print(f"  - Isolated Events (Noise / -1): {n_noise}")
    print(f"  - Clustered Events: {len(clusters) - n_noise}")
    
    # Group by cluster to see size of each hotspot
    cluster_counts = flood_pts_utm[flood_pts_utm['cluster_id'] != -1]['cluster_id'].value_counts()
    if not cluster_counts.empty:
        print("\nTop 5 Largest Hotspots:")
        for cluster_id, count in cluster_counts.head(5).items():
            print(f"  - Cluster {cluster_id}: {count} points")

    # 4. Save the results
    print(f"\nReprojecting back to EPSG:4326 for mapping...")
    # Map back to standard latitude/longitude
    hotspots_out = flood_pts_utm.to_crs(epsg=4326)
    
    # Save as GeoJSON
    hotspots_out.to_file(out_hotspots, driver="GeoJSON", engine="pyogrio")
    print(f"Saved hotspot clusters to {out_hotspots}")
    print("Note: DBSCAN is used here exclusively for hotspot discovery, NOT for supervised flood prediction.")

if __name__ == "__main__":
    run_dbscan_hotspots()