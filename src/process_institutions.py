import geopandas as gpd
import pandas as pd
from pathlib import Path

def generate_manual_institutions():
    print("Generating manual Chennai institutions dataset...")
    
    # Extensive manual list of institutions across all Chennai zones
    data = [
        # --- CENTRAL CHENNAI ---
        {"name": "Loyola College", "amenity": "college", "lat": 13.0604, "lon": 80.2336},
        {"name": "Stella Maris College", "amenity": "college", "lat": 13.0473, "lon": 80.2505},
        {"name": "Presidency College", "amenity": "college", "lat": 13.0580, "lon": 80.2825},
        {"name": "Madras Medical College (MMC)", "amenity": "college", "lat": 13.0802, "lon": 80.2783},
        {"name": "Ethiraj College for Women", "amenity": "college", "lat": 13.0620, "lon": 80.2580},
        {"name": "Women's Christian College (WCC)", "amenity": "college", "lat": 13.0645, "lon": 80.2435},
        {"name": "Pachaiyappa's College", "amenity": "college", "lat": 13.0735, "lon": 80.2300},
        {"name": "MOP Vaishnav College", "amenity": "college", "lat": 13.0580, "lon": 80.2450},
        {"name": "Don Bosco Matriculation School, Egmore", "amenity": "school", "lat": 13.0720, "lon": 80.2580},
        {"name": "DAV Boys Senior Secondary, Gopalapuram", "amenity": "school", "lat": 13.0470, "lon": 80.2580},
        {"name": "Church Park (Sacred Heart)", "amenity": "school", "lat": 13.0585, "lon": 80.2545},
        {"name": "Good Shepherd Convent", "amenity": "school", "lat": 13.0520, "lon": 80.2450},
        {"name": "Madras Veterinary College", "amenity": "college", "lat": 13.0860, "lon": 80.2605},
        {"name": "Vidyodaya Schools, T Nagar", "amenity": "school", "lat": 13.0435, "lon": 80.2400},
        {"name": "Ramakrishna Mission Vivekananda College", "amenity": "college", "lat": 13.0325, "lon": 80.2670},
        
        # --- SOUTH CHENNAI & VELACHERY ---
        {"name": "IIT Madras", "amenity": "university", "lat": 12.9915, "lon": 80.2337},
        {"name": "Anna University (Guindy)", "amenity": "university", "lat": 13.0102, "lon": 80.2359},
        {"name": "Guru Nanak College", "amenity": "college", "lat": 12.9770, "lon": 80.2180},
        {"name": "Chettinad Vidyashram, RA Puram", "amenity": "school", "lat": 13.0240, "lon": 80.2650},
        {"name": "St. Bede's, Santhome", "amenity": "school", "lat": 13.0330, "lon": 80.2780},
        {"name": "Kendriya Vidyalaya, IIT Campus", "amenity": "school", "lat": 12.9930, "lon": 80.2380},
        {"name": "Vana Vani Matriculation, IIT", "amenity": "school", "lat": 12.9890, "lon": 80.2320},
        {"name": "National Institute of Fashion Technology (NIFT)", "amenity": "college", "lat": 12.9896, "lon": 80.2450},
        
        # --- SOUTH WEST (TAMBARAM / PALLAVARAM CORRIDOR) ---
        {"name": "Madras Christian College (MCC)", "amenity": "college", "lat": 12.9213, "lon": 80.1218},
        {"name": "BS Abdur Rahman Crescent Institute", "amenity": "university", "lat": 12.8780, "lon": 80.0840},
        {"name": "Vels University, Pallavaram", "amenity": "university", "lat": 12.9550, "lon": 80.1550},
        {"name": "Sri Sai Ram Engineering College", "amenity": "college", "lat": 12.9600, "lon": 80.0550},
        {"name": "Madras Institute of Technology (MIT)", "amenity": "college", "lat": 12.9480, "lon": 80.1400},
        {"name": "SDNB Vaishnav College for Women", "amenity": "college", "lat": 12.9510, "lon": 80.1380},
        {"name": "Sita Devi Garodia Hindu Vidyalaya", "amenity": "school", "lat": 12.9260, "lon": 80.1110},
        
        # --- OMR / IT CORRIDOR ---
        {"name": "Sathyabama Institute of Science and Technology", "amenity": "university", "lat": 12.8730, "lon": 80.2200},
        {"name": "St. Joseph's College of Engineering", "amenity": "college", "lat": 12.8680, "lon": 80.2180},
        {"name": "KCG College of Technology", "amenity": "college", "lat": 12.9211, "lon": 80.2350},
        {"name": "MNM Jain Engineering College", "amenity": "college", "lat": 12.9360, "lon": 80.2340},
        {"name": "Jeppiaar Engineering College", "amenity": "college", "lat": 12.8710, "lon": 80.2200},
        {"name": "BVM Global School, Perungudi", "amenity": "school", "lat": 12.9630, "lon": 80.2450},
        {"name": "Abacus Montessori School", "amenity": "school", "lat": 12.9550, "lon": 80.2420},
        
        # --- WEST CHENNAI (ANNA NAGAR / PORUR / POONAMALLEE) ---
        {"name": "SBOA School and Junior College", "amenity": "school", "lat": 13.0880, "lon": 80.2000},
        {"name": "Chennai Public School, Anna Nagar", "amenity": "school", "lat": 13.0890, "lon": 80.1930},
        {"name": "DG Vaishnav College", "amenity": "college", "lat": 13.0768, "lon": 80.2155},
        {"name": "Velammal Matriculation, Mogappair", "amenity": "school", "lat": 13.0830, "lon": 80.1800},
        {"name": "SRM University, Ramapuram", "amenity": "university", "lat": 13.0334, "lon": 80.1804},
        {"name": "PSBB Millennium School, Porur", "amenity": "school", "lat": 13.0280, "lon": 80.1550},
        {"name": "Saveetha Dental College", "amenity": "college", "lat": 13.0450, "lon": 80.1380},
        {"name": "Panimalar Engineering College", "amenity": "college", "lat": 13.0480, "lon": 80.0570},
        {"name": "SA Engineering College", "amenity": "college", "lat": 13.1040, "lon": 80.1250},
        {"name": "Maharishi Vidya Mandir, Chetpet", "amenity": "school", "lat": 13.0730, "lon": 80.2400},
        {"name": "PSBB, KK Nagar", "amenity": "school", "lat": 13.0380, "lon": 80.1980},
        {"name": "Meenakshi College for Women", "amenity": "college", "lat": 13.0530, "lon": 80.2220},
        {"name": "Vellore Institute of Technology (VIT) Chennai", "amenity": "university", "lat": 12.8406, "lon": 80.1534},
        
        # --- NORTH CHENNAI ---
        {"name": "Stanley Medical College", "amenity": "college", "lat": 13.1050, "lon": 80.2850},
        {"name": "St. Anne's Girls School, Royapuram", "amenity": "school", "lat": 13.1110, "lon": 80.2930},
        {"name": "Velammal Engineering College, Surapet", "amenity": "college", "lat": 13.1430, "lon": 80.1830},
        {"name": "Government Kilpauk Medical College", "amenity": "college", "lat": 13.0805, "lon": 80.2410},
        {"name": "Bhavan's Rajaji Vidyashram, Kilpauk", "amenity": "school", "lat": 13.0850, "lon": 80.2420},
        {"name": "Sriram Engineering College, Veppampattu", "amenity": "college", "lat": 13.1360, "lon": 80.0530},
        {"name": "Sir Theagaraya College", "amenity": "college", "lat": 13.1160, "lon": 80.2830},
        {"name": "Don Bosco School, Perambur", "amenity": "school", "lat": 13.1130, "lon": 80.2460},
        {"name": "Chennai Public School, Thirumazhisai", "amenity": "school", "lat": 13.0550, "lon": 80.0520}
    ]

    # Convert to DataFrame
    df = pd.DataFrame(data)
    
    # Create GeoDataFrame using EPSG:4326 (Lat/Lon)
    gdf = gpd.GeoDataFrame(
        df, 
        geometry=gpd.points_from_xy(df.lon, df.lat), 
        crs="EPSG:4326"
    )

    # Setup directories
    raw_dir = Path("data/raw/institutions")
    raw_dir.mkdir(parents=True, exist_ok=True)
    raw_path = raw_dir / "chennai_institutions.geojson"

    processed_dir = Path("data/processed/institutions")
    processed_dir.mkdir(parents=True, exist_ok=True)
    processed_path = processed_dir / "chennai_institutions_cleaned.geojson"

    # Save to raw
    print(f"Saving raw manual dataset to {raw_path}...")
    gdf.to_file(raw_path, driver="GeoJSON")
    
    # (Optional) Normally you'd clean raw data here, but since it's manually curated, 
    # it is already perfectly clean. We just drop lat/lon and save to processed.
    gdf_clean = gdf[['name', 'amenity', 'geometry']].copy()
    
    print(f"Saving processed dataset to {processed_path}...")
    gdf_clean.to_file(processed_path, driver="GeoJSON")
    
    print(f"\nSuccess! Built and saved a highly distributed local dataset containing {len(gdf_clean)} major institutions.")

if __name__ == "__main__":
    generate_manual_institutions()