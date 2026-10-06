# Chennai Flood Risk and Road Accessibility Predictor

A geospatial machine-learning system for predicting flood-prone areas in Chennai and identifying roads that may have increased accessibility risk during heavy rainfall.

## Overview

Urban flooding in Chennai can affect transportation, infrastructure, public safety, and emergency response. This project combines rainfall, elevation, slope, land-use, drainage, water-body, road-network, and historical flood data to estimate flood risk across Chennai.

The system uses machine learning to predict flood risk for geographic grid cells, DBSCAN to identify recurring flood hotspots, and road-network analysis to recommend lower-risk routes.

> This project is intended for research and educational purposes. Its route recommendations should not be treated as official emergency instructions.

## Features

- Chennai flood-risk prediction using geospatial features.
- Rainfall-based flood-risk analysis.
- Elevation and slope analysis.
- Distance-to-water-body and drainage-density features.
- Historical flood hotspot detection using DBSCAN.
- Road-network analysis using OpenStreetMap data.
- Risk-adjusted route recommendations.
- Model comparison using Logistic Regression, Random Forest, and XGBoost.
- Explainable predictions using SHAP.
- Interactive Streamlit dashboard.
- Flood-risk mapping using Folium and GeoPandas.

## Project Workflow

```text
Collect datasets
      ↓
Validate raw data
      ↓
Create Chennai grid
      ↓
Process rainfall, elevation, roads, water bodies, land use, and flood data
      ↓
Extract spatial features
      ↓
Create flood labels
      ↓
Train machine-learning models
      ↓
Evaluate predictions
      ↓
Detect flood hotspots using DBSCAN
      ↓
Assign risk to road segments
      ↓
Calculate lower-risk routes
      ↓
Display results in Streamlit
```

## Project Structure

```text
chennai-flood-project/
│
├── app/
│   └── streamlit_app.py
│
├── data/
│   ├── raw/
│   │   ├── rainfall/
│   │   │   ├── chennai_rainfall.csv
│   │   │   └── rainfall_source.txt
│   │   │
│   │   ├── elevation/
│   │   │   └── chennai_elevation.tif
│   │   │
│   │   ├── roads/
│   │   │   ├── chennai_roads.graphml
│   │   │   ├── chennai_roads.geojson
│   │   │   └── chennai_nodes.geojson
│   │   │
│   │   ├── water_bodies/
│   │   │   └── chennai_water.geojson
│   │   │
│   │   ├── land_use/
│   │   │   └── chennai_landcover.tif
│   │   │
│   │   └── flood_history/
│   │       ├── chennai_flood_points_2015.kml
│   │       ├── chennai_inundation_depth.kml
│   │       ├── chennai_flood_hazard_zones.kml
│   │       └── flood_source.txt
│   │
│   ├── processed/
│   │   ├── rainfall/
│   │   ├── elevation/
│   │   ├── roads/
│   │   ├── water_bodies/
│   │   ├── land_use/
│   │   ├── flood_history/
│   │   ├── grid/
│   │   └── metadata/
│   │
│   └── final/
│       └── chennai_flood_training_dataset.csv
│
├── models/
│   ├── logistic_regression.pkl
│   ├── random_forest.pkl
│   └── xgboost.pkl
│
├── notebooks/
│   ├── 01_data_validation.ipynb
│   ├── 02_grid_creation.ipynb
│   ├── 03_data_processing.ipynb
│   ├── 04_model_training.ipynb
│   └── 05_hotspot_analysis.ipynb
│
├── reports/
│   ├── model_metrics.csv
│   ├── confusion_matrix.png
│   └── feature_importance.png
│
├── src/
│   ├── validate_raw_data.py
│   ├── create_grid.py
│   ├── process_rainfall.py
│   ├── process_elevation.py
│   ├── process_roads.py
│   ├── process_water_bodies.py
│   ├── process_land_use.py
│   ├── process_flood_history.py
│   ├── combine_features.py
│   ├── train_models.py
│   ├── evaluate_models.py
│   ├── dbscan_hotspots.py
│   └── route_analysis.py
│
├── .gitignore
├── requirements.txt
└── README.md
```

## Study Area

The project focuses on Chennai, Tamil Nadu, India.

Approximate study-area bounding box:

```text
Minimum longitude: 80.05
Minimum latitude: 12.85
Maximum longitude: 80.35
Maximum latitude: 13.25
```

Coordinate systems:

```text
EPSG:4326  - Geographic coordinates and web mapping
EPSG:32644 - Distance, area, road length, and spatial calculations
```

## Data Sources

### Rainfall

NASA POWER provides daily precipitation data.

- Website: https://power.larc.nasa.gov/data-access-viewer/
- Parameter: `PRECTOTCORR`
- Unit: millimetres per day
- File: `chennai_rainfall.csv`

### Elevation

SRTM elevation data is accessed through Google Earth Engine.

- Dataset: https://developers.google.com/earth-engine/datasets/catalog/USGS_SRTMGL1_003
- Output: `chennai_elevation.tif`

### Road Network

Road data is collected from OpenStreetMap using OSMnx.

- OpenStreetMap: https://www.openstreetmap.org/
- OSMnx documentation: https://osmnx.readthedocs.io/
- Outputs:
  - `chennai_roads.graphml`
  - `chennai_roads.geojson`
  - `chennai_nodes.geojson`

### Water Bodies and Drainage

Rivers, lakes, canals, and drainage features are collected from OpenStreetMap.

- OpenStreetMap: https://www.openstreetmap.org/
- Optional India water-resource source: https://indiawris.gov.in/

### Land Use

Land-use information is collected from ESA WorldCover.

- Website: https://esa-worldcover.org/en/data-access
- Output: `chennai_landcover.tif`

### Historical Flood Data

Historical Chennai flood information is collected from OpenCity and related flood-data sources.

- OpenCity Chennai flood dataset: https://data.opencity.in/dataset/chennai-flooding-data
- Copernicus Emergency Management Service: https://emergency.copernicus.eu/data/
- Global Surface Water Explorer: https://global-surface-water.appspot.com/

## Required Libraries

The project uses the following Python libraries:

- `pandas`
- `numpy`
- `geopandas`
- `rasterio`
- `rasterstats`
- `shapely`
- `fiona`
- `pyogrio`
- `osmnx`
- `networkx`
- `scikit-learn`
- `xgboost`
- `shap`
- `folium`
- `streamlit`
- `streamlit-folium`
- `joblib`
- `matplotlib`
- `seaborn`
- `plotly`

## Installation

Clone the repository:

```bash
git clone [https://github.com/PaariKarthikeyan/chennai-flood-project.git](https://github.com/PaariKarthikeyan/chennai-flood-project.git)
cd chennai-flood-project
```

Create a virtual environment:

```bash
python -m venv .venv
```

Activate it on Windows:

```bash
.venv\Scripts\activate
```

Activate it on macOS or Linux:

```bash
source .venv/bin/activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

## Data Processing

Run the scripts in the following order:

```bash
python src/validate_raw_data.py
python src/create_grid.py
python src/process_rainfall.py
python src/process_elevation.py
python src/process_roads.py
python src/process_water_bodies.py
python src/process_land_use.py
python src/process_flood_history.py
python src/combine_features.py
```

The main processed dataset will be created at:

```text
data/processed/grid/chennai_grid_with_features.geojson
```

The model-ready dataset will be created at:

```text
data/final/chennai_flood_training_dataset.csv
```

## Important Features

The final dataset may include:

```text
cell_id
centroid_latitude
centroid_longitude
elevation_mean
elevation_min
elevation_max
slope_mean
slope_min
slope_max
rainfall_24h
rainfall_72h
rainfall_7day
distance_to_water_m
distance_to_river_m
distance_to_drain_m
drainage_length_m
drainage_density
built_up_percentage
water_percentage
vegetation_percentage
cropland_percentage
bare_land_percentage
historical_flood_count
flood_depth_mean
flood_label
```

The target variable is:

```text
flood_label
```

where:

```text
0 = no observed flood
1 = observed flood
```

## Machine-Learning Models

The project uses several models:

### Logistic Regression

Used as an interpretable baseline model.

### Random Forest

Used to model nonlinear relationships between environmental features and flood risk.

### XGBoost

Used as the main high-performance classification model.

### DBSCAN

Used separately for identifying recurring geographical flood hotspots. DBSCAN is not used as the supervised flood-risk prediction model.

## Model Evaluation

The models are evaluated using:

- Precision.
- Recall.
- F1-score.
- ROC-AUC.
- PR-AUC.
- Confusion matrix.

Recall is especially important because missing an actual flood-prone area may be more serious than incorrectly marking a low-risk area as high risk.

Run model training:

```bash
python src/train_models.py
```

Run evaluation:

```bash
python src/evaluate_models.py
```

Model files are saved in:

```text
models/
```

Evaluation results are saved in:

```text
reports/
```

## Flood Hotspot Detection

DBSCAN is applied to historical flood coordinates to identify clusters of recurring flood events.

Run:

```bash
python src/dbscan_hotspots.py
```

The output is saved to:

```text
data/processed/flood_history/flood_hotspots.geojson
```

The output contains:

- Hotspot cluster IDs.
- Noise points.
- Cluster locations.
- Number of observations in each cluster.

## Road Accessibility Analysis

The road-network module assigns predicted flood-risk values to road segments.

The route cost is calculated using:

```text
risk_adjusted_cost =
road_length + flood_probability × risk_penalty
```

This allows the system to compare:

- Shortest route.
- Lower predicted-risk route.
- Route length.
- Estimated flood exposure.
- Number of high-risk road segments.

The system provides lower predicted-risk routes, not guaranteed safe routes.

## Streamlit Application

Run the application:

```bash
streamlit run app/streamlit_app.py
```

The interface provides:

- Interactive Chennai flood-risk map.
- Flood-hotspot layer.
- Risk filtering.
- Start and destination selection.
- Shortest-route option.
- Lower-risk route option.
- Flood-risk summary.
- Basic model explanation.

Streamlit is used to build the interactive data application and can also be deployed through Streamlit Community Cloud. [web:159][web:161]

## Expected Output

The completed project should produce:

- Flood-risk map for Chennai.
- Flood-prone grid cells.
- Historical flood hotspots.
- Risk-adjusted road network.
- Lower predicted-risk routes.
- Trained machine-learning models.
- Model evaluation metrics.
- Explainable feature-importance results.
- Interactive Streamlit dashboard.

## Limitations

- Historical flood records may be incomplete.
- Flood labels may not represent every flooded location.
- NASA POWER rainfall data may initially be available only for a point location.
- Road accessibility predictions are estimates rather than live traffic or emergency information.
- The model does not directly simulate drainage capacity or real-time water depth.
- Prediction quality depends on the quality and spatial accuracy of the historical flood labels.
- A spatially random train-test split may produce overly optimistic results because nearby grid cells are related.

## Future Improvements

- Add real-time rainfall data.
- Add live water-level information.
- Add drainage-capacity data.
- Add traffic data.
- Add satellite-based flood observations for multiple years.
- Add flood-depth prediction.
- Add a mobile-friendly interface.
- Use spatial cross-validation.
- Add real-time alerts.
- Integrate official emergency-management information.

## Reproducibility

To reproduce the project:

1. Place the raw datasets in the required `data/raw/` folders.
2. Install the dependencies from `requirements.txt`.
3. Run the data-validation script.
4. Create the Chennai grid.
5. Run all processing scripts in order.
6. Train and evaluate the models.
7. Run the Streamlit application.

Raw datasets should not be modified. Processed files should be generated from the raw data using the scripts in `src/`.

## License

This project is intended for educational and research purposes.

The datasets belong to their respective providers and may have separate licensing and attribution requirements. Check the license conditions of NASA POWER, OpenStreetMap, ESA WorldCover, OpenCity, Copernicus, and other data providers before redistributing datasets.

## Acknowledgements

This project uses data and tools from:

- NASA POWER.
- OpenStreetMap.
- OSMnx.
- ESA WorldCover.
- SRTM.
- OpenCity.
- Copernicus Emergency Management Service.
- Global Surface Water Explorer.
- GeoPandas.
- Rasterio.
- Scikit-learn.
- XGBoost.
- Streamlit.

## Author

**Paari D K**

- GitHub: https://github.com/PaariKarthikeyan