import pandas as pd
import numpy as np
from pathlib import Path
import os

def detect_columns(df):
    """Automatically detect the date and rainfall columns based on common naming conventions."""
    df.columns = df.columns.str.strip().str.lower()
    
    date_col = None
    rain_col = None
    
    # Check for a single date column
    date_candidates = ['date', 'time', 'datetime', 'yyyy-mm-dd', 'timestamp']
    for col in date_candidates:
        if col in df.columns:
            date_col = col
            break
            
    # Check for split Year/Month/Day columns (common in NASA POWER)
    if not date_col and set(['year', 'mo', 'dy']).issubset(df.columns):
        df['date'] = pd.to_datetime(dict(year=df['year'], month=df['mo'], day=df['dy']))
        date_col = 'date'
    elif not date_col and set(['year', 'month', 'day']).issubset(df.columns):
        df['date'] = pd.to_datetime(df[['year', 'month', 'day']])
        date_col = 'date'

    # Check for rainfall/precipitation column
    rain_candidates = ['prectotcorr', 'prectot', 'rainfall', 'precip', 'precipitation', 'rain', 'rainfall_mm']
    for col in df.columns:
        if any(candidate in col for candidate in rain_candidates):
            rain_col = col
            break

    return date_col, rain_col, df

def process_rainfall():
    raw_file = Path("data/raw/rainfall/chennai_rainfall.csv")
    out_dir = Path("data/processed/rainfall")
    out_dir.mkdir(parents=True, exist_ok=True)
    
    if not raw_file.exists():
        print(f"ERROR: {raw_file} not found. Cannot process rainfall.")
        return

    print(f"Reading {raw_file}...")
    # NASA POWER files sometimes have metadata headers. We attempt to read normally, 
    # but if it fails, one might need to use skiprows. Assuming clean CSV for now.
    df = pd.read_csv(raw_file)
    
    # 1. Detect columns
    date_col, rain_col, df = detect_columns(df)
    
    if not date_col or not rain_col:
        print(f"ERROR: Could not automatically detect date ({date_col}) or rainfall ({rain_col}) columns.")
        print(f"Available columns: {list(df.columns)}")
        return
        
    print(f"Detected Date column: '{date_col}', Rainfall column: '{rain_col}'")
    
    # 2. Rename and format columns
    df = df.rename(columns={date_col: 'date', rain_col: 'rainfall_mm'})
    df['date'] = pd.to_datetime(df['date'], errors='coerce')
    df['rainfall_mm'] = pd.to_numeric(df['rainfall_mm'], errors='coerce')
    
    # Drop rows with invalid dates
    df = df.dropna(subset=['date']).sort_values('date').reset_index(drop=True)
    
    # 3. Handle missing/invalid rainfall values
    # NASA POWER uses -999 for missing data. Convert negatives to NaN, then fill with 0.
    invalid_mask = df['rainfall_mm'] < 0
    num_invalid = invalid_mask.sum()
    if num_invalid > 0:
        print(f"Replacing {num_invalid} negative/invalid rainfall values with 0.")
        df.loc[invalid_mask, 'rainfall_mm'] = np.nan
        
    num_missing = df['rainfall_mm'].isna().sum()
    if num_missing > 0:
        print(f"Filling {num_missing} missing rainfall values with 0.")
        df['rainfall_mm'] = df['rainfall_mm'].fillna(0)

    # Cleaned base dataframe
    df_clean = df[['date', 'rainfall_mm']].copy()
    
    # 4. Create Features (Rolling sums for antecedent moisture)
    print("Calculating rolling rainfall features...")
    # Because data is daily, 24h is the daily value, 72h is 3 days.
    df_clean['rainfall_24h'] = df_clean['rainfall_mm']
    df_clean['rainfall_3day'] = df_clean['rainfall_mm'].rolling(window=3, min_periods=1).sum()
    df_clean['rainfall_72h'] = df_clean['rainfall_3day'] # Alias for compatibility
    df_clean['rainfall_7day'] = df_clean['rainfall_mm'].rolling(window=7, min_periods=1).sum()
    df_clean['rainfall_30day'] = df_clean['rainfall_mm'].rolling(window=30, min_periods=1).sum()
    
    # Round to 2 decimal places to keep data clean
    feature_cols = ['rainfall_mm', 'rainfall_24h', 'rainfall_3day', 'rainfall_72h', 'rainfall_7day', 'rainfall_30day']
    df_clean[feature_cols] = df_clean[feature_cols].round(2)
    
    # 5. Save outputs
    cleaned_file = out_dir / "rainfall_cleaned.csv"
    features_file = out_dir / "rainfall_features.csv"
    summary_file = out_dir / "rainfall_summary.csv"
    
    # Save cleaned
    df_clean[['date', 'rainfall_mm']].to_csv(cleaned_file, index=False)
    
    # Save features
    df_clean.to_csv(features_file, index=False)
    
    # Save summary stats
    summary = df_clean[feature_cols].describe().round(2)
    summary.to_csv(summary_file)
    
    print(f"Success! Processed {len(df_clean)} daily rainfall records.")
    print(f"Outputs saved to {out_dir}/")

if __name__ == "__main__":
    process_rainfall()