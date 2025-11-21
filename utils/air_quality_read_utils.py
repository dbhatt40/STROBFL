import os
import glob
import pandas as pd
import numpy as np

# === 1. Path configuration ===
# You are running the script from ./your_script/
# The data is in ../data/air_quality/
DATA_DIR = os.path.join("..", "data", "air_quality")
OUTPUT_CSV = "AirQuality_Clean.csv"
NA_MARKERS = ["No", "no", "NA", "N/A", "NaN", "", " "]

NUMERIC_COLS = [
    "PM2.5", "PM10", "SO2", "NO2", "CO", "O3",
    "TEMP", "PRES", "DEWP", "RAIN", "WSPM"
]

def read_and_clean_one_file(path):
    # --- Read the CSV and handle missing markers ---
    df = pd.read_csv(path, na_values=NA_MARKERS)

    # Infer station name from filename if missing
    if "station" not in df.columns:
        station_name = os.path.basename(path).split("_")[2]
        df["station"] = station_name

    # Convert year/month/day/hour to numeric
    for col in ["year", "month", "day", "hour"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    # Drop useless 'No' column if present
    df = df.drop(columns=["No"], errors="ignore")

    # --- Create datetime column ---
    df["datetime"] = pd.to_datetime(
        dict(year=df["year"], month=df["month"], day=df["day"], hour=df["hour"]),
        errors="coerce"
    )
    # Convert all numeric columns safely
    for col in NUMERIC_COLS:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
	    # Drop rows missing datetime or key measurements
    df = df.dropna(subset=["datetime"])
    df = df.dropna(subset=[c for c in NUMERIC_COLS if c in df.columns])
    df = df.drop(columns=["year", "month", "day", "hour"], errors="ignore")
	
    df = encode_wind_direction(df)

    # Sort for time alignment
	
    # df = df.sort_values(["station_id", "datetime"]).reset_index(drop=True)
    return df

def encode_wind_direction(df, col='wd'):
    """
    Encode wind direction column into two numeric cyclic features: wd_sin, wd_cos.
    Handles missing or 'C' (calm) values.
    """
    # Define angle mapping (in degrees)
    wd_to_deg = {
        "N": 0, "NNE": 22.5, "NE": 45, "ENE": 67.5,
        "E": 90, "ESE": 112.5, "SE": 135, "SSE": 157.5,
        "S": 180, "SSW": 202.5, "SW": 225, "WSW": 247.5,
        "W": 270, "WNW": 292.5, "NW": 315, "NNW": 337.5,
        "C": np.nan  # Calm / variable wind
    }

    if col not in df.columns:
        print(f"Column '{col}' not found; skipping wind encoding.")
        return df

    # Map to degrees
    df['wd_deg'] = df[col].map(wd_to_deg)

    # Convert to radians
    radians = np.deg2rad(df['wd_deg'])

    # Compute sin/cos
    df['wd_sin'] = np.sin(radians)
    df['wd_cos'] = np.cos(radians)

    # Drop original string column (optional)
    df = df.drop(columns=[col])
    return df

def replace_station_with_id(df, col='station'):
	
    # df = pd.DataFrame({
    # "station": ["Aotizhongxin", "Changping", "Dongsi", "Aotizhongxin", "Dongsi"]
    # })
	s = df['station'].astype(str).str.strip()
	stations = s.unique()
	stations.sort()
# 	print("Stations", stations)
	station_map = {name: idx for idx, name in enumerate(stations)}
# 	print("Station map (first few):", dict(list(station_map.items())[:3]))
	df["station_id"] = df["station"].map(station_map)
   
	# Drop original string column (optional)

	return df
	

def main():
    # --- 2. Find all CSVs starting with PRSA_Data ---
    files = glob.glob(os.path.join(DATA_DIR, "PRSA_Data*.csv"))
    if not files:
        raise FileNotFoundError(f"No PRSA_Data files found in {DATA_DIR}")
    print(f"Found {len(files)} files.")

    # --- 3. Read and combine all stations ---
    frames = [read_and_clean_one_file(f) for f in files]
    df_all = pd.concat(frames, ignore_index=True)
    df_all_stationids = replace_station_with_id(df_all)
    df_all_stationids = df_all_stationids.drop(columns=['station'])
    df_all_stationids = df_all_stationids.sort_values(["station_id", "datetime"]).reset_index(drop=True)

    # --- 4. Save final cleaned dataset ---
    output_dir = os.path.join(DATA_DIR,OUTPUT_CSV)
    df_all_stationids.to_csv(output_dir, index=False)
    print(f"✅ Cleaned and combined dataset saved to {OUTPUT_CSV}")
    print("Final shape:", df_all.shape)

if __name__ == "__main__":
    main()
