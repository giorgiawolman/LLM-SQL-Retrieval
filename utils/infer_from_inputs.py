import pandas as pd

def infer_features(apt_type, zone, wall_material, window_material):
    df = pd.read_csv("sql/updated_acoustic_dataset_2000_samples.csv")

    # Tier 1: Full match on all input parameters
    match = df[
        (df["Apt Type"] == apt_type) &
        (df["ZONE"] == zone) &
        (df["Wall Material"] == wall_material) &
        (df["Window Material"] == window_material)
    ]
    if not match.empty:
        print("Using exact match (zone, materials, and apartment type).")
        return match.iloc[0].to_dict()

    # Tier 2: Match on apartment type and zone
    match = df[
        (df["Apt Type"] == apt_type) &
        (df["ZONE"] == zone)
    ]
    if not match.empty:
        print("Fallback to match by apartment type and zone.")
        return match.iloc[0].to_dict()

    # Tier 3: Match on apartment type only
    match = df[df["Apt Type"] == apt_type]
    if not match.empty:
        print("Fallback to match by apartment type only.")
        return match.iloc[0].to_dict()

    # Final fallback: use average of the dataset
    if not df.empty:
        print("Fallback to average of dataset.")
        return df.mean(numeric_only=True).to_dict()

    # No data available
    raise ValueError("No matching data or fallback options found.")
