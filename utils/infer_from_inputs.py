import pandas as pd
import re
import json
import os

def clean_col(col):
    col = col.strip().lower()
    col = re.sub(r'[():]', '', col)
    col = re.sub(r'\s+', '_', col)
    return col

def infer_features(apartment_type, zone, element=None, element_material=None, floor_level=None):
    df = pd.read_csv("sql/Ecoform_Dataset_v1.csv")
    df.columns = [clean_col(col) for col in df.columns]

    apt_col, zone_col, mat_col = "apartment_type_string", "zone_string", "element_materials_string"

    apartment_type = apartment_type.strip().lower().replace(" ", "").replace("-", "")
    zone = zone.strip().lower()
    mat_kw = element_material.lower() if element_material else ""
    el_kw = element.lower() if element else ""

    df[apt_col] = df[apt_col].str.lower().str.replace(" ", "").str.replace("-", "")
    df[zone_col] = df[zone_col].str.lower()

    # Tiered match
    conditions = [
        ((df[apt_col] == apartment_type) & (df[zone_col] == zone) &
         df[mat_col].str.lower().str.contains(mat_kw) & df[mat_col].str.lower().str.contains(el_kw)),
        ((df[apt_col] == apartment_type) & (df[zone_col] == zone) &
         df[mat_col].str.lower().str.contains(mat_kw)),
        ((df[apt_col] == apartment_type) & (df[zone_col] == zone)),
        (df[apt_col] == apartment_type)
    ]
    tiers = ["Tier 1", "Tier 2", "Tier 3", "Tier 4"]

    for cond, tier in zip(conditions, tiers):
        match = df[cond]
        if not match.empty:
            features = match.iloc[0].to_dict()
            print(f"{'✅' if tier == 'Tier 1' else '⚠️'} {tier}: Match on apartment and zone.")
            break
    else:
        print("⚠️ Tier 5: Using dataset average values.")
        features = df.mean(numeric_only=True).to_dict()
        features.update({apt_col: apartment_type, zone_col: zone, mat_col: f"{el_kw}: {mat_kw}"})
        tier = "Tier 5"

    if floor_level is not None:
        features["floor_height_m"] = round(floor_level * 3.0, 2)
        features["floor_level"] = floor_level

    # ✅ Filter to model-compatible columns
    model_feature_path = "model/ecoform_model_features.json"
    if os.path.exists(model_feature_path):
        with open(model_feature_path) as f:
            allowed = json.load(f)
        features = {k: v for k, v in features.items() if k in allowed}
    else:
        print("⚠️ Warning: model feature list not found, no filtering applied.")

    return features, tier
