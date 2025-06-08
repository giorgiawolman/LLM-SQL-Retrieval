import pandas as pd
import re

# === Column Cleaner ===
def clean_col(col):
    col = col.strip().lower()
    col = re.sub(r'[():]', '', col)
    col = re.sub(r'\s+', '_', col)
    return col

# === Tiered Feature Inference Function ===
def infer_features(apartment_type, zone, element=None, element_material=None, floor_level=None):
    """
    Infers the best-matching acoustic dataset entry given partial or full user input.

    Args:
        apartment_type (str): e.g. "1Bed", "2Bed"
        zone (str): e.g. "HD-Urban-V1"
        element (str): Room use or function (e.g. "Living", "Sleeping") – optional
        element_material (str): Material keyword (e.g. "concrete", "single glazing")
        floor_level (int): Optional floor level (used to compute floor_height_m)

    Returns:
        features (dict): Matched or inferred feature row
        tier (str): Match strength description (Tier 1 to Tier 4)
    """

    # === Load and clean dataset ===
    df = pd.read_csv("sql/Ecoform_Dataset_v1.csv")
    df.columns = [clean_col(col) for col in df.columns]

    # === Normalize inputs ===
    apartment_type = apartment_type.lower()
    zone = zone.lower()
    material_kw = element_material.lower() if element_material else ""
    element_kw = element.lower() if element else ""

    # === Define keys ===
    apt_col = "apartment_type_string"
    zone_col = "zone_string"
    material_col = "element_materials_string"

    # === Tier 1: Match apartment + zone + material keyword + element keyword ===
    match = df[
        (df[apt_col].str.lower() == apartment_type) &
        (df[zone_col].str.lower() == zone) &
        (df[material_col].str.lower().str.contains(material_kw)) &
        (df[material_col].str.lower().str.contains(element_kw))
    ]
    if not match.empty:
        features = match.iloc[0].to_dict()
        tier = "Tier 1"
        print("✅ Tier 1: Match on apartment, zone, material, and element.")

    else:
        # === Tier 2: Match apartment + zone + material keyword ===
        match = df[
            (df[apt_col].str.lower() == apartment_type) &
            (df[zone_col].str.lower() == zone) &
            (df[material_col].str.lower().str.contains(material_kw))
        ]
        if not match.empty:
            features = match.iloc[0].to_dict()
            tier = "Tier 2"
            print("⚠️ Tier 2: Match on apartment, zone, and material.")
        else:
            # === Tier 3: Match apartment + zone only ===
            match = df[
                (df[apt_col].str.lower() == apartment_type) &
                (df[zone_col].str.lower() == zone)
            ]
            if not match.empty:
                features = match.iloc[0].to_dict()
                tier = "Tier 3"
                print("⚠️ Tier 3: Match on apartment and zone.")
            else:
                # === Tier 4: Use dataset mean fallback ===
                print("⚠️ Tier 4: Using dataset average values.")
                means = df.mean(numeric_only=True).to_dict()
                features = {
                    apt_col: apartment_type,
                    zone_col: zone,
                    material_col: f"{element_kw}: {material_kw}",
                    **means
                }
                tier = "Tier 4"

    # === Add derived floor height ===
    if floor_level is not None:
        features["floor_height_m"] = round(floor_level * 3.0, 2)
        features["floor_level"] = floor_level

    return features, tier
