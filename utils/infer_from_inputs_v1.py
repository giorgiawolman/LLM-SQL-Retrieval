import pandas as pd
import re

def clean_col(col):
    col = col.strip().lower()
    col = re.sub(r'[():]', '', col)
    col = re.sub(r'\s+', '_', col)
    return col

def infer_features(apartment_type, zone, element_keyword=None, material_keyword=None, floor_level=None):
    # Load and clean dataset
    df = pd.read_csv("sql/Ecoform_Dataset_v1.csv")
    df.columns = [clean_col(col) for col in df.columns]

    # Normalize inputs
    apartment_type = apartment_type.lower()
    zone = zone.lower()
    material_kw = material_keyword.lower() if material_keyword else ""
    element_kw = element_keyword.lower() if element_keyword else ""

    # Column keys
    apt_col = "apartment_type_string"
    zone_col = "zone_string"
    material_col = "element_materials_string"

    # === Tier 1: Match all key descriptors ===
    match = df[
        (df[apt_col].str.lower() == apartment_type) &
        (df[zone_col].str.lower() == zone) &
        (df[material_col].str.lower().str.contains(material_kw)) &
        (df[material_col].str.lower().str.contains(element_kw))
    ]
    if not match.empty:
        features = match.iloc[0].to_dict()
        tier = "Tier 1"
        print("✅ Tier 1: Match on apartment, zone, element, and material.")
    else:
        # === Tier 2: Match apartment + zone only ===
        match = df[
            (df[apt_col].str.lower() == apartment_type) &
            (df[zone_col].str.lower() == zone)
        ]
        if not match.empty:
            features = match.iloc[0].to_dict()
            tier = "Tier 2"
            print("⚠️ Tier 2: Match on apartment and zone.")
        else:
            # === Tier 3: Match apartment only ===
            match = df[df[apt_col].str.lower() == apartment_type]
            if not match.empty:
                features = match.iloc[0].to_dict()
                tier = "Tier 3"
                print("⚠️ Tier 3: Match on apartment type.")
            else:
                # === Tier 4: Use mean fallback ===
                print("⚠️ Tier 4: Using dataset average values.")
                means = df.mean(numeric_only=True).to_dict()
                features = {
                    apt_col: apartment_type,
                    zone_col: zone,
                    material_col: f"{element_kw}: {material_kw}",
                    **means
                }
                tier = "Tier 4"

    # Inject calculated height from floor level if available
    if floor_level is not None:
        features["floor_height_m"] = round(floor_level * 3.0, 2)
        features["floor_level"] = floor_level

    return features, tier
