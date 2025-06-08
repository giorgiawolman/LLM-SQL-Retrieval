import pandas as pd

def infer_features(apartment_type, zone, element, wall_material=None, window_material=None, floor_level=None):
    df = pd.read_csv("sql/cleaned_dataset.csv")

    material_kw = wall_material if element.lower() == "wall" else window_material

    # === Tier 1: Full match by element and material ===
    match = df[
        (df["Apartment_Type"] == apartment_type) &
        (df["Zone"] == zone) &
        (df["Element"].str.lower() == element.lower()) &
        (df["Material"].str.contains(material_kw or "", case=False, na=False))
    ]
    if not match.empty:
        print("✅ Tier 1: Exact match on zone, apartment type, element, and material.")
        features = match.iloc[0].to_dict()
        tier = "Tier 1"
    else:
        # === Tier 2: Match by apartment type, zone, and element only ===
        match = df[
            (df["Apartment_Type"] == apartment_type) &
            (df["Zone"] == zone) &
            (df["Element"].str.lower() == element.lower())
        ]
        if not match.empty:
            print("⚠️ Tier 2: Fallback on zone, apartment type, and element.")
            features = match.iloc[0].to_dict()
            tier = "Tier 2"
        else:
            # === Tier 3: Match by apartment type and element only ===
            match = df[
                (df["Apartment_Type"] == apartment_type) &
                (df["Element"].str.lower() == element.lower())
            ]
            if not match.empty:
                print("⚠️ Tier 3: Fallback on apartment type and element.")
                features = match.iloc[0].to_dict()
                tier = "Tier 3"
            else:
                # === Final fallback: dataset average values or zone average for Laeq ===
                print("⚠️ Tier 4: Final fallback using dataset averages.")
                means = df.mean(numeric_only=True).to_dict()

                # Try zone-specific average LAeq if available
                laeq_zone = df[df["Zone"] == zone]["Laeq"].mean()
                if pd.isna(laeq_zone):
                    laeq_zone = means.get("Laeq", 55.0)

                features = {
                    "Zone": zone,
                    "Apartment_Type": apartment_type,
                    "Element": element,
                    "Laeq": laeq_zone,
                    "SPL": means.get("SPL", 0.35),
                    "RT60(seconds)": means.get("RT60(seconds)", 0.5),
                    "RT60 (material ac)": means.get("RT60 (material ac)", 0.5),
                    "Surface_Area(m)": means.get("Surface_Area(m)", 10),
                    "Height": means.get("Height", 2.8),
                    "Absortion_Coefficient": means.get("Absortion_Coefficient", 0.1),
                    "Facade_Dampening(Score)": means.get("Facade_Dampening(Score)", 7.5)
                }
                tier = "Tier 4"

    # === Override height if floor level provided ===
    if floor_level is not None:
        features["Height"] = round(floor_level * 3.0, 2)
        features["Floor_Level"] = floor_level

    return features, tier
