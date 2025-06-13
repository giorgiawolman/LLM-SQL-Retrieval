# recommend_recompute.py

import os
import json
import joblib
import pandas as pd
from utils.infer_from_inputs import infer_features

# === Paths ===
MODEL_PATH = "model/ecoform_acoustic_comfort_model.pkl"
COMPLIANCE_JSON = "knowledge/compliance_thresholds_extended.json"
GUIDANCE_JSON = "knowledge/compliance_guidance.json"
DATA_PATH = "sql/Ecoform_Dataset_v1.csv"

# === Activity Thresholds ===
activity_thresholds = {
    "Sleeping": 0.85, "Working": 0.75, "Learning": 0.80, "Living": 0.70,
    "Healing": 0.80, "Co-working": 0.75, "Exercise": 0.60, "Dining": 0.65
}

# === Compliance Check ===
def check_compliance(activity, laeq, rt60):
    with open(COMPLIANCE_JSON) as f:
        thresholds = json.load(f)
    for entry in thresholds:
        if entry["use"].lower() == activity.lower():
            return {
                "LAeq": laeq <= entry["LAeq_max"],
                "LAeq_max": entry["LAeq_max"],
                "RT60": rt60 <= entry["RT60_max"],
                "RT60_max": entry["RT60_max"],
                "source": entry["source"]
            }
    return {"LAeq": None, "RT60": None, "LAeq_max": None, "RT60_max": None, "source": "N/A"}

# === Main Function ===
def recommend_recompute(user_input):
    model = joblib.load(MODEL_PATH)
    comfort_threshold = activity_thresholds.get(user_input.get("activity", "Living"), 0.70)

    # Reference input structure from dataset
    reference_df = pd.read_csv(DATA_PATH)
    input_cols = [col for col in reference_df.columns if "comfort" not in col]

    # === Step 1: Initial Inference ===
    features, tier = infer_features(
        apartment_type=user_input["Apartment_Type"],
        zone=user_input["Zone"],
        element=user_input.get("Element"),
        element_material=f"{user_input['window_material']} and {user_input['wall_material']}",
        floor_level=user_input.get("Floor_Level")
    )

    row = pd.DataFrame([{col: features.get(col, 0) for col in input_cols}])
    try:
        row = row.fillna(0)
        comfort_score = model.predict(row)[0]
    except Exception as e:
        print(f"❌ Model prediction failed: {e}")
        comfort_score = None

    compliance = check_compliance(
        user_input["activity"],
        features.get("laeq_db", 0),
        features.get("rt60_s", 0)
    )

    result = {
        "comfort_score": round(comfort_score, 3) if comfort_score is not None else None,
        "source": tier,
        "compliance": {
            "status": "✅ Compliant" if compliance["LAeq"] and compliance["RT60"] else "❌ Not Compliant",
            "reason": f"Compared against {compliance['source']}",
            "LAeq": f"{features.get('laeq_db')} dB (Non-compliant, should be ≤ {compliance['LAeq_max']})"
                    if not compliance["LAeq"] else f"{features.get('laeq_db')} dB (Compliant)",
            "RT60": f"{features.get('rt60_s')} s (Non-compliant, should be ≤ {compliance['RT60_max']})"
                    if not compliance["RT60"] else f"{features.get('rt60_s')} s (Compliant)"
        },
        "recommendations": {},
        "material_swap": {},
        "improved_score": None,
        "best_materials": {},
        "best_score": None
    }

    # === Step 2: Generate Recommendations ===
    if not compliance["LAeq"] or not compliance["RT60"]:
        with open(GUIDANCE_JSON) as f:
            guidance = json.load(f)
        if not compliance["LAeq"]:
            result["recommendations"]["LAeq"] = guidance["LAeq_non_compliant"]["general_recommendations"]
        if not compliance["RT60"]:
            result["recommendations"]["RT60"] = guidance["RT60_non_compliant"]["general_recommendations"]

        # === Step 3: Try Material Swaps ===
        wall_options = ["Acoustic Panel", "Brick", "Timber Insulation"]
        window_options = ["Triple Glazing", "Double Pane Glass", "Acoustic Glazing"]

        for new_wall in wall_options:
            for new_window in window_options:
                features_try, _ = infer_features(
                    apartment_type=user_input["Apartment_Type"],
                    zone=user_input["Zone"],
                    element=user_input.get("Element"),
                    element_material=f"{new_window} and {new_wall}",
                    floor_level=user_input.get("Floor_Level")
                )
                row_try = pd.DataFrame([{col: features_try.get(col, 0) for col in input_cols}])
                try:
                    row_try = row_try.fillna(0)
                    score_try = model.predict(row_try)[0]
                    if score_try >= comfort_threshold:
                        result["material_swap"] = {
                            "wall_material": new_wall,
                            "window_material": new_window
                        }
                        result["improved_score"] = round(score_try, 3)
                        result["best_materials"] = result["material_swap"]
                        result["best_score"] = result["improved_score"]
                        return result
                except:
                    continue

    return result
