# recommend_recompute.py

import os
import json
import joblib
import pandas as pd
import sqlite3
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

# === Main Recompute Function ===
def recommend_recompute(user_input):
    model = joblib.load(MODEL_PATH)

    COMFORT_THRESHOLD = activity_thresholds.get(user_input["activity"], 0.70)

    # Infer features from dataset
    features, tier = infer_features(
        apartment_type=user_input["Apartment_Type"],
        zone=user_input["Zone"],
        element_material=f"{user_input['window_material']} and {user_input['wall_material']}",
        floor_level=user_input.get("Floor_Level")
    )

    # Predict comfort score
    try:
        X = pd.DataFrame([features])
        comfort_score = model.predict(X)[0]
    except:
        comfort_score = None

    # Compliance check
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
            "LAeq": f"{features.get('laeq_db')} dB ≤ {compliance['LAeq_max']}",
            "RT60": f"{features.get('rt60_s')} s ≤ {compliance['RT60_max']}",
        },
        "recommendations": {},
        "improved_score": None
    }

    # Verbose guidance
    if not compliance["LAeq"] or not compliance["RT60"]:
        with open(GUIDANCE_JSON) as f:
            guidance = json.load(f)
        if not compliance["LAeq"]:
            result["recommendations"]["LAeq"] = guidance["LAeq_non_compliant"]["general_recommendations"]
        if not compliance["RT60"]:
            result["recommendations"]["RT60"] = guidance["RT60_non_compliant"]["general_recommendations"]

    return result
