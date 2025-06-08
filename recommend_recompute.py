import sys
import os
import sqlite3
import pandas as pd
import joblib
import json

# Add project root to system path so 'utils' works
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from utils.infer_from_inputs import infer_features

# --- Config ---
MODEL_PATH = "model/ecoform_acoustic_comfort_model.pkl"
MATERIAL_DB_PATH = "sql/material-database.db"
COMPLIANCE_JSON = "knowledge/compliance_thresholds_extended.json"
GUIDANCE_JSON = "knowledge/compliance_guidance.json"

# --- Main Function ---
def recommend_recompute(user_input):
    activity_thresholds = {
        "Sleeping": 0.85, "Working": 0.75, "Learning": 0.80, "Living": 0.70,
        "Healing": 0.80, "Co-working": 0.75, "Exercise": 0.60, "Dining": 0.65
    }
    activity = user_input.get("activity", "Living")
    COMFORT_THRESHOLD = activity_thresholds.get(activity, 0.70)

    # --- Load Model ---
    model = joblib.load(MODEL_PATH)

    # --- Infer Features ---
    features, tier = infer_features(
        apartment_type=user_input["Apartment_Type"],
        zone=user_input["Zone"],
        element_keyword=user_input["Element"],
        material_keyword=user_input["wall_material"],
        floor_level=user_input["Floor_Level"]
    )
    features["material"] = user_input["wall_material"]

    # --- Material Lookup ---
    def get_absorption_from_db(material_name, element):
        conn = sqlite3.connect(MATERIAL_DB_PATH)
        query = """
            SELECT Absorption_Coefficient_500Hz 
            FROM material_knowledge 
            WHERE LOWER(material) = ? AND LOWER(Element) = ?
        """
        result = conn.execute(query, (material_name.lower(), element.lower())).fetchone()
        conn.close()
        return result[0] if result else None

    wall_abs = get_absorption_from_db(user_input["wall_material"], "wall")
    window_abs = get_absorption_from_db(user_input["window_material"], "window")

    if wall_abs is not None:
        features["absortion_coefficient"] = wall_abs
    elif window_abs is not None:
        features["absortion_coefficient"] = window_abs

    # --- Predict Comfort Score ---
    X = pd.DataFrame([features])
    try:
        original_score = model.predict(X)[0]
    except Exception:
        volume = features["total_surface_sqm"] * features["floor_height_m"]
        rt60 = 0.161 * volume / (features["absortion_coefficient"] * features["total_surface_sqm"] + 1e-5)
        features["rt60_s"] = rt60
        features["rt60_material_ac"] = rt60
        original_score = max(0, min(1, 1.2 - 0.01 * features["laeq_db"] - 0.3 * rt60 + 0.015 * features["facade_dampening_score"]))

    # --- Compliance Check ---
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

    compliance = check_compliance(activity, features["laeq_db"], features["rt60_s"])
    reason_string = f"LAeq = {features['laeq_db']} vs {compliance['LAeq_max']}, RT60 = {features['rt60_s']} vs {compliance['RT60_max']}"

    # --- Suggest Better Materials ---
    def suggest_better_materials(element, current_abs):
        conn = sqlite3.connect(MATERIAL_DB_PATH)
        query = """
            SELECT material, Absorption_Coefficient_500Hz 
            FROM material_knowledge 
            WHERE LOWER(Element) = ? AND Absorption_Coefficient_500Hz > ?
            ORDER BY Absorption_Coefficient_500Hz DESC 
            LIMIT 3
        """
        rows = conn.execute(query, (element.lower(), current_abs)).fetchall()
        conn.close()
        return rows

    best_score = original_score
    best_material = None
    best_abs = wall_abs or window_abs

    for element_type, current_abs in [("wall", wall_abs), ("window", window_abs)]:
        if current_abs is None:
            continue
        alternatives = suggest_better_materials(element_type, current_abs)
        for alt_material, alt_abs in alternatives:
            test_features = features.copy()
            test_features["absortion_coefficient"] = alt_abs
            test_features["material"] = alt_material
            try:
                alt_score = model.predict(pd.DataFrame([test_features]))[0]
            except:
                volume = test_features["total_surface_sqm"] * test_features["floor_height_m"]
                rt60 = 0.161 * volume / (alt_abs * test_features["total_surface_sqm"] + 1e-5)
                alt_score = max(0, min(1, 1.2 - 0.01 * test_features["laeq_db"] - 0.3 * rt60 + 0.015 * test_features["facade_dampening_score"]))
            if alt_score > best_score:
                best_score = alt_score
                best_material = alt_material
                best_abs = alt_abs

    # --- Return Result Object ---
    return {
        "comfort_score": round(original_score, 3),
        "source": f"model ({tier})",
        "compliance": {
            "status": "compliant" if original_score >= COMFORT_THRESHOLD else "non-compliant",
            "reason": reason_string,
            "thresholds": compliance,
        },
        "recommendations": {
            "material_upgrade": best_material if best_material else None,
            "absorption_score": best_abs if best_abs else None,
        },
        "improved_score": round(best_score, 3) if best_score > original_score else None
    }
