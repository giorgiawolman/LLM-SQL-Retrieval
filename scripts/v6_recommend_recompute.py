import sys
import os
import sqlite3
import pandas as pd
import joblib
import json
import re

# Add project root to system path so 'utils' works
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from utils.infer_from_inputs_v1 import infer_features

# --- Config ---
MODEL_PATH = "model/ecoform_acoustic_comfort_model.pkl"
MATERIAL_DB_PATH = "sql/material-database.db"
COMPLIANCE_JSON = "knowledge/compliance_thresholds_extended.json"
GUIDANCE_JSON = "knowledge/compliance_guidance.json"

VERBOSE = True

# --- User Input ---
user_input = {
    "Apartment_Type": "3Bed",
    "Zone": "HD-Urban-V1",
    "Element": "Wall, window",
    "Floor_Level": 1,
    "wall_material": "Painted Brick",
    "window_material": "Double Glazing",
    "activity": "Living",
}

activity_thresholds = {
    "Sleeping": 0.85, "Working": 0.75, "Learning": 0.80, "Living": 0.70,
    "Healing": 0.80, "Co-working": 0.75, "Exercise": 0.60, "Dining": 0.65
}
COMFORT_THRESHOLD = activity_thresholds.get(user_input["activity"], 0.70)

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

# Lookup absorption for wall and window
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

compliance = check_compliance(user_input["activity"], features["laeq_db"], features["rt60_s"])

# --- Suggest Material Upgrade ---
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

# Evaluate both wall and window alternatives
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

# --- OUTPUT CLEANLY ---
print("===== COMFORT ANALYSIS =====")
print(f"Comfort Score: {round(original_score, 3)}")
print(f"Target Threshold: {COMFORT_THRESHOLD}")
print(f"Result: {'✅ Acceptable' if original_score >= COMFORT_THRESHOLD else '❌ Not Acceptable'}")
print(f"Prediction Tier: {tier}")
print(f"Wall Material: {user_input['wall_material']} (Abs: {wall_abs})")
print(f"Window Material: {user_input['window_material']} (Abs: {window_abs})")
if best_material:
    print(f"Suggested Upgrade: {best_material} (Absorption: {best_abs}) → Score: {round(best_score, 3)}")

print("\n===== COMPLIANCE CHECK =====")
print(f"Activity Type: {user_input['activity']}")
print(f"LAeq: {features['laeq_db']} dB vs Max {compliance['LAeq_max']} → {'✅ Compliant' if compliance['LAeq'] else '❌ Not compliant'}")
print(f"RT60: {features['rt60_s']} s vs Max {compliance['RT60_max']} → {'✅ Compliant' if compliance['RT60'] else '❌ Not compliant'}")
print(f"Standard: {compliance['source']}")

# --- Optional Guidance ---
if VERBOSE:
    with open(GUIDANCE_JSON) as f:
        guidance = json.load(f)

    if not compliance["LAeq"]:
        print("\n⚠ LAeq exceeds limits.")
        print(guidance["LAeq_non_compliant"]["description"])
        for rec in guidance["LAeq_non_compliant"]["general_recommendations"]:
            print(f" - {rec}")
        if user_input["Floor_Level"] == 1:
            print(" - Use green buffers on ground level.")
        else:
            print(" - Improve window glazing for higher floors.")

    if not compliance["RT60"]:
        print("\n⚠ RT60 exceeds limits.")
        print(guidance["RT60_non_compliant"]["description"])
        for rec in guidance["RT60_non_compliant"]["general_recommendations"]:
            print(f" - {rec}")
