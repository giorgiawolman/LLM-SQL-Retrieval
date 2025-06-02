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
MODEL_PATH = "model/acoustic_comfort_score_model.pkl"
MATERIAL_DB_PATH = "sql/material-database.db"
COMPLIANCE_JSON = "knowledge/compliance_thresholds_extended.json"
GUIDANCE_JSON = "knowledge/compliance_guidance.json"

VERBOSE = True  # Toggle for detailed printouts

# --- User Input ---
user_input = {
    "Apartment_Type": "2Bed",
    "Zone": "HD-Urban-V1",
    "Element": "Wall",
    "Floor_Level": 1,
    "wall_material": "Painted Brick",
    "window_material": "Double Pane Glass",
    "activity": "Living",
    "Laeq": 57
}

# --- Activity-Based Comfort Threshold ---
activity_thresholds = {
    "Sleeping": 0.85,
    "Working": 0.75,
    "Learning": 0.80,
    "Living": 0.70,
    "Healing": 0.80,
    "Co-working": 0.75,
    "Exercise": 0.60,
    "Dining": 0.65
}
COMFORT_THRESHOLD = activity_thresholds.get(user_input["activity"], 0.70)

# --- Load Model ---
model = joblib.load(MODEL_PATH)

# --- Infer Features ---
features, tier = infer_features(
    apartment_type=user_input["Apartment_Type"],
    zone=user_input["Zone"],
    element=user_input["Element"],
    wall_material=user_input["wall_material"],
    window_material=user_input["window_material"],
    floor_level=user_input["Floor_Level"]
)
features["Material"] = user_input["wall_material"]
features["Laeq"] = user_input["Laeq"]

# --- Absorption Lookup ---
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

material_key = user_input["wall_material"] if user_input["Element"].lower() == "wall" else user_input["window_material"]
current_abs = get_absorption_from_db(material_key, user_input["Element"])
if current_abs is not None:
    features["Absortion_Coefficient"] = current_abs

# --- Model Prediction ---
X = pd.DataFrame([features])
try:
    original_score = model.predict(X)[0]
except Exception:
    volume = features["Surface_Area(m)"] * features["Height"]
    rt60 = 0.161 * volume / (features["Absortion_Coefficient"] * features["Surface_Area(m)"] + 1e-5)
    features["RT60(seconds)"] = rt60
    features["RT60 (material ac)"] = rt60
    original_score = max(0, min(1, 1.2 - 0.01 * features["Laeq"] - 0.3 * rt60 + 0.015 * features["Facade_Dampening(Score)"]))

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

compliance = check_compliance(user_input["activity"], features["Laeq"], features["RT60(seconds)"])

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
best_abs = current_abs

if original_score < COMFORT_THRESHOLD:
    alternatives = suggest_better_materials(user_input["Element"], best_abs)
    for alt_material, alt_abs in alternatives:
        test_features = features.copy()
        test_features["Absortion_Coefficient"] = alt_abs
        test_features["Material"] = alt_material
        try:
            alt_score = model.predict(pd.DataFrame([test_features]))[0]
        except:
            volume = test_features["Surface_Area(m)"] * test_features["Height"]
            rt60 = 0.161 * volume / (alt_abs * test_features["Surface_Area(m)"] + 1e-5)
            alt_score = max(0, min(1, 1.2 - 0.01 * test_features["Laeq"] - 0.3 * rt60 + 0.015 * test_features["Facade_Dampening(Score)"]))
        if alt_score > best_score:
            best_score = alt_score
            best_material = alt_material
            best_abs = alt_abs

# --- OUTPUT SECTION ---
print("----- COMFORT PREDICTION -----")
print(f"Prediction Tier Used: {tier}")
print(f"Comfort Score: {round(original_score, 3)}")
print(f"Material: {features.get('Material')} | Abs Coef: {current_abs}")
if best_material:
    print(f"Suggested Better Material: {best_material} (Abs: {best_abs}) → Score: {round(best_score, 3)}")

# Acoustic Metrics (only once)
print("\n----- Acoustic Metrics Used -----")
for k in [
    "Laeq", "SPL", "RT60(seconds)", "RT60 (material ac)",
    "Surface_Area(m)", "Height", "Absortion_Coefficient", "Facade_Dampening(Score)"
]:
    print(f"{k}: {features.get(k)}")

# Compliance
print("\n----- COMPLIANCE CHECK -----")
if compliance["LAeq"] is None:
    print(f"No compliance data found for activity: {user_input['activity']}")
else:
    print(f"Activity: {user_input['activity']}")
    print(f"  LAeq: {features['Laeq']} {'<=' if compliance['LAeq'] else '>'} {compliance['LAeq_max']} → {'Compliant' if compliance['LAeq'] else 'Not compliant'}")
    print(f"  RT60: {features['RT60(seconds)']} {'<=' if compliance['RT60'] else '>'} {compliance['RT60_max']} → {'Compliant' if compliance['RT60'] else 'Not compliant'}")
    print(f"  Source: {compliance['source']}")

# Optional contextual guidance
if VERBOSE:
    with open(GUIDANCE_JSON) as f:
        guidance = json.load(f)

    if not compliance["LAeq"]:
        print("\n⚠ LAeq is non-compliant.")
        print(guidance["LAeq_non_compliant"]["description"])
        for rec in guidance["LAeq_non_compliant"]["general_recommendations"]:
            print(f" - {rec}")
        if user_input["Floor_Level"] == 1:
            print(" - Use green buffers on ground level.")
        else:
            print(" - Improve window glazing for higher floors.")

    if not compliance["RT60"]:
        print("\n⚠ RT60 is non-compliant.")
        print(guidance["RT60_non_compliant"]["description"])
        for rec in guidance["RT60_non_compliant"]["general_recommendations"]:
            print(f" - {rec}")
