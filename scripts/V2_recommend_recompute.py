import sys
import os
import sqlite3
import pandas as pd
import joblib

# Add project root to system path so 'utils' works
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from utils.infer_from_inputs import infer_features

# --- Config ---
MODEL_PATH = "model/acoustic_comfort_score_model.pkl"
MATERIAL_DB_PATH = "sql/material-database.db"
COMFORT_THRESHOLD = 0.7

# --- User Input ---
user_input = {
    "Apartment_Type": "1Bed",
    "Zone": "HD-Urban-V1",
    "Element": "Wall",
    "Floor_Level": 2,
    "wall_material": "Painted Brick",
    "window_material": "Double Pane Glass"
}

# --- Load Model ---
model = joblib.load(MODEL_PATH)

# --- Infer Features ---
features, tier = infer_features(
    apartment_type=user_input["Apartment_Type"],
    zone=user_input["Zone"],
    element=user_input["Element"],
    wall_material=user_input.get("wall_material"),
    window_material=user_input.get("window_material"),
    floor_level=user_input["Floor_Level"]
)

# --- Load Absorption from Material SQL DB ---
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

# --- Try Prediction with Model ---
X = pd.DataFrame([features])
try:
    original_score = model.predict(X)[0]
    model_success = True
except Exception:
    model_success = False

# --- Manual fallback computation ---
def compute_rt60(volume, absorption, surface_area):
    return 0.161 * volume / (absorption * surface_area + 1e-5)

def compute_comfort_score(laeq, rt60, facade_damping):
    return max(0, min(1, 1.2 - 0.01 * laeq - 0.3 * rt60 + 0.015 * facade_damping))

if not model_success:
    print("⚠️ Model prediction failed. Falling back to formula-based computation.")
    volume = features["Surface_Area(m)"] * features["Height"]
    rt60 = compute_rt60(volume, features["Absortion_Coefficient"], features["Surface_Area(m)"])
    features["RT60(seconds)"] = rt60
    features["RT60 (material ac)"] = rt60
    original_score = compute_comfort_score(
        features["Laeq"],
        rt60,
        features["Facade_Dampening(Score)"]
    )

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

alternatives = suggest_better_materials(user_input["Element"], current_abs)

# --- Evaluate Alternatives ---
best_score = original_score
best_material = None
best_abs = current_abs

for alt_material, alt_abs in alternatives:
    test_features = features.copy()
    test_features["Absortion_Coefficient"] = alt_abs
    try:
        alt_score = model.predict(pd.DataFrame([test_features]))[0]
    except:
        volume = test_features["Surface_Area(m)"] * test_features["Height"]
        rt60 = compute_rt60(volume, alt_abs, test_features["Surface_Area(m)"])
        alt_score = compute_comfort_score(
            test_features["Laeq"], rt60, test_features["Facade_Dampening(Score)"]
        )
    if alt_score > best_score:
        best_score = alt_score
        best_material = alt_material
        best_abs = alt_abs

# --- Output ---
print("----- COMFORT PREDICTION -----")
print(f"Prediction Tier Used: {tier}")
print(f"Inferred Original Material: {features.get('Material', 'N/A')}")
print(f"Original Absorption Coefficient: {current_abs}")
print(f"Original Comfort Score: {round(original_score, 3)}")

if best_material:
    print(f"\nBest Substitution: {best_material} (Abs: {best_abs})")
    print(f"Recomputed Comfort Score: {round(best_score, 3)}")
else:
    print("\nNo better substitution found (original material is best).")

print("\n----- Acoustic Metrics Used -----")
for k in [
    "Laeq", "SPL", "RT60(seconds)", "RT60 (material ac)",
    "Surface_Area(m)", "Height", "Absortion_Coefficient", "Facade_Dampening(Score)"
]:
    print(f"{k}: {features.get(k)}")
