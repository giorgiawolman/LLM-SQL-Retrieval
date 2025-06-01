import sys
import os
import json
import pandas as pd
import joblib

# Add project root to system path so 'utils' works
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from utils.infer_from_inputs import infer_features

# --- Config ---
MODEL_PATH = "model/acoustic_comfort_score_model.pkl"
JSON_KNOWLEDGE_PATH = "knowledge/material_acoustic_knowledge.json"
COMFORT_THRESHOLD = 0.7

# --- User Input (Wall as selected element) ---
user_input = {
    "Apartment_Type": "1Bed",
    "Zone": "HD-Urban-V1",
    "Element": "Wall",
    "Floor_Level": 1,
    "wall_material": "Painted Brick",
    "window_material": "Double Pane Glass"
}

# --- Load model ---
model = joblib.load(MODEL_PATH)

# --- Infer features ---
features, tier = infer_features(
    apartment_type=user_input["Apartment_Type"],
    zone=user_input["Zone"],
    element=user_input["Element"],
    wall_material=user_input.get("wall_material"),
    window_material=user_input.get("window_material"),
    floor_level=user_input["Floor_Level"]
)

# --- Predict original comfort score ---
X = pd.DataFrame([features])
original_score = model.predict(X)[0]

# --- Load material knowledge ---
with open(JSON_KNOWLEDGE_PATH, "r") as f:
    material_db = json.load(f)

# --- Suggest better materials (top 3 with higher absorption) ---
def suggest_better_materials(current_abs, element):
    element = element.lower()
    category = "wall" if "wall" in element else "window" if "window" in element else element
    options = [
        m for m in material_db
        if m["category"].lower() == category and m["Absorption_Coefficient_500Hz"] > current_abs
    ]
    options.sort(key=lambda x: x["Absorption_Coefficient_500Hz"], reverse=True)
    return options[:3]

# --- Evaluate substitutions ---
current_abs = features["Absortion_Coefficient"]
element_type = features["Element"]
alternatives = suggest_better_materials(current_abs, element_type)

best_score = original_score
best_material = None
best_abs = current_abs

for alt in alternatives:
    test_features = features.copy()
    test_features["Absortion_Coefficient"] = alt["Absorption_Coefficient_500Hz"]
    test_score = model.predict(pd.DataFrame([test_features]))[0]
    if test_score > best_score:
        best_score = test_score
        best_material = alt["material"]
        best_abs = alt["Absorption_Coefficient_500Hz"]

# --- Output ---
print("----- COMFORT PREDICTION -----")
print(f"Prediction Tier Used: {tier}")
print(f"Inferred Original Material (from dataset): {features.get('Material', 'N/A')}")
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
