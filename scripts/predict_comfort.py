import sys
import os
import json
import joblib
import pandas as pd

# Add project root so 'utils' works
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from utils.infer_from_inputs import infer_features

# 🔹 Load trained model and thresholds
model = joblib.load("model/acoustic_comfort_score_model.pkl")
thresholds = json.load(open("knowledge/compliance_thresholds_extended.json"))

# 🔹 User input (minimal)
user_input = {
    "Apt Type": "2b",
    "ZONE": "HD-Urban-V1",
    "Wall Material": "Concrete (20cm)",
    "Window Material": "Double Glazing"
}

# 🔹 Infer full input row
features = infer_features(
    apt_type=user_input["Apt Type"],
    zone=user_input["ZONE"],
    wall_material=user_input["Wall Material"],
    window_material=user_input["Window Material"]
)

# 🔹 Predict comfort score
X = pd.DataFrame([features]).drop(columns=["comfort_score"])
score = model.predict(X)[0]

# 🔹 Compliance check
room_type = "Living" if user_input["Apt Type"] in ["1b", "2b"] else "Sleeping"
rule = next(r for r in thresholds if r["use"] == room_type)

compliant = (
    features["RT60"] <= rule["RT60_max"] and
    features["dBa (Laeq)"] <= rule["LAeq_max"]
)

# 🔹 Recommendations
recommendations = []
if not compliant:
    if features["RT60"] > rule["RT60_max"]:
        recommendations.append("Add ceiling absorbers.")
    if features["dBa (Laeq)"] > rule["LAeq_max"]:
        recommendations.append("Upgrade to triple glazing or use better facade materials.")

# 🔹 Load material knowledge base
with open("knowledge/material_acoustic_knowledge.json", "r") as f:
    material_db = json.load(f)

def suggest_better_materials(current_stl, current_absorption, category):
    return [
        mat["material"]
        for mat in material_db
        if mat["category"] == category and (
            mat["STL_dB"] > current_stl or mat["Absorption_Coefficient_500Hz"] > current_absorption
        )
    ]

wall_stl = features.get("STL", 35)
wall_abs = features.get("Absorption", 0.1)

better_walls = suggest_better_materials(wall_stl, wall_abs, "Wall")
better_windows = suggest_better_materials(wall_stl, wall_abs, "Window")

if not compliant:
    if better_walls:
        recommendations.append(f"Try alternative wall materials: {', '.join(better_walls[:3])}")
    if better_windows:
        recommendations.append(f"Try alternative window materials: {', '.join(better_windows[:3])}")

# 🔹 Output
result = {
    "comfort_score": round(score, 2),
    "compliance": "Compliant" if compliant else "Non-compliant",
    "recommendations": recommendations
}

print(json.dumps(result, indent=2))
