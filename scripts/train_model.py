import pandas as pd
import joblib
import os
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler

# === 1. Load Cleaned Dataset ===
df = pd.read_csv("sql/cleaned_dataset.csv")  # Make sure the path is correct

# === 2. Define Target and Features ===
y = df["Comfort_Index"]
X = df.drop(columns=["Comfort_Index", "Material"])  # Drop target and unused

# === 3. Specify Column Types ===
categorical = ["Zone", "Apartment_Type", "Element"]
numeric = [
    "Laeq", "SPL", "RT60(seconds)", "RT60 (material ac)",
    "Surface_Area(m)", "Height", "Absortion_Coefficient", "Facade_Dampening(Score)"
]

# === 4. Build Preprocessing Pipeline ===
preprocessor = ColumnTransformer([
    ("cat", OneHotEncoder(handle_unknown="ignore"), categorical),
    ("num", StandardScaler(), numeric)
])

# === 5. Combine Preprocessing with Model ===
pipeline = Pipeline([
    ("preprocessor", preprocessor),
    ("regressor", RandomForestRegressor(n_estimators=100, random_state=42))
])

# === 6. Split Data and Train ===
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
pipeline.fit(X_train, y_train)

# === 7. Save Trained Model ===
os.makedirs("model", exist_ok=True)
joblib.dump(pipeline, "model/acoustic_comfort_score_model.pkl")

print("✅ Model trained and saved to model/acoustic_comfort_score_model.pkl")
