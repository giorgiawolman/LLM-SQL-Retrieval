import pandas as pd
import joblib
import os
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler

# ✅ Load your dataset
df = pd.read_csv("sql/updated_acoustic_dataset_2000_samples.csv")

# ✅ Define features and target
X = df.drop(columns=["comfort_score"])
y = df["comfort_score"]

# ✅ Define column types (based on your screenshot)
categorical = ["ZONE", "Apt Type", "Wall Material", "Window Material"]
numeric = ["dBa (Laeq)", "Walls", "Volume", "Height", "STL", "Absorption", "RT60", "SPL"]

# ✅ Preprocessing pipeline
preprocessor = ColumnTransformer([
    ("cat", OneHotEncoder(handle_unknown="ignore"), categorical),
    ("num", StandardScaler(), numeric)
])

# ✅ Define model pipeline
pipeline = Pipeline([
    ("preprocessor", preprocessor),
    ("regressor", RandomForestRegressor(n_estimators=100, random_state=42))
])

# ✅ Split dataset and train
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
pipeline.fit(X_train, y_train)

# ✅ Save trained model
os.makedirs("model", exist_ok=True)
joblib.dump(pipeline, "model/acoustic_comfort_score_model.pkl")

print("✅ Model trained and saved to model/acoustic_comfort_score_model.pkl")
