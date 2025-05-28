import pandas as pd
import joblib
import os
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler

# Load the dataset
df = pd.read_csv("sql/updated_acoustic_dataset_2000_samples.csv")

# Define features and target variable
X = df.drop(columns=["comfort_score"])
y = df["comfort_score"]

# Define column types
categorical = ["ZONE", "Apt Type", "Wall Material", "Window Material"]
numeric = ["dBa (Laeq)", "Walls", "Volume", "Height", "STL", "Absorption", "RT60", "SPL"]

# Define preprocessing steps
preprocessor = ColumnTransformer([
    ("cat", OneHotEncoder(handle_unknown="ignore"), categorical),
    ("num", StandardScaler(), numeric)
])

# Define the full pipeline (preprocessing + model)
pipeline = Pipeline([
    ("preprocessor", preprocessor),
    ("regressor", RandomForestRegressor(n_estimators=100, random_state=42))
])

# Split the dataset and train the model
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
pipeline.fit(X_train, y_train)

# Save the trained model to disk
os.makedirs("model", exist_ok=True)
joblib.dump(pipeline, "model/acoustic_comfort_score_model.pkl")

print("Model trained and saved to model/acoustic_comfort_score_model.pkl")
