import pandas as pd
import joblib
import re
import time
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.metrics import r2_score, mean_absolute_error

# === 1. Load Dataset ===
df = pd.read_csv("sql/Ecoform_Dataset_v1.csv")  # Adjusted path

# === 2. Clean Column Names ===
def clean_col(col):
    col = col.strip().lower()
    col = re.sub(r'[():]', '', col)
    col = re.sub(r'\s+', '_', col)
    return col

df.columns = [clean_col(col) for col in df.columns]

# === 3. Identify Target Column (robust match) ===
target_col_candidates = [col for col in df.columns if "comfort" in col and "index" in col]
assert len(target_col_candidates) == 1, "❌ Could not uniquely identify comfort index column."
target_col = target_col_candidates[0]

y = df[target_col]
X = df.drop(columns=[target_col])

# === 4. Auto-detect Feature Types ===
categorical = X.select_dtypes(include=["object"]).columns.tolist()
numeric = X.select_dtypes(exclude=["object"]).columns.tolist()

# === 5. Build Preprocessing + Model Pipeline ===
preprocessor = ColumnTransformer([
    ("cat", OneHotEncoder(handle_unknown="ignore"), categorical),
    ("num", StandardScaler(), numeric),
])

pipeline = Pipeline([
    ("preprocessor", preprocessor),
    ("regressor", RandomForestRegressor(
        n_estimators=30,        # Balanced: better than 10, faster than 100
        random_state=42,
        n_jobs=-1,              # Use all cores
        verbose=1               # Show progress
    )),
])

# === 6. Train/Test Split and Training ===
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

start = time.time()
pipeline.fit(X_train, y_train)
elapsed = time.time() - start
print(f"\n✅ Model trained in {elapsed:.2f} seconds")

# === 7. Evaluate Model ===
y_pred = pipeline.predict(X_test)
r2 = r2_score(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)

print(f"📊 R² Score: {r2:.3f}")
print(f"📉 MAE: {mae:.4f}")

# === 8. Save Model ===
joblib.dump(pipeline, "model/ecoform_acoustic_comfort_model.pkl")
print("✅ Model saved to model/ecoform_acoustic_comfort_model.pkl")
