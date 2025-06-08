import sqlite3
import pandas as pd
import os
import joblib
import sys

# Ensure local paths work correctly
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

from utils.infer_from_inputs_v1 import infer_features

# Config paths
DB_PATH = "sql/comfort-database.db"
MODEL_PATH = "model/ecoform_acoustic_comfort_model.pkl"

def query_or_predict(user_input):
    """
    Attempts to retrieve comfort score from SQL DB, otherwise uses ML model.
    """
    abs_db_path = os.path.abspath(DB_PATH)
    print(f"🔍 Using database file: {abs_db_path}")
    conn = sqlite3.connect(abs_db_path)

    # List available columns
    cursor = conn.cursor()
    cursor.execute("PRAGMA table_info(comfort_lookup);")
    columns = [row[1] for row in cursor.fetchall()]
    print("📋 Available Columns in comfort_lookup:", columns)

    # SQL condition building
    conditions = []

    if "Apartment_Type" in user_input:
        conditions.append(f"LOWER(apartment_type_string) = '{user_input['Apartment_Type'].lower()}'")
    if "Zone" in user_input:
        conditions.append(f"LOWER(zone_string) = '{user_input['Zone'].lower()}'")
    if "wall_material" in user_input:
        material = user_input["wall_material"].lower()
        conditions.append(f"LOWER(element_materials_string) LIKE '%{material}%'")
    if "Floor_Level" in user_input:
        floor_height = user_input["Floor_Level"] * 3
        if "floor_height_m" in columns:
            conditions.append(f"floor_height_m = {floor_height}")
        elif "floor_level" in columns:
            conditions.append(f"floor_level = {user_input['Floor_Level']}")

    sql = f"""
    SELECT comfort_index_float, 'Compliant' AS compliance
    FROM comfort_lookup
    WHERE {' AND '.join(conditions)}
    ORDER BY comfort_index_float DESC
    LIMIT 1;
    """

    try:
        result = pd.read_sql_query(sql, conn)
        if not result.empty:
            return {
                "score": result.iloc[0]["comfort_index_float"],
                "source": "SQL Match",
                "compliance": result.iloc[0]["compliance"]
            }
        else:
            raise ValueError("No match in SQL.")
    except Exception as e:
        print("⚠️ Not found in SQL. Using model prediction.")
        print(f"❌ SQL Error: {e}\n")

        model = joblib.load(MODEL_PATH)
        features, tier = infer_features(
            apartment_type=user_input.get("Apartment_Type"),
            zone=user_input.get("Zone"),
            element_material=user_input.get("wall_material"),
            floor_level=user_input.get("Floor_Level")
        )
        score = model.predict(pd.DataFrame([features]))[0]
        return {
            "score": score,
            "source": f"model ({tier})",
            "compliance": "N/A"
        }
