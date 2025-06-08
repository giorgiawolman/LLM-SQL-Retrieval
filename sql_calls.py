import sqlite3
import pandas as pd
import os
from recommend_recompute import recommend_recompute

# === Path to SQLite DB ===
DB_PATH = "sql/comfort-database.db"

# === Main SQL Call Function ===
def query_or_recommend(user_input):
    """
    First attempts to retrieve the acoustic comfort score from the SQL database.
    If no match is found, falls back to the ML model and recommendation pipeline.
    """
    abs_db_path = os.path.abspath(DB_PATH)
    print(f"🔍 Using database file: {abs_db_path}")
    conn = sqlite3.connect(abs_db_path)

    # Check available columns in the table
    cursor = conn.cursor()
    cursor.execute("PRAGMA table_info(comfort_lookup);")
    columns = [row[1] for row in cursor.fetchall()]
    print("📋 Available Columns in comfort_lookup:", columns)

    # Build WHERE clause based on available user input
    conditions = []

    if "Apartment_Type" in user_input:
        conditions.append(f"LOWER(apartment_type_string) = '{user_input['Apartment_Type'].lower()}'")
    if "Zone" in user_input:
        conditions.append(f"LOWER(zone_string) = '{user_input['Zone'].lower()}'")

    # Try full match of both materials first
    if "wall_material" in user_input and "window_material" in user_input:
        combo = f"{user_input['window_material']} and {user_input['wall_material']}".lower()
        conditions.append(f"LOWER(element_materials_string) LIKE '%{combo}%'")
    else:
        if "wall_material" in user_input:
            conditions.append(f"LOWER(element_materials_string) LIKE '%{user_input['wall_material'].lower()}%'")
        if "window_material" in user_input:
            conditions.append(f"LOWER(element_materials_string) LIKE '%{user_input['window_material'].lower()}%'")

    if "Floor_Level" in user_input:
        floor_height = round(user_input["Floor_Level"] * 3.0, 2)
        if "floor_height_m" in columns:
            # Add tolerance to avoid float mismatch
            conditions.append(f"ABS(floor_height_m - {floor_height}) < 0.1")
        elif "floor_level" in columns:
            conditions.append(f"floor_level = {user_input['Floor_Level']}")

    # Final SQL query
    sql_query = f"""
    SELECT comfort_index_float, 'Compliant' AS compliance
    FROM comfort_lookup
    WHERE {' AND '.join(conditions)}
    ORDER BY comfort_index_float DESC
    LIMIT 1;
    """
    print("📝 SQL Query:", sql_query)

    try:
        result = pd.read_sql_query(sql_query, conn)
        if not result.empty:
            print("✅ Match found in SQL database.")
            return {
                "comfort_score": round(result.iloc[0]["comfort_index_float"], 3),
                "source": "SQL Match",
                "compliance": {"status": "compliant", "reason": "Matched from database"},
                "recommendations": {},
                "improved_score": None
            }
        else:
            raise ValueError("No match in SQL.")
    except Exception as e:
        print("⚠️ SQL lookup failed or no match:", e)
        print("🔄 Switching to model + compliance + recommendation...")
        return recommend_recompute(user_input)
