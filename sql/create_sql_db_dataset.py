# sql/create_material_db.py

import sqlite3
import pandas as pd
from pathlib import Path

# File paths
csv_file_path = Path("sql/material_acoustic_knowledge_cleaned.csv")
db_file_path = Path("sql/material-database.db")

# Load CSV
df = pd.read_csv(csv_file_path)
df.columns = df.columns.str.strip().str.replace(" ", "_")  # Clean column names

# Connect to SQLite
conn = sqlite3.connect(db_file_path)
cursor = conn.cursor()

# Drop existing table if it exists
cursor.execute("DROP TABLE IF EXISTS material_knowledge")

# Define SQL column types based on DataFrame
types = df.dtypes
column_defs = []
for col, dtype in zip(df.columns, types):
    if pd.api.types.is_integer_dtype(dtype):
        col_type = 'INTEGER'
    elif pd.api.types.is_float_dtype(dtype):
        col_type = 'REAL'
    else:
        col_type = 'TEXT'
    column_defs.append(f'"{col}" {col_type}')

# Create table
create_table_sql = f'''
CREATE TABLE material_knowledge (
    {', '.join(column_defs)}
)
'''
cursor.execute(create_table_sql)

# Insert data into table
df.to_sql("material_knowledge", conn, if_exists="append", index=False)
print(f"✅ Material acoustic knowledge inserted into {db_file_path}")

# Close connection
conn.close()
