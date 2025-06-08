# sql/create_ecoform_db.py

import sqlite3
import pandas as pd
from pathlib import Path
import re

# File paths
csv_file_path = Path("sql/Ecoform_Dataset_v1.csv")
db_file_path = Path("sql/comfort-database.db")

# Load CSV
df = pd.read_csv(csv_file_path)

# === Clean and Deduplicate Column Names ===
def clean_col(col):
    col = col.strip().lower()
    col = re.sub(r'[():]', '', col)
    col = re.sub(r'\s+', '_', col)
    return col

raw_cols = [clean_col(col) for col in df.columns]
deduped_cols = []
seen = {}
for col in raw_cols:
    if col not in seen:
        seen[col] = 1
        deduped_cols.append(col)
    else:
        seen[col] += 1
        deduped_cols.append(f"{col}_{seen[col]}")

df.columns = deduped_cols

# Connect to SQLite
conn = sqlite3.connect(db_file_path)
cursor = conn.cursor()

# Drop existing table if it exists
cursor.execute("DROP TABLE IF EXISTS comfort_lookup")

# Define SQL column types based on DataFrame
columns = df.columns
types = df.dtypes

column_defs = []
for col, dtype in zip(columns, types):
    if pd.api.types.is_integer_dtype(dtype):
        col_type = 'INTEGER'
    elif pd.api.types.is_float_dtype(dtype):
        col_type = 'REAL'
    else:
        col_type = 'TEXT'
    column_defs.append(f'"{col}" {col_type}')

# Create table
create_table_sql = f'''
CREATE TABLE comfort_lookup (
    {', '.join(column_defs)}
)
'''
cursor.execute(create_table_sql)

# Insert data into table
df.to_sql("comfort_lookup", conn, if_exists="append", index=False)
print(f"✅ Ecoform dataset inserted into {db_file_path}")

# Close connection
conn.close()
