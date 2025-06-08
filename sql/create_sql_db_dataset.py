import sqlite3
import pandas as pd
from pathlib import Path

# File paths
csv_file_path = Path("sql/Ecoform_Dataset_v1.csv")
db_file_path = Path("sql/comfort-database.db")

# Load CSV
df = pd.read_csv(csv_file_path)
df.columns = df.columns.str.strip().str.replace(" ", "_").str.replace("(", "").str.replace(")", "").str.lower()

# Connect to SQLite
conn = sqlite3.connect(db_file_path)
cursor = conn.cursor()

# Drop existing table if it exists
cursor.execute("DROP TABLE IF EXISTS comfort_lookup")

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
CREATE TABLE comfort_lookup (
    {', '.join(column_defs)}
)
'''
cursor.execute(create_table_sql)

# Insert data into table
df.to_sql("comfort_lookup", conn, if_exists="append", index=False)
print(f"✅ Comfort dataset inserted into {db_file_path}")

# Close connection
conn.close()