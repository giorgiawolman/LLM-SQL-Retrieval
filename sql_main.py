import sys
import os
import pandas as pd

# --- Path Setup ---
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

# --- Imports ---
from server.config import get_dB_schema, format_dB_context
from server.keys import *
from sql_calls_v1 import query_or_predict
from utils.rag_utils import sql_rag_call
from llm_calls import generate_sql_query, fetch_sql, build_answer

# --- USER INPUT ---
user_input = {
    "Apartment_Type": "1Bed",
    "Zone": "GreenEdge-V3",
    "Element": "Window",
    "wall_material": "Double Glazing",
    "Floor_Level": 6
}

# --- USER QUESTION ---
user_question = (
    f"Evaluate acoustic comfort and compliance for a {user_input['Apartment_Type']} apartment "
    f"in {user_input['Zone']} with {user_input['wall_material']} windows. "
    f"Include predictions and material recommendations."
)

# --- Load DB Schema ---
db_path = "sql/comfort-database.db"
print("🔵 Fetching DB schema...")
db_schema = get_dB_schema(db_path)
normalized_schema = {k.lower(): v for k, v in db_schema.items()}

# --- Find Relevant Table ---
print("🧠 Starting RAG process...")
table_descriptions_path = "knowledge/table_descriptions_vectors.json"
relevant_table, table_description = sql_rag_call(user_question, table_descriptions_path, n_results=1)

if not relevant_table:
    print("❌ No relevant table found.")
    exit()

print(f"✅ Relevant table: {relevant_table}")

# --- Match Schema ---
filtered_schema = {relevant_table: normalized_schema.get(relevant_table.lower())}
if filtered_schema[relevant_table] is None:
    print(f"⚠️ Could not find schema info for table: {relevant_table}")
    exit()

# --- Format Context for LLM ---
db_context = format_dB_context(db_path, filtered_schema)

# --- Generate SQL Query ---
sql_query = generate_sql_query(db_context, table_description, user_question)
print(f"\n📤 SQL Query:\n{sql_query}")

if "No information" in sql_query:
    print("❌ LLM could not answer with schema.")
    exit()

# --- Try SQL Call ---
sql_query, query_result = fetch_sql(sql_query, db_context, user_question, db_path)

# --- Fallback to Model if SQL Fails ---
if not query_result or str(query_result) == "[(0,)]":
    print("⚠️ Not found in SQL. Using model prediction.")
    result = query_or_predict(user_input)
    print("\n---- Comfort Score Result ----")
    print(f"🎯 Score: {round(result['score'], 3)}")
    print(f"Source: {result['source']}")
    
    # Build LLM answer with fallback info
    full_prompt = f"{user_question}\nPredicted Comfort Score: {round(result['score'], 2)}"
    final_answer = build_answer("No SQL result (model fallback)", [], full_prompt)
    print("\n🧠 LLM Summary:\n" + final_answer)
    exit()

# --- SQL Result Found: Still use Model for Prediction ---
result = query_or_predict(user_input)

# --- Output Results ---
print("\n---- Comfort Score Result ----")
print(f"🎯 Score: {round(result['score'], 3)}")
print(f"Source: {result['source']}")

# --- Final Answer via LLM ---
full_prompt = f"{user_question}\nPredicted Comfort Score: {round(result['score'], 2)}"
final_answer = build_answer(sql_query, query_result, full_prompt)
print("\n🧠 LLM Summary:\n" + final_answer)
