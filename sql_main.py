from server.config import *
from llm_calls import *
from sql_calls import *
from utils.rag_utils import sql_rag_call
from scripts.predict_comfort import infer_features, model
import pandas as pd

# --- 🔹 USER INPUT: Define apartment setup ---
user_input = {
    "Apt Type": "2b",
    "ZONE": "HD-Urban-V1",
    "Wall Material": "Concrete (20cm)",
    "Window Material": "Double Glazing"
}

# --- 🔹 Generate semantic question for LLM + SQL ---
user_question = (
    f"Evaluate acoustic comfort and compliance for a {user_input['Apt Type']} apartment "
    f"in {user_input['ZONE']} with {user_input['Wall Material']} walls and "
    f"{user_input['Window Material']} windows. Include predictions and material recommendations."
)

# --- 🔹 Target database ---
db_path = "sql/acoustic-database.db"
db_schema = get_dB_schema(db_path)

# --- 🔹 Retrieve most relevant table using vector search ---
print("🔍 Initiating RAG...")
table_descriptions_path = "knowledge/table_descriptions.json"
relevant_table, table_description = sql_rag_call(user_question, table_descriptions_path, n_results=1)

if not relevant_table:
    print("❌ No relevant table found.")
    exit()

print(f"✅ Most relevant table: {relevant_table}")
filtered_schema = {relevant_table: db_schema.get(relevant_table)}
db_context = format_dB_context(db_path, filtered_schema)

# --- 🔹 Generate SQL query using LLM ---
sql_query = generate_sql_query(db_context, table_description, user_question)
print(f"\n📥 SQL Query:\n{sql_query}")

if "No information" in sql_query:
    print("❌ LLM says this question cannot be answered from available data.")
    exit()

# --- 🔁 Execute query with self-debug logic ---
sql_query, query_result = fetch_sql(sql_query, db_context, user_question, db_path)

if not query_result or str(query_result) == "[(0,)]":
    print("❌ SQL failed after retries.")
    exit()

# --- 🔮 Predict acoustic comfort score using trained model ---
features = infer_features(
    apt_type=user_input["Apt Type"],
    zone=user_input["ZONE"],
    wall_material=user_input["Wall Material"],
    window_material=user_input["Window Material"]
)
print("DEBUG features:", features)  # Debug print

X = pd.DataFrame([features])
print("DEBUG X columns:", X.columns)  # Debug print

if "comfort_score" in X.columns:
    X = X.drop(columns=["comfort_score"])
else:
    print("WARNING: 'comfort_score' not in features, skipping drop.")

predicted_score = model.predict(X)[0]

print(f"\n🎯 Predicted Comfort Score: {round(predicted_score, 2)}")

# --- 🧠 Ask LLM to summarize results and advice ---
full_prompt = f"{user_question}\nPredicted Comfort Score: {round(predicted_score, 2)}"
final_answer = build_answer(sql_query, query_result, full_prompt)

print(f"\n📖 Final LLM Answer:\n{final_answer}")