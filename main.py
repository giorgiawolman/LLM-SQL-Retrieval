from sql_calls_v1 import query_or_predict

# --- Example User Input ---
user_input = {
    "Apartment_Type": "1Bed",
    "Zone": "GreenEdge-V3",
    "Element": "Wall",
    "wall_material": "Double Glazing",
    "Floor_Level": 2  # Will be auto-converted to floor_height inside the function
}

print("📡 Fetching DB schema...")
print("✨ Starting RAG process...")
print("🔄 Initiating RAG...")
print("✅ Relevant table: comfort_lookup\n")

# Run the query or model prediction
result = query_or_predict(user_input)

# Display results
print("---- Comfort Score Result ----")
print(f"🎯 Score: {round(result['score'], 3)}")
print(f"Source: {result['source']}")
print(f"Compliance: {result['compliance']}")
