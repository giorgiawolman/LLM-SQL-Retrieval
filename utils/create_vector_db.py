import sys
import os
import json

# Add project root for config access
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from server.config import *  # uses embedding_model, mode, client

# File paths
input_file = "knowledge/table_descriptions.json"
output_file = "knowledge/table_descriptions_vectors.json"

# Load JSON entries
with open(input_file, 'r', encoding='utf-8') as f:
    data = json.load(f)

# Generate embeddings
embeddings = []
for entry in data:
    name = entry.get("table_name", "unknown_table")
    desc = entry.get("description", "")
    content = f"Table: {name}. Description: {desc}"

    print(f"🔗 Embedding {name}...")
    vector = client.embeddings.create(input=[content], model=embedding_model).data[0].embedding

    embeddings.append({
        "name": name,         # ✅ key change here
        "content": content,
        "vector": vector
    })

# Save JSON
with open(output_file, "w", encoding="utf-8") as f:
    json.dump(embeddings, f, indent=2, ensure_ascii=False)

print(f"✅ Table description vectors saved to: {output_file}")
