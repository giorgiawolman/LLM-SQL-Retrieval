import sys
import os
import json

# Add project root for config access
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from server.config import *  # uses embedding_model, mode, client

# File paths
input_file = "knowledge/material_acoustic_knowledge.json"
output_file = "knowledge/material_acoustic_knowledge_vectors.json"

# Load JSON entries (expecting a dictionary with categories)
with open(input_file, 'r', encoding='utf-8') as f:
    data = json.load(f)

# Generate embedding chunks for each material in each category
embeddings = []
for category, items in data.items():
    for i, item in enumerate(items):
        name = item["material"]
        content = (
            f"{item['material']} ({category}): STL {item['STL_dB']} dB, "
            f"Absorption Coeff. @500Hz {item['Absorption_Coefficient_500Hz']}, "
            f"Scattering {item['Scattering_Coefficient']}. "
            f"Use: {item['Typical_Use']}"
        )
        print(f"🔗 Embedding {name} ({category})...")
        vector = client.embeddings.create(input=[content], model=embedding_model).data[0].embedding
        embeddings.append({
            "name": name,
            "category": category,
            "content": content,
            "vector": vector
        })

# Save as JSON
with open(output_file, "w", encoding="utf-8") as f:
    json.dump(embeddings, f, indent=2, ensure_ascii=False)

print(f"✅ Vector embeddings saved to: {output_file}")
