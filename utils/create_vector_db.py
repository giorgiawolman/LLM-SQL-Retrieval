import sys
import os
import json

# Add project root for config access
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from server.config import *  # uses embedding_model, mode, client

# File paths
input_file = "knowledge/compliance_thresholds_extended.json"
output_file = "knowledge/compliance_thresholds_vectors.json"

# Load JSON entries
with open(input_file, 'r', encoding='utf-8') as f:
    data = json.load(f)

# Generate embedding chunks
embeddings = []
for i, entry in enumerate(data):
    name = entry["use"]
    content = f"{entry['use']} space: LAeq max {entry['LAeq_max']} dB, RT60 max {entry['RT60_max']} s. Source: {entry['source']}."
    print(f"🔗 Embedding {name} ({i + 1}/{len(data)})...")
    vector = client.embeddings.create(input=[content], model=embedding_model).data[0].embedding
    embeddings.append({
        "name": name,
        "content": content,
        "vector": vector
    })

# Save as JSON
with open(output_file, "w", encoding="utf-8") as f:
    json.dump(embeddings, f, indent=2, ensure_ascii=False)

print(f"✅ Vector embeddings saved to: {output_file}")
