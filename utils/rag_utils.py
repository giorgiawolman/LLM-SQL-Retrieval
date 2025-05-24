import numpy as np
import json
import os
import sys
import re

# Add the project root to path for module imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from server.config import *

# Embedding wrapper
def get_embedding(text, model=embedding_model):
    text = text.replace("\n", " ")
    if mode == "openai":
        response = client.embeddings.create(input=[text], dimensions=768, model=model)
    else:
        response = client.embeddings.create(input=[text], model=model)
    return response.data[0].embedding

# Compute cosine similarity
def similarity(v1, v2):
    return np.dot(v1, v2)

# Load vectorized JSON
def load_embeddings(filepath):
    with open(filepath, 'r', encoding='utf-8') as f:
        return json.load(f)

# Get top-N similar entries
def get_vectors(query_vector, index_lib, n_results):
    scored = []
    for item in index_lib:
        score = similarity(query_vector, item["vector"])
        scored.append({
            "name": item["name"],
            "content": item["content"],
            "score": score
        })
    scored.sort(key=lambda x: x["score"], reverse=True)
    return scored[:n_results]

# RAG-style chat completion
def rag_answer(question, prompt, model=completion_model):
    completion = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": prompt},
            {"role": "user", "content": question}
        ],
        temperature=0.1,
    )
    return completion.choices[0].message.content

# Main RAG call
def sql_rag_call(question, embedding_file, n_results=3):
    print("🔍 Initiating RAG...")

    # Step 1: Embed the user's question
    question_vector = get_embedding(question)

    # Step 2: Load pre-embedded table descriptions
    index_lib = load_embeddings(embedding_file)

    # Step 3: Rank and retrieve top entries
    top_matches = get_vectors(question_vector, index_lib, n_results)
    names = "\n".join([match["name"] for match in top_matches])
    descriptions = "\n".join([match["content"] for match in top_matches])

    return names, descriptions
