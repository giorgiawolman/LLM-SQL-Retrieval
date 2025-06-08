import sys
import os

# Ensure local import path
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

from llm_calls import extract_variables, build_answer
from sql_calls import query_or_recommend

# === CONFIG: Choose input mode ===
use_structured_input = True  # ⬅️ Set to True to test structured inputs directly

# === INPUT BLOCK ===
if use_structured_input:
    print("🟢 Using structured input...")
    user_input = {
        "Apartment_Type": "2Bed",
        "Zone": "GreenEdge-V3",
        "Element": "Living",
        "wall_material": "Rammed Earth",
        "window_material": "Single Glazing",
        "Floor_Level": 1,
        "activity": "Sleeping"
    }
    user_question = (
        f"Evaluate acoustic comfort and compliance for a {user_input['Apartment_Type']} apartment "
        f"in {user_input['Zone']} with {user_input['wall_material']} walls and "
        f"{user_input['window_material']} windows on floor {user_input['Floor_Level']}."
    )

else:
    print("🔵 Using free-form question...")
    user_question = (
        "How can I improve acoustic comfort in a 1Bed apartment in HD-Urban-V1 "
        "with single glazing and concrete walls on the 3rd floor?"
    )

    print("🤖 Extracting structured parameters from question...")
    user_input = extract_variables(user_question)

    if not user_input:
        print("❌ Failed to extract parameters from question.")
        sys.exit(1)

    print("✅ Extracted input:", user_input)

    # Add default activity if not mentioned
    user_input.setdefault("activity", "Living")

# === Acoustic Evaluation ===
print("🔍 Evaluating acoustic comfort...")
try:
    result = query_or_recommend(user_input)
except Exception as e:
    print(f"❌ Acoustic evaluation failed: {e}")
    sys.exit(1)

# === Print Raw Output ===
print("\n📦 Raw Output:")
for key, val in result.items():
    print(f"{key}: {val}")

# === Generate LLM Summary ===
print("\n🧠 Generating summary...")
try:
    summary = build_answer(user_question, result)
    print("\n" + summary)
except Exception as e:
    print(f"❌ Failed to generate summary: {e}")
