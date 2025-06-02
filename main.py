import sys
import os
import json

# --- Add root for relative imports ---
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), ".")))

# --- Imports from scripts folder ---
from scripts.v4_recommend_recompute import run_acoustic_prediction
from scripts.llm_acoustic_query_handler import handle_llm_query

# --- Example 1: Structured prediction input (disabled for now) ---
user_input = None
# user_input = {
#     "Apartment_Type": "2Bed",
#     "Zone": "HD-Urban-V1",
#     "Element": "Wall",
#     "Floor_Level": 1,
#     "wall_material": "Painted Brick",
#     "window_material": "Double Pane Glass",
#     "activity": "Living",
#     "Laeq": 57
# }

# --- Example 2: Natural language guidance query (active now) ---
user_question = "What should I do if the noise level in a Living room is too high?"
# user_question = None

# --- Decide what to do ---
if user_question:
    print("Interpreting as natural language guidance query...\n")
    guidance = handle_llm_query(user_question)
    print(guidance)

elif user_input:
    print("Interpreting as structured prediction query...\n")
    results = run_acoustic_prediction(user_input)
    print(results)

else:
    print("No input provided.")