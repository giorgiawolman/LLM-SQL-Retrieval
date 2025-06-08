from server.config import client, completion_model
import re

# 🔹 Extract structured variables from free-form question
def extract_variables(user_question: str) -> dict:
    response = client.chat.completions.create(
        model=completion_model,
        messages=[
            {
                "role": "system",
                "content": """
You are an assistant for acoustic comfort evaluation.
You receive a user's question and return a dictionary of structured inputs.

Extract ONLY these fields if mentioned:
- Apartment_Type
- Zone
- Element (room type)
- wall_material
- window_material
- Floor_Level (numeric)
- activity (e.g. Living, Sleeping, Working)

Return a Python dictionary (no explanation).
If a field is missing, omit it.
"""
            },
            {
                "role": "user",
                "content": f"User Question: {user_question}"
            }
        ]
    )

    try:
        content = response.choices[0].message.content.strip()
        variables = eval(content) if isinstance(content, str) else content
        return variables
    except Exception as e:
        print("⚠️ Extraction failed:", e)
        return {}

# 🔹 Summarize acoustic score + compliance + recommendations
def build_answer(user_question: str, result: dict) -> str:
    score = result.get("comfort_score")
    source = result.get("source", "N/A")
    compliance = result.get("compliance", {})
    recommendations = result.get("recommendations", {})
    improved_score = result.get("improved_score", None)

    reason = compliance.get("reason", "")
    status = compliance.get("status", "unknown")

    summary_prompt = f"""
User Question:
{user_question}

📊 Evaluation Result:
- Comfort Score: {score}
- Source: {source}
- Compliance Status: {status} — {reason}
- Recommendations: {recommendations if recommendations else "None needed"}
- Improved Score: {improved_score if improved_score else "N/A"}
"""

    response = client.chat.completions.create(
        model=completion_model,
        messages=[
            {
                "role": "system",
                "content": """
You are an assistant that summarizes acoustic comfort evaluations clearly and concisely.

Instructions:
- Be brief and avoid repetition.
- Never say the same sentence multiple times.
- Clearly state whether the space is compliant.
- If there are recommendations, summarize them as a helpful list.
- Avoid unnecessary elaboration if the result is already compliant.
"""
            },
            {
                "role": "user",
                "content": summary_prompt
            }
        ]
    )

    return response.choices[0].message.content.strip()
