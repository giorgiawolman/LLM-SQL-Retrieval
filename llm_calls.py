from server.config import client, completion_model

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

Return a valid Python dictionary (no explanation).
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
    score = result.get("comfort_score", "N/A")
    source = result.get("source", "N/A")
    compliance = result.get("compliance", {})
    recommendations = result.get("recommendations", {})
    improved_score = result.get("improved_score", "N/A")
    material_swap = result.get("material_swap", {})

    summary_prompt = f"""
User Question:
{user_question}

📊 Evaluation Summary:
- Comfort Score: {score}
- Source: {source}
- Compliance: {compliance.get("status")} — {compliance.get("reason")}
- LAeq: {compliance.get("LAeq")}
- RT60: {compliance.get("RT60")}

🛠 Recommendations:
{recommendations if recommendations else "None needed"}

💡 Material Upgrade Suggestions:
{material_swap if material_swap else "No material change recommended"}
Improved Score: {round(improved_score, 3) if improved_score and improved_score != 'N/A' else "N/A"}
"""

    response = client.chat.completions.create(
        model=completion_model,
        messages=[
            {
                "role": "system",
                "content": """
You summarize acoustic comfort evaluations clearly for architects and sustainability consultants.

Instructions:
- Do not repeat content.
- Clearly state whether the result is compliant.
- Mention original and improved score (if applicable).
- List upgraded materials if available.
- Avoid suggesting upgrades if already compliant.
"""
            },
            {
                "role": "user",
                "content": summary_prompt
            }
        ]
    )

    return response.choices[0].message.content.strip()
