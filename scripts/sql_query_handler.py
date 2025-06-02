import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from server.config import client, completion_model

def generate_sql_query(db_schema: str, table_descriptions: str, user_question: str) -> str:
    messages = [
        {
            "role": "system",
            "content": f"""
You are an expert SQL assistant generating SELECT queries for an architectural acoustics dataset.

# Schema #
{db_schema}

# Description #
{table_descriptions}

- Only use exact column names.
- Return a valid SQLite SELECT query.
- Do not explain the result — just give the query.
"""
        },
        {
            "role": "user",
            "content": f"User question: {user_question}"
        }
    ]

    response = client.chat.completions.create(
        model=completion_model,
        messages=messages
    )
    return response.choices[0].message.content.strip()