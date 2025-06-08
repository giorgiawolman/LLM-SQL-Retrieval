from server.config import client, completion_model
import re

# 🔹 Generate SQL query from user question and schema
def generate_sql_query(dB_context: str, retrieved_descriptions: str, user_question: str) -> str:
    response = client.chat.completions.create(
        model=completion_model,
        messages=[
            {
                "role": "system",
                "content": f"""
You are an SQLite expert helping with acoustic comfort evaluation.
You generate SQL queries for a database that includes:

- Acoustic metrics by apartment and material
- Material acoustic properties (e.g. STL, absorption)
- WHO/ISO compliance thresholds
- Pretrained comfort scores and simulation results

# Schema #
{dB_context}

# Table Descriptions #
{retrieved_descriptions}

# Instructions #
- Carefully interpret the user’s question
- Use only exact table and column names found in the schema
- Return a single SQL query (no explanations, no formatting)
- If you can’t answer, return: No information
"""
            },
            {
                "role": "user",
                "content": f"User Question: {user_question}"
            }
        ]
    )
    return response.choices[0].message.content.strip()

# 🔹 Explain SQL result in context of acoustic comfort
def build_answer(sql_query: str, sql_result: str, user_question: str) -> str:
    response = client.chat.completions.create(
        model=completion_model,
        messages=[
            {
                "role": "system",
                "content": f"""
You are an expert in architectural acoustics and SQL.
You are helping interpret SQL results related to acoustic comfort, material performance, and compliance.

# Instructions #
- Use the query and result to explain the answer
- Reference relevant fields or material properties
- Be clear, concise, and useful to an acoustic consultant
"""
            },
            {
                "role": "user",
                "content": f"""
User Question: {user_question}
SQL Query: {sql_query}
SQL Result: {sql_result}
"""
            }
        ]
    )
    return response.choices[0].message.content.strip()

# 🔹 Fix SQL query that failed
def fix_sql_query(dB_context: str, user_question: str, attempted_queries: list, exceptions: list) -> str:
    error_log = "\n".join([
        f"# Query: {q}\n# Error: {e}" for q, e in zip(attempted_queries, exceptions)
    ])

    response = client.chat.completions.create(
        model=completion_model,
        messages=[
            {
                "role": "system",
                "content": f"""
You are an SQL query expert for an architectural acoustics database.
You will correct failed queries by analyzing errors and the table schema.

# Schema #
{dB_context}

# Instructions #
- Analyze why previous queries failed (wrong field, table, etc.)
- Propose a new working SQL query using only schema fields
- Do not make up names
- Return format: #NEW QUERY#: corrected SQL query
"""
            },
            {
                "role": "user",
                "content": f"""
User Question: {user_question}

# Failed Queries and Errors #
{error_log}
"""
            }
        ]
    )

    content = response.choices[0].message.content.strip()
    match = re.search(r'#NEW QUERY#:(.*)', content)
    return match.group(1).strip() if match else None
