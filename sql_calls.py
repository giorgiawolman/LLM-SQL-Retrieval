import sqlite3
import pandas as pd
import re
from llm_calls import fix_sql_query

# 🔹 Get full schema from a given SQLite database
def get_dB_schema(dB_path):
    conn = sqlite3.connect(dB_path)
    cursor = conn.cursor()
    schema_info = {}

    cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
    table_names = cursor.fetchall()

    for (table_name,) in table_names:
        cursor.execute(f"PRAGMA table_info({table_name});")
        columns = [col[1] for col in cursor.fetchall()]
        schema_info[table_name] = columns

    conn.close()
    return schema_info

# 🔹 Format schema + sample rows for LLM prompt
def format_dB_context(dB_path, filtered_schema: dict) -> str:
    def fetch_example_rows(db_path, table_name):
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        cursor.execute(f"SELECT * FROM {table_name} ORDER BY RANDOM() LIMIT 3")
        rows = cursor.fetchall()
        conn.close()
        return rows

    chunks = []
    for table_name, columns in filtered_schema.items():
        sample_rows = fetch_example_rows(dB_path, table_name)
        df = pd.DataFrame(sample_rows, columns=columns)

        chunk = f"""CREATE TABLE "{table_name}" ({', '.join(columns)})
        /*
        {df.to_string(index=False)}
        */
        """
        chunks.append(chunk)

    return "\n".join(chunks)

# 🔹 Run a direct SQL query
def execute_sql_query(dB_path, sql_query):
    conn = sqlite3.connect(dB_path)
    cursor = conn.cursor()
    cursor.execute(sql_query)
    result = cursor.fetchall()
    conn.close()
    return result

# 🔁 Retry a SQL query with LLM fix suggestions
def fetch_sql(sql_query, dB_context, user_question, dB_path):
    attempt = 1
    max_retries = 3
    attempted_queries = []
    exceptions = []

    while attempt <= max_retries:
        try:
            print("____________________")
            print(f"Execute Attempt {attempt}/{max_retries}")
            sql_result = execute_sql_query(dB_path, sql_query)

            if not sql_result or str(sql_result) == "[(0,)]":
                error = "Query returned empty."
                attempted_queries.append(sql_query)
                exceptions.append(error)

                sql_query = fix_sql_query(dB_context, user_question, attempted_queries, exceptions)
                print(f"Query result: EMPTY. Trying new query:\n{sql_query}")
                attempt += 1
                continue
            else:
                print("✅ SQL query returned valid result.")
                return sql_query, sql_result

        except Exception as e:
            error = str(e)
            attempted_queries.append(sql_query)
            exceptions.append(error)

            sql_query = fix_sql_query(dB_context, user_question, attempted_queries, exceptions)
            print(f"Query result: ERROR. Trying new query:\n{sql_query}")
            attempt += 1

    return None, "❌ Failed after multiple attempts"
