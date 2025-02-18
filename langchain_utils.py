# import os
# import pyodbc
# import streamlit as st
# import re
# from operator import itemgetter
# from dotenv import load_dotenv, find_dotenv

# from langchain.chains.openai_tools import create_extraction_chain_pydantic
# from langchain_core.pydantic_v1 import BaseModel, Field
# from langchain_openai import ChatOpenAI
# from langchain_groq import ChatGroq
# from langchain_huggingface import HuggingFaceEndpoint, HuggingFaceEmbeddings, HuggingFaceEndpointEmbeddings
# from langchain_community.utilities.sql_database import SQLDatabase
# from langchain.chains import create_sql_query_chain
# from langchain_community.tools.sql_database.tool import QuerySQLDataBaseTool
# from langchain_core.runnables import RunnablePassthrough, RunnableLambda
# from langchain.memory import ChatMessageHistory
# from langchain_core.output_parsers import StrOutputParser

# from table_details import table_chain as select_table
# from my_prompts import final_prompt, answer_prompt

# # Load environment variables
# load_dotenv(find_dotenv())

# api_key = os.environ["HF_TOKEN"]

# # Initialize language models
# llm_mistral = HuggingFaceEndpoint(repo_id=os.environ["repo_mistral"], temperature=0.2)
# llm_deepseek = HuggingFaceEndpoint(repo_id=os.environ["repo_id_deepSeek"], temperature=0.1)
# llm_meta = HuggingFaceEndpoint(repo_id=os.environ["repo_id_meta"], temperature=0.1)
# llm_groq = ChatGroq(model=os.environ["repo_id_groq"], api_key=os.environ["GROQ"])
# llm_openai = ChatOpenAI(model=os.environ["repo_id_openai"])

# #embeddings = HuggingFaceEndpointEmbeddings(model=os.environ["repo_embed"])

# LANGCHAIN_TRACING_V2 = os.getenv("LANGCHAIN_TRACING_V2")
# LANGCHAIN_API_KEY = os.getenv("LANGCHAIN_API_KEY")

# def clean_sql_query(text: str) -> str:
#     """
#     Clean SQL query by extracting the SQL code from code blocks or text,
#     removing code block syntax, various SQL tags, backticks, square brackets,
#     prefixes, and unnecessary whitespace while preserving the core SQL query.

#     Args:
#         text (str): Raw SQL query text that may contain code blocks, tags, and backticks

#     Returns:
#         str: Cleaned SQL query
#     """
#     # Step 1: Extract SQL from code blocks if present
#     code_block_pattern = r"```(?:sql|SQL|SQLQuery|mysql|postgresql)?\s*(.*?)\s*```"
#     code_blocks = re.findall(code_block_pattern, text, flags=re.DOTALL)
    
#     if code_blocks:
#         sql = code_blocks[0]
#     else:
#         # Step 2: Extract SQL statements without requiring semicolons
#         sql_statement_pattern = r"(?i)\b(SELECT|INSERT|UPDATE|DELETE)\b.*?(?=;|$)"
#         sql_match = re.search(sql_statement_pattern, text, flags=re.DOTALL)
#         sql = sql_match.group().strip() if sql_match else ""
    
#     # Step 3: Remove non-SQL prefixes (e.g., "AI:") and clean formatting
#     sql = re.sub(r'^\s*AI:\s*', '', sql, flags=re.IGNORECASE)  # Remove "AI:" prefix
#     sql = re.sub(r'`|\[|\]', '', sql)  # Remove backticks and brackets
#     sql = re.sub(r'\s+', ' ', sql).strip()  # Normalize whitespace
    
#     return sql

# @st.cache_resource
# def get_chain():
#     print("Creating chain")
#     print("Connecting Database")
#     server = os.environ["db_host"]
#     database = os.environ["db_name"]
#     connection_string = f'mssql+pyodbc://@{server}/{database}?driver=ODBC+Driver+17+for+SQL+Server&trusted_connection=yes'
#     db = SQLDatabase.from_uri(connection_string)
#     print("Database connected "+ db)

#     generate_query = create_sql_query_chain(llm_deepseek, db, final_prompt)
#     execute_query = QuerySQLDataBaseTool(db=db)
#     rephrase_answer = answer_prompt | llm_deepseek | StrOutputParser()

#     chain = (
#         RunnablePassthrough.assign(table_names_to_use=select_table) |
#         RunnablePassthrough.assign(query=generate_query | RunnableLambda(clean_sql_query)).assign(
#             result=itemgetter("query") | execute_query
#         ) |
#         rephrase_answer
#     )
#     return chain

# def create_history(messages):
#     history = ChatMessageHistory()
#     for message in messages:
#         if message["role"] == "user":
#             history.add_user_message(message["content"])
#         else:
#             history.add_ai_message(message["content"])
#     return history

# def invoke_chain(question,messages):
#     chain = get_chain()
#     history = create_history(messages)
#     response = chain.invoke({"question": question,"top_k":3,"messages":history.messages})
#     history.add_user_message(question)
#     history.add_ai_message(response)
#     return response

# # In langchain_utils.py
# try:
#     from table_details import table_chain as select_table
# except ImportError as e:
#     print(f"Error importing from table_details: {e}")

# try:
#     from my_prompts import final_prompt, answer_prompt
# except ImportError as e:
#     print(f"Error importing from prompts: {e}")


# # Example usage
# if __name__ == "__main__":
#     question = "Give me records of items stuck in carton error"
#     tables=select_table.invoke({"input":question"})
#     print(f"Relevant tables: {tables}")




import os
import re
from operator import itemgetter
import pyodbc
import streamlit as st
from dotenv import load_dotenv, find_dotenv
from langchain.chains.openai_tools import create_extraction_chain_pydantic
from pydantic import BaseModel, Field

from langchain_openai import ChatOpenAI
from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEndpoint, HuggingFaceEmbeddings, HuggingFaceEndpointEmbeddings
from langchain_community.utilities.sql_database import SQLDatabase
from langchain.chains import create_sql_query_chain
from langchain_community.tools.sql_database.tool import QuerySQLDataBaseTool
from langchain_core.runnables import RunnablePassthrough, RunnableLambda
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.output_parsers import StrOutputParser
from table_details import table_chain as select_table
from my_prompts import final_prompt, answer_prompt
from sqlalchemy import create_engine, text


# Load environment variables
load_dotenv(find_dotenv())

api_key = os.environ["HF_TOKEN"]

# Initialize language models
llm_mistral = HuggingFaceEndpoint(repo_id=os.environ["repo_mistral"], temperature=0.4)
llm_deepseek = HuggingFaceEndpoint(repo_id=os.environ["repo_id_deepSeek"], temperature=0.4)
llm_groq = ChatGroq(model=os.environ["repo_id_groq"], api_key=os.environ["GROQ"])
llm_openai = ChatOpenAI(model=os.environ["repo_id_openai"])

LANGCHAIN_TRACING_V2 = os.getenv("LANGCHAIN_TRACING_V2")
LANGCHAIN_API_KEY = os.getenv("LANGCHAIN_API_KEY")

def clean_sql_query(text: str) -> str:
    """
    Clean SQL query by extracting the SQL code from code blocks or text,
    removing code block syntax, various SQL tags, backticks, square brackets,
    prefixes, and unnecessary whitespace while preserving the core SQL query.

    Args:
        text (str): Raw SQL query text that may contain code blocks, tags, and backticks

    Returns:
        str: Cleaned SQL query
    """
    code_block_pattern = r"```(?:sql|SQL|SQLQuery|mysql|postgresql)?\s*(.*?)\s*```"
    code_blocks = re.findall(code_block_pattern, text, flags=re.DOTALL)
    
    if code_blocks:
        sql = code_blocks[0]
    else:
        sql_statement_pattern = r"(?i)\b(SELECT|INSERT|UPDATE|DELETE)\b.*?(?=;|$)"
        sql_match = re.search(sql_statement_pattern, text, flags=re.DOTALL)
        sql = sql_match.group().strip() if sql_match else ""
    
    sql = re.sub(r'^\s*AI:\s*', '', sql, flags=re.IGNORECASE)
    sql = re.sub(r'`|\[|\]', '', sql)
    sql = re.sub(r'\s+', ' ', sql).strip()
    
    return sql



@st.cache_resource
def get_chain():
    print("Creating chain")
    print("Connecting Database")
    server = st.secrets["server"]
    database = st.secrets["database"]
    connection_string = f'mssql+pyodbc://@{server}/{database}?driver=ODBC+Driver+17+for+SQL+Server&trusted_connection=yes'
    db = SQLDatabase.from_uri(connection_string)
    print(db)
    print("Database connected")

    generate_query = create_sql_query_chain(llm_openai, db, final_prompt)
    execute_query = QuerySQLDataBaseTool(db=db)
    rephrase_answer = answer_prompt | llm_openai | StrOutputParser()

    chain = (
        RunnablePassthrough.assign(table_names_to_use=select_table) |
        RunnablePassthrough.assign(query=generate_query | RunnableLambda(clean_sql_query)).assign(
            result=itemgetter("query") | execute_query
        ) |
        rephrase_answer
    )
    return chain



def create_history(messages):
    history = ChatMessageHistory()
    for message in messages:
        if message["role"] == "user":
            history.add_user_message(message["content"])
        else:
            history.add_ai_message(message["content"])
    return history

def invoke_chain(question,messages):
    chain = get_chain()
    history = create_history(messages)
    response = chain.invoke({"question": question,"top_k":3,"messages":history.messages})
    history.add_user_message(question)
    history.add_ai_message(response)
    return response



# if __name__ == "__main__":
#     question = "Give me count of all order"
#     tables = select_table.invoke({"question": question})
#     print(f"Relevant tables: {tables}")
#     get_chain()












