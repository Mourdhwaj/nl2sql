import os
import pandas as pd
import pyodbc
import streamlit as st
from operator import itemgetter
from typing import List
from dotenv import load_dotenv, find_dotenv

# LangChain imports
from langchain.chains.openai_tools import create_extraction_chain_pydantic

from pydantic import BaseModel, Field


from langchain_openai import ChatOpenAI
from langchain_groq import ChatGroq
from langchain_community.llms import HuggingFaceHub
from langchain_huggingface import HuggingFaceEndpoint, HuggingFaceEmbeddings, HuggingFaceEndpointEmbeddings

# Load environment variables
load_dotenv(find_dotenv())
# api_key = os.environ["HF_TOKEN"]

# Initialize language models
# llm_mistral = HuggingFaceEndpoint(repo_id=os.environ["repo_mistral"], temperature=0.2)
# llm_deepseek = HuggingFaceEndpoint(repo_id=os.environ["repo_id_deepSeek"], temperature=0.1)
# llm_meta = HuggingFaceEndpoint(repo_id=os.environ["repo_id_meta"], temperature=0.1)
llm_groq = ChatGroq(model=os.environ["repo_id_groq"], api_key=os.environ["GROQ"])
llm_openai = ChatOpenAI(model=os.environ["repo_id_openai"])



@st.cache_data
def get_table_details():
    # Read the CSV file into a DataFrame
    table_description = pd.read_csv("tables_description.csv")
    table_docs = []

    # Iterate over the DataFrame rows to create Document objects
    table_details = ""
    for index, row in table_description.iterrows():
        table_details = table_details + "Table Name:" + row['Table'] + "\n" + "Table Description:" + row['Description'] + "\n\n"

    return table_details


class Table(BaseModel):
    """Table in SQL database."""

    name: str = Field(description="Name of table in SQL database.")

def get_tables(tables: List[Table]) -> List[str]:
    tables  = [table.name for table in tables]
    return tables


# table_names = "\n".join(db.get_usable_table_names())
table_details = get_table_details()
table_details_prompt = f"""Return the names of ALL the SQL tables that MIGHT be relevant to the user question. \
The tables are:

{table_details}

Remember to include ALL POTENTIALLY RELEVANT tables, even if you're not sure that they're needed."""

table_chain = {"input": itemgetter("question")} | create_extraction_chain_pydantic(Table, llm_openai, system_message=table_details_prompt) | get_tables

# if __name__ == "__main__":
#     print("Table Details")