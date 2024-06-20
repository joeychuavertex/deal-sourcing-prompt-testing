import pandas as pd
import streamlit as st
from st_aggrid import AgGrid
from st_aggrid.grid_options_builder import GridOptionsBuilder

st.set_page_config(page_title="Is Tech Startup?", layout="wide")
st.markdown("# Extracted Website Information")

# displaying results from RAG:
#   - scrape data from websites of each company -> chunking -> store in vector database -> RAG using RetrievalQAChain (LangChain) + Cohere Reranker 
#   - objective: get more information on product offering, target audience, problem statement, founding team, clients
website_info = pd.read_csv('20240228_website_retrieval.csv').drop(columns=["Unnamed: 0"])

st.markdown("\n\n")
web = GridOptionsBuilder.from_dataframe(website_info)
web_gridoptions = web.build()
web_grid_table1 = AgGrid(website_info, gridOptions=web_gridoptions)  
