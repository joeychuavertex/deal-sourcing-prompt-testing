import streamlit as st
import time
import pandas as pd

st.set_page_config(
    page_title="Authentication",
    page_icon="🔐",
    layout="wide"
)

st.write("# Authentication Page 🔐")
st.markdown("\n\n")

if 'key' not in st.session_state:
    st.session_state['key'] = 'api'

@st.cache_data
def loading_dfs():
    part1_df = pd.read_csv("20240318_final_data.csv").drop(columns=["Unnamed: 0"])
    part2_df= pd.read_csv("20240419_fulldata_final.csv").drop(columns=["Unnamed: 0"])
    return part1_df, part2_df
part1_df, part2_df = loading_dfs()

if 'part1_df' not in st.session_state:
        st.session_state['part1_df'] = part1_df
if 'part2_df' not in st.session_state:
    st.session_state['part2_df'] = part2_df

with st.form(key ='api_form'):
    api_key = st.text_input("Enter valid API key:")
    submitted = st.form_submit_button('Submit')
    if submitted: 
        if not api_key:
            st.error('Invalid API key!', icon="🚨")
            time.sleep(3)
            st.rerun()
        else:
            st.session_state['key'] = api_key
            st.success('Received', icon="✅")
