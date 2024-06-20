import streamlit as st
import pandas as pd
import numpy as np

from st_aggrid import AgGrid, GridUpdateMode
from st_aggrid.grid_options_builder import GridOptionsBuilder
import openai, time, re
from prompt import prompt2run
from openai import AuthenticationError

st.set_page_config(page_title="Scoring", layout="wide")

try:
    api_key = st.session_state['key'] if st.session_state.key else None
    part2_df = st.session_state['part2_df']
except AttributeError as e:
    st.switch_page("pages/Authentication.py")

st.markdown("# Scoring (Individual)")
tab1_prompt, tab2_df, tab3_stats  = st.tabs(["Prompt", "Run", "Statistics"])
    
with tab1_prompt:
    st.warning("""In your prompt, please ensure that:\n 1. The input features are encapsulated within a pair of curly brackets (e.g. {country}) \n 2. The input features within the curly brackets are not empty, and must match its corresponding column name in the dataset as shown in the "Run" tab (e.g. {website_content})""", icon="⚠️")
    st.markdown("\n\n")
    default_prompt = '''You are advising a venture capital investor focusing on Series A start-ups. You will be given a set of information and statistics a tech start-up. Your task is to assess the likelihood that this company exhibits breakthrough signals, which makes it a good investment target to recommend to the investor. You will give:
- A likelihood score that the company exhibits breakthrough signals, ranging from 0 (for no likelihood) to 100 (maximum likelihood).
- A detailed reason for giving this score in seven bullet points, quoting relevant statistics and evidence. Every bullet point must correspond to each criterion given below and must contain why it has or has not shown breakthrough signals for that particular criterion.
Your output MUST be in JSON format as follows: {{"score" : <score>, "reason": "- <reason 1>\n...\n- <reason 7>"}}.

You are given the following general information about the company:
 - Industries: {industry}
 - Description: {description}
 
If there is no information given for a criteria, then it does not show breakthrough potential. Carefully analyse the information given and determine whether the company described has made a breakthrough based on the following criteria:

1. Number of employees: If the company has been hiring more employees, with a significant increase over the recent period, it is more likely a breakthrough.
- Employees time series: {employee_chart}
- Current job openings: {job_openings}

2. Web traffic: If the company website traffic has increased significantly over the recent period, it is more likely a breakthrough.
- Website traffic time series: {website_traffic_estimates_chart}

3. Social media presence: If the company has a lot of followers on LinkedIn, it is gaining traction and is more likely a breakthrough.
- LinkedIn followers: {followers}

4. Funding: If the company has raised funds in recent years, it is more likely a breakthrough. Furthermore, the following should be viewed more positively: (i) if the company's most recent fundraising round was in the past three years, (ii) if the company has raised more than one funding round to date, (iii) if the total funds raised is more than US $1 million, (iv) if their investors include institutional investors.
- Raised rounds: {raised_rounds}

5. News exposure: If the company has been mentioned more frequently in news reports, such as for reaching certain fundraising or business milestones, it is more likely a breakthrough.
- News: {news}

6. Patents: If the company has obtained patents, it is more likely to have a breakthrough. This is more critical for companies in deep tech sectors such as healthcare, sustainability, and AI, though patents should be viewed positively in any sector.
- Patents: {patents}

7. Founder profile: If the company has a "super founder", a "strong founder" or a "promising founder", it is more likely to have a breakthrough. A "super founder" holds more weight over a "strong founder", which in turn holds more weight over a "promising founder" (tiered based on prior entrepreneurship experience), which is better than none of the above. Furthermore, if the founder has experience relevant to the industries where the company operates in, this is advantageous too.
- Has super founder?: {has_super_founder}
- Has strong founder?: {has_strong_founder}
- Has promising founder?: {has_promising_founder}'''
    
    prompt_df2 = pd.DataFrame([{"Edit prompt (press 'Enter' to save changes):": default_prompt}])
    edited_df2 = st.data_editor(prompt_df2, hide_index=True, use_container_width=True)
    prompt2 = edited_df2["Edit prompt (press 'Enter' to save changes):"][0]
    st.markdown("\n\n")
    prompt_features = re.findall(r"(?<!\{)\{([^\{\}]*)\}", prompt2)
    st.write("Input features:", prompt_features)

with tab2_df:
    gd2 = GridOptionsBuilder.from_dataframe(part2_df)
    gd2.configure_selection(selection_mode="multiple", use_checkbox=True)
    gd2.configure_column("name", headerCheckboxSelection = True)
    gridoptions2 = gd2.build()
    grid_table2 = AgGrid(part2_df, height=500, gridOptions=gridoptions2,
                        update_mode=GridUpdateMode.SELECTION_CHANGED,custom_css={
        "#gridToolBar": {
            "padding-bottom": "0px !important",
        }
    })

    selected_row2 = grid_table2["selected_rows"]
    selected_df2 = pd.DataFrame(selected_row2)
    st.write(len(selected_df2), "out of", len(part2_df), "rows selected")

    if len(selected_df2) > 0 and st.button('Run'):
        if not api_key:
            st.error('Enter the API key in the Authentication page!', icon="🚨")
            time.sleep(3)
            st.switch_page("pages/Authentication.py")
        try:
            with st.spinner('Running...'):
                start = time.time()
                result_df2 = prompt2run(selected_df2, api_key, prompt2, prompt_features)
                end = time.time()
                time_elapsed = "Time elapsed: " + str(round(end-start,2)) + "s"
                st.toast(time_elapsed, icon='⏱️')
                result_df2["rank"] = result_df2["score"].rank(method='first', ascending=False) # ranking based on likelihood score
                st.markdown("# Results")
                cols_name = ['name', 'score', 'reasoning', 'rank']
                cols = cols_name + [ col for col in list(result_df2.columns.values) if col not in cols_name ] 
                result_df2 = result_df2[cols]
                st.write(result_df2)
                
        except AuthenticationError as e:
            st.error('AuthenticationError: Please enter a valid API key at the Authentication page.', icon="🚨")
            time.sleep(3)
            st.switch_page("pages/Authentication.py")
        except ValueError as e:
            st.error("ValueError: Ensure that all {input features} in the prompt are identical to their corresponding column names in the dataframe and not empty!", icon="🚨")
            time.sleep(7)
            st.rerun()
        except KeyError as e:
            st.error("KeyError: Ensure that all {input features} in the prompt are identical to their corresponding column names in the dataframe!", icon="🚨")
            time.sleep(5)
            st.rerun()
        except SyntaxError as e:
            st.error("SyntaxError: Ensure that all {input features} in the prompt are entered correctly.", icon="🚨")
            time.sleep(5)
            st.rerun()
        except NameError as e:
            st.error('NameError: Please enter a valid API key at the Authentication page.', icon="🚨")
            time.sleep(3)
            st.switch_page("pages/Authentication.py")

with tab3_stats:
    st.markdown("\n\n ##### Sparsity (% of empty values)")
    part2_df_copy = part2_df.copy()
    part2_df_copy.replace(['[]', '{}'], np.nan, inplace=True)
    all_cols = ['has_promising_founder', 'has_strong_founder', 'has_super_founder', 
    'website_traffic_estimates_chart', 'job_openings', 'followers', 'employee_chart', 'patents', 'news', 'raised_rounds',
    'employee_percent_inc', 'website_traffic_percent_inc']
    sparse_stats = pd.DataFrame(columns=all_cols)
    for col in all_cols:
        sparse_stats[col] = [round((part2_df_copy[col].isnull().sum()/len(part2_df_copy) * 100), 2)]
    st.write(sparse_stats)
    
    st.markdown("\n\n ##### Numerical Variables")
    num_cols = ['job_openings', 'followers', 'employee_percent_inc','website_traffic_percent_inc']
    num_stats = pd.DataFrame(columns=num_cols)
    for col in num_cols:
        stats = part2_df_copy[col].describe()
        num_stats[col] = [stats.iloc[3], stats.iloc[4], round(stats.iloc[5],2), stats.iloc[6], stats.iloc[7]]

    num_stats.insert(0,'stats', ['min', '25%', 'mean', '75%', 'max'])
    st.write(num_stats)
