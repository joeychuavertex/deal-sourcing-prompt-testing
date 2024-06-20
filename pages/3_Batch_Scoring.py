import streamlit as st
import pandas as pd
import numpy as np
from pandarallel import pandarallel
from st_aggrid import AgGrid, GridUpdateMode
from st_aggrid.grid_options_builder import GridOptionsBuilder
import openai, time, re, nltk, tiktoken
nltk.download('punkt')
from prompt import prompt3run
from openai import AuthenticationError
from datetime import datetime

def num_tokens_from_string(string: str, encoding_name: str) -> int:
    encoding = tiktoken.get_encoding(encoding_name)
    num_tokens = len(encoding.encode(string))
    return num_tokens

# ensure startup data is being formatted as a json string + companies are in random sequence for prompt
def making_prompt(selected_df3, options, run_method, random_state): 
    selected_df3 = selected_df3.sample(frac=1, random_state=random_state).reset_index(drop=True) # shuffle the selected companies randomly
    name_list = []
    all_col_mappings = {'name': 'company_name', 'country': 'country', 'sector': 'sector', 'description': 'company_description',
                        'has_promising_founder': 'has_promising_founder', 
                       'has_strong_founder': 'has_strong_founder', 'has_super_founder': 'has_super_founder', 
                       'website_traffic_estimates_chart': 'website_traffic_trend', 'app_downloads_android_chart': 'app_downloads_trend', 
                       'job_openings': 'job_openings', 'employee_chart': 'employee_headcount_trend', 'industry': 'industry',
                       'news': 'news', 'investors': 'investors', 'raised_rounds': 'raised_funding_rounds', 
                       'employee_percent_inc': 'employee_growth_percentage_increase', 'followers': 'linkedin_followers',
                       'website_traffic_percent_inc': 'website_traffic_growth_percentage_increase'}
    if 'name' not in options: options.append('name')
    if 'sector' not in options: options.append('sector')
    prompt_col_mappings = {key: all_col_mappings[key] for key in options if key in all_col_mappings}
    final_prompts = []
    final_formatted_prompt = '['
    total_tokens = 0
    if run_method == "Batch":
        k = 0
        for i in range(len(selected_df3)):
            if k >= MAX_BATCH_SIZE or total_tokens > 14000: 
                final_formatted_prompt = final_formatted_prompt[:-2] + ']'
                final_prompts.append(final_formatted_prompt)
                final_formatted_prompt = '['
                k, total_tokens=0, 0

            formatted_prompt= f'{{{{"id": {k+1}, '
            for feature, display_name in prompt_col_mappings.items():
                if feature == 'description' and not isinstance(selected_df3.loc[i, "description"], float) and selected_df3.loc[i, "description"]:
                    selected_df3.loc[i, "description"] = ' '.join(selected_df3.loc[i, "description"].split()) # minimise wastage of tokens
                formatted_prompt += f'"{display_name}": "{selected_df3.loc[i, feature]}", '
            formatted_prompt = formatted_prompt[:-2] + "}}, "
            total_tokens += num_tokens_from_string(formatted_prompt, "cl100k_base")
            k += 1
            final_formatted_prompt += formatted_prompt

        if final_formatted_prompt:
            final_formatted_prompt = final_formatted_prompt[:-2] + ']'
            final_prompts.append(final_formatted_prompt)
    # else:
    #     grouped = selected_df3.groupby('sector')
    #     for sector in selected_df3["sector"].unique():
    #         sector_df = grouped.get_group(sector).reset_index(drop=True)
    #         formatted_prompt = f"Sector: {sector}"
    #         name_list.extend(sector_df["name"].values.tolist())
    #         k = 0
    #         for i in range(len(sector_df)):
    #             formatted_prompt+= f"\n===Company {k + 1}===\n"
    #             k+=1
    #             for feature, display_name in prompt_col_mappings.items():
    #                 if feature == 'description' and not isinstance(sector_df.loc[i, "description"], float) and sector_df.loc[i, "description"]:
    #                     sector_df.loc[i, "description"] = ' '.join(sector_df.loc[i, "description"].split())
    #                 formatted_prompt += f"- {display_name}: {sector_df.loc[i, feature]}\n"

    #             if k >= MAX_BATCH_SIZE:
    #                 final_prompts.append(formatted_prompt)
    #                 formatted_prompt = f"Sector: {sector}\n"
    #                 k=0
    #         if formatted_prompt:
    #             final_prompts.append(formatted_prompt)        

    return final_prompts, name_list, selected_df3

st.set_page_config(page_title="Scoring (Batch)", layout="wide")
MAX_BATCH_SIZE = 20

st.markdown("# Scoring (Batch)")

try:
    api_key = st.session_state['key'] if st.session_state.key else None
    part2_df = st.session_state['part2_df'] 
    part2_df = part2_df[:10] #if the selection of the companies gets wonky (e.g. keeps refreshing), probably because data is too big
except AttributeError as e:
    st.switch_page("Authentication.py")

curly_cols = ["website_traffic_estimates_chart", "app_downloads_android_chart", "employee_range", "employee_chart", "patents", "news", "raised_rounds"]

for col in curly_cols:
    part2_df[col] = part2_df[col].str.replace("}","}}")
    part2_df[col] = part2_df[col].str.replace("{","{{")
    
st.markdown("#### Step 1")
default_context = """You are advising a venture capital investor focusing on Series A start-ups. You will be given a set of information and statistics about at most 30 companies, which are tech start-ups. Your task is to compare all the companies carefully, and assess the likelihood that each company exhibits breakthrough signals relative to other companies. For every company you are provided with, you must ouput: \n- A likelihood score that each company exhibits breakthrough signals, ranging from 0 (for no likelihood) to 100 (maximum likelihood)\n- A detailed reason for giving this score. You must quote relevant statistics and evidence from the information provided to support your answer. Your reason must be more than 100 words. Do not make evidence up. \nYou must answer for every company. Your output MUST be in JSON format as follows: {{"company_name": {{"score": <score>, "reason": <reason>}},...,"company_name": {{"score" : <score>, "reason": <reason>}}}}.\nDetermine whether the company described has made a breakthrough based on the following criteria:\n- Size: if the company is hiring more people, especially engineering talent, with significance compared to its current team size, it is more like a breakthrough.\n- Web traffic: if the company website traffic has increased significantly, it is more likely a breakthrough.\n- Funding: if the company has raised a new funding round, it is more likely a breakthrough.\n- News exposure: If the company has been mentioned more significantly in social media, it is more likely a breakthrough.\n- Founding team: If the founding team has relevant past experience to the startup they are running now, it is more likely to be a breakthrough.\n- Your own opinion\nThink step by step. Do not hallucinate. Do not make things up. You must output an answer for every single company given to you. You are given the following information about the different companies:"""
prompt_df3 = pd.DataFrame([{"Edit base prompt (press 'Enter' to save changes):": default_context}])
edited_df3 = st.data_editor(prompt_df3, hide_index=True, use_container_width=True)
prompt3 = edited_df3["Edit base prompt (press 'Enter' to save changes):"][0]
st.markdown("\n\n")

st.markdown("#### Step 2")
gd3 = GridOptionsBuilder.from_dataframe(part2_df)
gd3.configure_selection(selection_mode='multiple', use_checkbox=True)
gd3.configure_column("name", headerCheckboxSelection = True)
gridoptions3 = gd3.build()
grid_table3 = AgGrid(part2_df, height=500, gridOptions=gridoptions3,
                    update_mode=GridUpdateMode.SELECTION_CHANGED,custom_css={
        "#gridToolBar": {
            "padding-bottom": "0px !important",
        }
    })
selected_row3 = grid_table3["selected_rows"]
selected_df3 = pd.DataFrame(selected_row3)
st.write(len(selected_df3), "out of", len(part2_df), "rows selected")
st.markdown('\n\n')

st.markdown("#### Step 3")
options = st.multiselect(
    "Choose which features of the companies are to be fed into the model:",
    ['name', 'sector', 'description', 'has_promising_founder', 'has_strong_founder', 'has_super_founder', 
    'website_traffic_estimates_chart', 'job_openings', 'employee_chart', 'industry', 'followers',
    'news', 'investors', 'raised_rounds', 'employee_percent_inc', 'website_traffic_percent_inc', 'patents'],
    ["industry", "description", "employee_chart", "job_openings", "website_traffic_estimates_chart", "followers",
    "raised_rounds", "news", "patents", "has_super_founder", "has_strong_founder", "has_promising_founder"])
st.markdown("\n\n")

st.markdown("#### Step 4")
#run_method = st.selectbox("Run by...", ("Batch", "Sector"), index=None, placeholder="",)
run_method = "Batch"
final_prompt3 = []
if run_method:
    random_state=1
    final_prompt3, name_list, shuffled_df = making_prompt(selected_df3, options, run_method, random_state)
    shuffled_df.to_csv(f"{datetime.now().date().strftime('%Y%m%d')}_shuffle_{random_state}_part2_data.csv")
    for i in range(len(final_prompt3)): 
        final_prompt3[i] = prompt3 + "\n" + final_prompt3[i]
        if num_tokens_from_string(final_prompt3[i], "cl100k_base") > 14000:
            st.error('Please reduce the length of your final prompt! E.g. choose fewer features, edit base prompt.', icon="🚨")
            st.markdown('\n\n')
st.markdown('\n\n')
with st.expander("Expand to see your final prompt:"):
    with st.container(height=500):
        st.write(final_prompt3)
st.markdown('\n\n')

if len(selected_df3) > 0 and run_method and st.button('Run'):
    if not api_key:
        st.error('Enter the API key in the Authentication page!', icon="🚨")
        time.sleep(3)
        st.switch_page("Authentication.py")
    try:
        with st.spinner('Running...'):
            start = time.time()
            result = pd.DataFrame(columns=["name", "score", "reasoning"])
            for i in range(len(final_prompt3)):
                result_df3 = prompt3run(api_key, final_prompt3[i])
                result_df3 = pd.DataFrame.from_dict(result_df3)
                result = pd.concat([result, result_df3], axis=0).reset_index(drop=True)
            end = time.time()
            time_elapsed = "Time elapsed: " + str(round(end-start,2)) + "s"
            st.toast(time_elapsed, icon='⏱️')
            if run_method == "Batch":
                result = result.reset_index(drop=True)
                final_df = selected_df3.merge(result,on='name',how='left')
                final_df["rank"] = final_df["score"].rank(method='first', ascending=False) # rank by likelihood score within the batch
                st.markdown("# Results")
                cols_name = ['name', 'score', 'reasoning', 'rank']
                cols = cols_name + [col for col in list(final_df.columns.values) if col not in cols_name] 
                final_df = final_df[cols]
                st.dataframe(final_df, use_container_width=True)
                final_df_copy = final_df.copy()
                final_df['score'] = final_df['score'].fillna(0)
                if (final_df['score'] == 0).all():
                    st.warning('You are seeing this result because the prompt is too long. Please reduce the length of your final prompt by choosing fewer features, editing base prompt, etc.', icon="⚠️")
            # elif run_method == "Sector" and name_list: # ranking them based on l score in each sector
            #     name_df = pd.DataFrame({"name": name_list})
            #     merged_df = pd.concat([name_df, result], axis=1)
            #     merged_df.drop(columns=['company_name'], inplace=True)
            #     final_df = selected_df3.merge(merged_df,on='name', how='left')
            #     grouped = final_df.groupby('sector')
            #     st.markdown("# Results")
            #     for sector in selected_df3["sector"].unique():
            #         final_sector_df = grouped.get_group(sector).reset_index(drop=True)
            #         final_sector_df["rank"] = final_sector_df["score"].rank(method='first', ascending=False)
            #         cols_name = ['name', 'score', 'reasoning', 'rank']
            #         cols= cols_name + [ col for col in list(final_sector_df.columns.values) if col not in cols_name ] 
            #         final_sector_df = final_sector_df[cols]
            #         st.markdown(f"## {sector} sector:")
            #         st.dataframe(final_sector_df, use_container_width=True)
            #     final_sector_df_copy = final_sector_df.copy()
            #     final_sector_df_copy['score'] = final_sector_df_copy['score'].fillna(0)
            #     if (final_sector_df_copy['score'] == 0).all():
            #         st.warning('You are seeing this result because the prompt is too long. Please reduce the length of your final prompt by choosing fewer features, editing base prompt, etc.', icon="⚠️")
    except AuthenticationError as e:
        st.error('AuthenticationError: Please enter a valid API key at the Authentication page.', icon="🚨")
        time.sleep(3)
        st.switch_page("Authentication.py")
    except SyntaxError as e:
        st.error("SyntaxError: Ensure that all {input features} in the prompt are entered correctly.", icon="🚨")
        time.sleep(5)
        st.rerun()
    except NameError as e:
        st.error('NameError: Please enter a valid API key at the Authentication page.', icon="🚨")
        time.sleep(3)
        st.switch_page("Authentication.py")
