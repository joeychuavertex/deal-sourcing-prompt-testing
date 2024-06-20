import streamlit as st
import pandas as pd
import numpy as np

from st_aggrid import AgGrid, GridUpdateMode
from st_aggrid.grid_options_builder import GridOptionsBuilder
from sklearn import metrics
import matplotlib.pyplot as plt
from PIL import Image
from prompt import prompt1run
import openai, re, time
from openai import AuthenticationError
from datetime import datetime
    
st.set_page_config(page_title="Is Tech Startup?", layout="wide")
try:
    api_key = st.session_state['key'] if st.session_state.key else None
    part1_df = st.session_state['part1_df'] 
except AttributeError as e:
    st.switch_page("Authentication.py")

st.markdown("# Is Tech Startup?")
tab1_prompt, tab2_df = st.tabs(["Prompt", "Run"])
    
with tab1_prompt:
    st.warning("""In your prompt, please ensure that:\n 1. The input features are encapsulated within a pair of curly brackets (e.g. {description}) \n 2. The input features within the curly brackets are not empty, and must match its corresponding column name in the dataset as shown in the "Run" tab (e.g. {website_content})""", icon="⚠️")
    st.markdown("\n\n")
    default_prompt = """You are a startup scouter for a Venture Capital firm, looking for a tech startup to invest funds in. It is more important to exclude companies that:\n- Professional service providers like consultancy firms that might be involved in management consulting, strategy consulting, change management, branding, communication, growth hacking, digital-transformation, tech / IT consulting or sustainability advisory services\n- Professional service providers like market research or market insights firms - even if they are subject matter experts for high-tech topics like AI, Deep Tech, Web3.0 or claim to use advanced analytics. They may also have "IT Consulting and IT Services" or “Business Consulting and Services” as their industries.\nHowever, do classify them as tech startups if they are developing their own a software product/ database / marketplace instead of relying on only providing traditional consultation services\n- Professional service providers like agency businesses involved in talent recruiting, headhunting, advertising, traditional digital marketing agencies, traditional performance marketing agencies, PR agencies, design studios (such as but not limited to graphic and web design), game publishers, interior design studio, creative art studios\n- Investment management firms, asset management firms, wealth management firms, venture capital firms, venture funds, growth equity firms, private equity firms, family office, hedge funds, crypto funds, angel networks / syndicates that invest in tech startups but are not in the business of providing the tech related product or service. These companies would usually have the word "Ventures" in their company names. They may also have "Venture Capital and Private Equity Principles" as their industries. However, do not confuse them with startups in the wealth management or asset management space that leverage technology to democratize wealth management and that usually, but not always, provide robo-advisory, digital platforms that grant easy accessibility to financial products, and low-cost solutions for a broader audience. Similarly, do not confuse the investment firms themselves with startups creating tools that help these firms automate deal sourcing or other office functions.\n- Venture builders, venture studios, venture accelerators, venture incubators, venture partners that build or scale the growth of early-stage tech startups but are not in the business of providing the tech related product or service. These companies would usually have the word "Ventures" in their company names. They may also have "Venture Capital and Private Equity Principles" as their industries.\n- Traditional businesses like manufacturing and construction companies, distribution and wholesalers, data labelling firms, call centres, traditional therapy / psychologist / psychiatrist clinics that are not tech-enabled\n- Clubs, telegram groups, volunteer or youth development groups, NGOs, social enterprises, not-for-profits\n- Media publications, media and content production companies, editorials, articles, podcasts, newsletters, event organizers, networking events, traditional media agencies, social media channels, blogs\n- Academic institutions, academic thinktanks, educational workshops, digital course creator, mastermind groups, trade associations, mentoring circles, hobby clubs, corporate coaching programmes\n- Corporate spin-offs and subsidiaries\nA tech startup is a young company that focuses on using technology to create innovative products and services. These companies often aim to solve specific problems or disrupt traditional industries by leveraging technology in creative ways.\nYOU MUST TELL THE TRUTH. DO NOT LIE. DO NOT HALLUCINATE. DO NOT MAKE UP INFORMATION.\n1. On a score from 1 to 10, rate the likelihood that the company is a tech startup\n2. Provide a detailed reason of at most 500 words for giving this score\n3. Categorize the company into one of the following categories: Consumer, Enterprise, FinTech, B2B Marketplaces, HealthTech, Sustainability, Mobility, DeepTech, EdTech, Gaming, PropertyTech, Logistics, Others\n4. Return your result in JSON format with {{"score": <score>, "reason": <reason>, "category": <category>}}\nExample 1 - Company description: "<Company name> aims to <business objective>. We design innovative solutions that <product use>. We have raised seed funding from some investors in this region."\nWebsite content: <website content>\nIndustry: <industry>\nOutput 1 - {{"score": 9, "reason": <detailed reason>, "category": <category>}}\nExample 2 - Company description: "<Company name> invests in early-stage ventures using innovative technology."\nWebsite content: "We invest in emerging startups. <website content>"\nIndustry: "Venture Capital and Private Equity Principles"\nOutput 2 - {{"score": 1, "reason": "This is a venture capital firm, not a startup", "category": <category>}}\nExample 3 - Company description: "We are a startup building a cloud-based platform. Our mission is to deliver innovative technology to revolutionize the future."\nWebsite content: <website content>\nIndustry: <industry>\nOutput 3 - {{"score": 9, "reason": <detailed reason>, "category": <category>}}\nExample 4 - Company description: "<Company name> provides consultancy services to help startups conceptualise business strategies."\nWebsite content: "We are a consultancy." <website content>\nIndustry: <industry>\nOutput 4 - {{"score": 1, "reason": "This is a consultancy firm, not a startup", "category": <category>}}\nActual - Company description: ```{description}```\nWebsite content: ```{website_content}```\nIndustry: ```{industry}```"""
    prompt_df1 = pd.DataFrame([{"Edit prompt (press 'Enter' to save changes):": default_prompt}])
    edited_df1 = st.data_editor(prompt_df1, hide_index=True, use_container_width=True, height= 70)
    prompt1 = edited_df1["Edit prompt (press 'Enter' to save changes):"][0]
    st.markdown("\n\n")
    prompt_features = re.findall(r"(?<!\{)\{([^\{\}]*)\}", prompt1)
    st.write("Input features:", prompt_features)

with tab2_df:               
    gd1 = GridOptionsBuilder.from_dataframe(part1_df)
    gd1.configure_selection(selection_mode='multiple', use_checkbox=True)
    gd1.configure_column("name", headerCheckboxSelection = True)

    gridoptions1 = gd1.build()

    grid_table1 = AgGrid(part1_df, height=600, gridOptions=gridoptions1,
                        update_mode=GridUpdateMode.SELECTION_CHANGED,custom_css={
        "#gridToolBar": {
            "padding-bottom": "0px !important",
        }
    })
    
    selected_row1 = grid_table1["selected_rows"]

    # form a dataframe of the selected rows
    selected_df1 = pd.DataFrame(selected_row1)
    st.write(len(selected_df1), "out of", len(part1_df), "rows selected")
    
    if len(selected_df1) > 0 and st.button('Run'):
        if not api_key:
            st.error('Enter the API key in the Authentication page!', icon="🚨")
            time.sleep(3)
            st.switch_page("Authentication.py")
        try:
            with st.spinner('Running...'):
                start = time.time()
                result_df1 = prompt1run(selected_df1, api_key, prompt1, prompt_features)
                end = time.time()
                time_elapsed = "Time elapsed: " + str(round(end-start,2)) + "s"
                st.toast(time_elapsed, icon='⏱️')
                result_df1['is_tech_startup'] = np.where((result_df1['tech_startup_likelihood'] >= 5), 'Yes', 'No')
                st.markdown("---")
                st.markdown("# Results")
                st.markdown("### Metrics")
                accuracy = metrics.accuracy_score(result_df1['verdict'], result_df1['is_tech_startup'])
                st.write("Accuracy:", accuracy)
                f1 = metrics.f1_score(result_df1['verdict'], result_df1['is_tech_startup'], pos_label='Yes')
                st.write("F1-Score:", f1)
                st.write("Confusion Matrix:")
                cm = metrics.confusion_matrix(result_df1['verdict'], result_df1['is_tech_startup'], labels=["Yes", "No"])
                cm_display = metrics.ConfusionMatrixDisplay(confusion_matrix = cm, display_labels = [True, False])
                cm_display.plot()
                fig = plt.gcf()
                fig.savefig("cm_results.png")
                image = Image.open('cm_results.png')
                st.image(image,width=500)
                # creating final dataframe with is_tech_startup, score, reason, category, result [TP/TN/FP/FN]
                st.markdown("### Dataframe")
                conditions = [(result_df1['is_tech_startup'] == 'Yes') & (result_df1['verdict'] == 'Yes'),
                            (result_df1['is_tech_startup'] == 'No') & (result_df1['verdict'] == 'No'),
                            (result_df1['is_tech_startup'] == 'Yes') & (result_df1['verdict'] == 'No'),
                            (result_df1['is_tech_startup'] == 'No') & (result_df1['verdict'] == 'Yes')]
                labels = ['TP', 'TN', 'FP', 'FN']
                result_df1['result'] = np.select(conditions, labels, default='')
                cols_name = ['name', 'result', 'is_tech_startup', 'tech_startup_likelihood', 'reason', 'category']
                cols = cols_name + [ col for col in list(part1_df.columns.values) if col not in cols_name ] 
                result_df1 = result_df1[cols]
                
                result = GridOptionsBuilder.from_dataframe(result_df1)
                result_gridoptions = result.build()
                result_grid_table1 = AgGrid(result_df1, gridOptions=result_gridoptions, height = 500,
                                    update_mode=GridUpdateMode.SELECTION_CHANGED) 
                result_csv = result_df1.to_csv()              
                st.download_button(label="Download CSV",
                                    data=result_csv,
                                    file_name=f"{datetime.now().date().strftime('%Y%m%d')}_is_tech_startup_results.csv",
                                    mime='text/csv',)
        except AuthenticationError as e:
            st.error('AuthenticationError: Please enter a valid API key at the Authentication page.', icon="🚨")
            time.sleep(3)
            st.switch_page("Authentication.py")
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
            st.switch_page("Authentication.py")
    
    
    
