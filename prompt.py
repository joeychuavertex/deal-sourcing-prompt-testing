from langchain.chains import LLMChain
from bs4 import BeautifulSoup
from langchain.chains import RetrievalQA
from selenium import webdriver 
from pandarallel import pandarallel
import multiprocessing
import numpy as np
import pandas as pd
from openai import BadRequestError
import json, time, nltk, tiktoken
from urllib.parse import urlparse, urlunparse
import streamlit as st
from langchain_community.callbacks import get_openai_callback
from langchain_community.document_loaders import SeleniumURLLoader
from langchain_community.vectorstores import Chroma
from langchain_core.prompts import PromptTemplate
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_text_splitters import CharacterTextSplitter

def prompt1run(selected_df, api_key, prompt1, input_variables):
    prompt_templates = {
        "startup": {
            "template": prompt1,
            "temperature": 0,
            "model_name": "gpt-3.5-turbo",
        },
    }

    def get_llm_response_1(row):
        print(f"Fetching for {row['name']}")
        score = 0
        reason = ""
        category = ""
        startup_prompt_config = prompt_templates.get("startup")
        prompt_template = startup_prompt_config.get("template")
        prompt = PromptTemplate(template=prompt_template, input_variables=input_variables)

        llm = ChatOpenAI(
            model_name=startup_prompt_config.get("model_name"),
            temperature=startup_prompt_config.get("temperature", 0),
            openai_api_key=api_key,
            model_kwargs={"response_format": {"type": "json_object"}},
        )

        chain = LLMChain(llm=llm, prompt=prompt)
        try:
            with get_openai_callback() as cb:
                response = chain.run(name=row["name"], description=row["description"], website_content=row["website_content"],
                                 sector=row["sector"], industry=row["industry"], website=row["website"],
                                 linkedin=row["linkedin"], comments=row["comments"], country=row["country"],
                                 revisit_at=row["revisit_at"], is_tech_startup=row["is_tech_startup"])
                print(response)
                print(cb)
        except BadRequestError as e:
            max_string_length = 30000  # Workaround truncate to fit max number of tokens
            response = chain.run(name=row["name"], action=row["action"], action_reason=row["action_reason"],
                                 description=row["description"], website_content=row["website_content"],
                                 sector=row["sector"], industry=row["industry"], website=row["website"],
                                 linkedin=row["linkedin"], comments=row["comments"], country=row["country"],
                                 lead_source=row["lead_source"], revisit_at=row["revisit_at"], is_tech_startup=row["is_tech_startup"][:max_string_length])

        try:
            json_response = json.loads(response, strict=False)
            score = json_response.get("score", 0)
            reason = json_response.get("reason", "")
            category = json_response.get("category", "")

        except Exception as e:
            score = 0
            reason = ""
            category = ""
            print(e)
        print(f"Completed {row['name']}")
        try:
            output = pd.Series({"tech_startup_likelihood": score, "reason": reason, "category": category})
            return output
        except UnboundLocalError as e:
            return pd.Series({"tech_startup_likelihood": 0, "reason": "", "category": ""})
        
    #if len(selected_df) <= multiprocessing.cpu_count(): # to circumvent ValueError: Number of processes must be at least 1 
    print(selected_df)
    list_df = selected_df.apply(get_llm_response_1, axis=1)
    list_df = pd.concat([selected_df,list_df], axis=1)
    '''else:
        pandarallel.initialize(progress_bar=True)
        list_df = np.array_split(selected_df, 10)
        chunked_dfs = []
        cnt = 0
        for chunked_df in list_df:
            chunked_df[["tech_startup_likelihood", "reason", "category"]] = chunked_df.parallel_apply(get_llm_response_1, axis=1)
            chunked_dfs.append(chunked_df)
            cnt += 1
            time.sleep(3)
        list_df = pd.concat(chunked_dfs)'''
    return list_df

def prompt2run(selected_df, api_key, prompt2, input_variables):
    prompt_templates = {
        "startup": {
            "template": prompt2,
            "temperature": 0,
            "model_name": "gpt-3.5-turbo",
        },
    }
    def get_llm_response_2(row):
        print(f"Fetching for {row['name']}")
        score = 0
        reason = ""
        startup_prompt_config = prompt_templates.get("startup")
        prompt_template = startup_prompt_config.get("template")
        prompt = PromptTemplate(template=prompt_template, input_variables=input_variables)

        llm = ChatOpenAI(
            model_name=startup_prompt_config.get("model_name"),
            temperature=startup_prompt_config.get("temperature", 0),
            openai_api_key=api_key,
            model_kwargs={"response_format": {"type": "json_object"}},
        )

        chain = LLMChain(llm=llm, prompt=prompt)
        try:
            with get_openai_callback() as cb:
                # response = chain.run(name=row["name"], description=row["description"], sector=row["sector"], industry=row["industry"], 
                #                     country=row["country"], website_content=row["website_content"], 
                #                     has_promising_founder=row["has_promising_founder"], has_strong_founder=row["has_strong_founder"], 
                #                     has_super_founder=row["has_super_founder"], website_traffic_estimates_chart=row["website_traffic_estimates_chart"], 
                #                     app_downloads_android_chart=row["app_downloads_android_chart"], job_openings=row["job_openings"], patents_count=row["patents_count"], 
                #                     last_updated=row["last_updated"], total_funding=row["total_funding"], dealroom_signal_rating=row["dealroom_signal_rating"], 
                #                     dealroom_signal_completeness=row["dealroom_signal_completeness"], dealroom_signal_team_strength=row["dealroom_signal_team_strength"], 
                #                     dealroom_signal_timing=row["dealroom_signal_timing"], employee_range=row["employee_range"], employee_chart=row["employee_chart"], 
                #                     specialities=row["specialities"], industries=row["industries"], technologies=row["technologies"], patents=row["patents"], 
                #                     news=row["news"], investors=row["investors"], raised_rounds=row["raised_rounds"], news_count=row["news_count"], 
                #                     raised_rounds_count=row["raised_rounds_count"], total_funds_raised=row["total_funds_raised"], round_kind=row["round_kind"], 
                #                     most_recent_round=row["most_recent_round"], employee_num=row["employee_num"], employee_percent_inc=row["employee_percent_inc"], 
                #                     website_traffic_abs_inc=row["website_traffic_abs_inc"], employee_abs_inc=row["employee_abs_inc"],
                #                     website_traffic_percent_inc=row["website_traffic_percent_inc"], investors_count=row["investors_count"],
                #                     product_offering=row['product_offering'], target_audience=row['target_audience'], problem_statement=row['problem_statement'],
                #                     founding_team=row['founding_team'], clients=row['clients'], followers_historical=row['followers_historical'],
                #                     followers_percent_inc=row['followers_percent_inc'], followers_abs_inc=row['followers_abs_inc'], news_sentiment=row['news_sentiment'],
                #                     percentile_employee_growth=row["percentile_employee_growth"], percentile_web_traffic_growth=row["percentile_web_traffic_growth"])
                # print(response)
                response = chain.run(name=row["name"], description=row["description"], sector=row["sector"], industry=row["industry"], 
                                    has_promising_founder=row["has_promising_founder"], has_strong_founder=row["has_strong_founder"], 
                                    has_super_founder=row["has_super_founder"], website_traffic_estimates_chart=row["website_traffic_estimates_chart"], 
                                    app_downloads_android_chart=row["app_downloads_android_chart"], job_openings=row["job_openings"], employee_range=row["employee_range"], employee_chart=row["employee_chart"], 
                                    specialities=row["specialities"], industries=row["industries"], technologies=row["technologies"], patents=row["patents"], 
                                    news=row["news"], investors=row["investors"], raised_rounds=row["raised_rounds"], employee_percent_inc=row["employee_percent_inc"], 
                                    website_traffic_abs_inc=row["website_traffic_abs_inc"], employee_abs_inc=row["employee_abs_inc"],
                                    website_traffic_percent_inc=row["website_traffic_percent_inc"], followers=row["followers"],
                                    percentile_employee_growth=row["percentile_employee_growth"], percentile_web_traffic_growth=row["percentile_web_traffic_growth"],
                                    percentile_employee_growth_sect=row["percentile_employee_growth_sect"], percentile_web_traffic_growth_sect=row["percentile_web_traffic_growth_sect"],
                                    percentile_employee_growth_fund=row["percentile_employee_growth_fund"], percentile_web_traffic_growth_fund=row["percentile_web_traffic_growth_fund"],
                                    raised_rounds_count=row["raised_rounds_count"])
                print(cb)
                print(response)
        except BadRequestError as e:
            max_string_length = 30000  # Workaround truncate to fit max number of tokens
            response = chain.run(name=row["name"], description=row["description"], sector=row["sector"], industry=row["industry"], 
                                    has_promising_founder=row["has_promising_founder"], has_strong_founder=row["has_strong_founder"], 
                                    has_super_founder=row["has_super_founder"], website_traffic_estimates_chart=row["website_traffic_estimates_chart"], 
                                    app_downloads_android_chart=row["app_downloads_android_chart"], job_openings=row["job_openings"], employee_range=row["employee_range"], employee_chart=row["employee_chart"], 
                                    specialities=row["specialities"], industries=row["industries"], technologies=row["technologies"], patents=row["patents"][:max_string_length], 
                                    news=row["news"], investors=row["investors"], raised_rounds=row["raised_rounds"], employee_percent_inc=row["employee_percent_inc"], 
                                    website_traffic_abs_inc=row["website_traffic_abs_inc"], employee_abs_inc=row["employee_abs_inc"],
                                    website_traffic_percent_inc=row["website_traffic_percent_inc"], followers=row["followers"],
                                    percentile_employee_growth=row["percentile_employee_growth"], percentile_web_traffic_growth=row["percentile_web_traffic_growth"],
                                    percentile_employee_growth_sect=row["percentile_employee_growth_sect"], percentile_web_traffic_growth_sect=row["percentile_web_traffic_growth_sect"],
                                    percentile_employee_growth_fund=row["percentile_employee_growth_fund"], percentile_web_traffic_growth_fund=row["percentile_web_traffic_growth_fund"],
                                    raised_rounds_count=row["raised_rounds_count"])
        try:
            json_response = json.loads(response, strict=False)
            score = json_response.get("score", 0)
            reason = json_response.get("reason", "")

        except Exception as e:
            score = 0
            reason = ""
            print(e)
        print(f"Completed {row['name']}")
        #try:
        output = pd.Series({"score": score, "reasoning": reason})
        return output
        # except UnboundLocalError as e:
        #     return pd.Series({"score": 0, "reasoning": ""})
        
    #if len(selected_df) <= multiprocessing.cpu_count():
    list_df = selected_df.apply(get_llm_response_2, axis=1)
    list_df = pd.concat([selected_df,list_df], axis=1)
    # else:
    #     pandarallel.initialize(progress_bar=True)
    #     list_df = np.array_split(selected_df, 6)
    #     chunked_dfs = []
    #     cnt = 0
    #     for chunked_df in list_df:
    #         chunked_df[["score", "reasoning"]] = chunked_df.parallel_apply(get_llm_response_2, axis=1)
    #         chunked_dfs.append(chunked_df)
    #         cnt += 1
    #         time.sleep(3)
    #     list_df = pd.concat(chunked_dfs)
    return list_df
    
def prompt3run(api_key, prompt3):
    prompt_templates = {
        "startup": {
            "template": prompt3,
            "temperature": 0,
            "model_name": "gpt-3.5-turbo",
        },
    }
    def get_llm_response_3():
        print(f"Fetching ...")
        names = []
        scores = []
        reasons = []
        startup_prompt_config = prompt_templates.get("startup")
        prompt_template = startup_prompt_config.get("template")
        prompt = PromptTemplate(template=prompt_template, input_variables=[])

        llm = ChatOpenAI(
            model_name=startup_prompt_config.get("model_name"),
            temperature=startup_prompt_config.get("temperature", 0),
            openai_api_key=api_key,
            model_kwargs={"response_format": {"type": "json_object"}},
        )
        chain = LLMChain(llm=llm, prompt=prompt)
        try:
            with get_openai_callback() as cb:
                response = chain.run({})
                print(response)
                print(cb)
        except BadRequestError as e:
            st.error('Please reduce the length of your final prompt! E.g. choose fewer features, edit base prompt.', icon="🚨")
            time.sleep(5)
            st.rerun()

        try:
            json_response = json.loads(response, strict=False)
            for company, details in json_response.items():
                names.append(company)
                scores.append(details['score'])
                reasons.append(details['reason'])
            print("names: ", names)
            print("scores: ", scores)
            print("reasons: ", reasons)
        except Exception as e:
            print(e)
        print(f"Completed!")
        
        try:
            output_dict = {'name': names, 'score': scores, 'reasoning': reasons}
            return output_dict
        except UnboundLocalError as e:
            output_dict = {'name': [], 'score': [], 'reasoning': []}
            return output_dict

    output_dict = get_llm_response_3() 
    return output_dict
