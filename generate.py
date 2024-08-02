import os
import pandas as pd
import json
import numpy as np
import ast
import re
import random
from prompts import relevance_score_prompt_zs, coherence_score_prompt_zs, aggressiveness_score_prompt_zs, suitableness_score_prompt_zs, common_input_prompt
from tqdm import tqdm
from openai import OpenAI
from together import Together

os.environ['http_proxy'] = "http://proxy61.iitd.ac.in:3128"
os.environ['https_proxy'] = "http://proxy61.iitd.ac.in:3128"

os.environ['TOGETHER_API_KEY'] = "90e216b11fbd62f9b0ce3a9666cbcb134a116857b3b62f95b6643bde91feb364"
os.environ['OPENAI_API_KEY'] = "caf5789914abb0b7d6fd76fec967706e5357000ecabdbdb37e4648a6f29b2f07"

openai_client = OpenAI(
  api_key=os.environ['OPENAI_API_KEY'] 
)

togetherai_client = Together(
    api_key=os.environ.get("TOGETHER_API_KEY")
)


def predict(
      prompt, 
      system_description,
      use_temperature_samping=False,
    #   model_name="gpt-4"
      model_name="mistralai/Mistral-7B-Instruct-v0.3"

):
    temperature = 0.5 if not use_temperature_samping else random.uniform(0, 0.5)

    if 'gpt' in model_name:
        response = openai_client.chat.completions.create(
        model=model_name,
        temperature=temperature,
        top_p=1,
        response_format = { "type": "json_object"},
        messages = [
                {"role": "system", "content": system_description},
                {"role": "user", "content": prompt},
            ],
        )
        content = response.choices[0].message.content
        return content
    
    else:
        response = togetherai_client.chat.completions.create(
            model=model_name,
            messages=[
                {"role": "system", "content": system_description},
                {"role": "user", "content": prompt}
            ],
            temperature=temperature,
            max_tokens=1024,
            top_p=1,
            frequency_penalty=0,
            presence_penalty=0,
            stop=["</s>"],
            logprobs=False,
            n=1
        )
        content = response.choices[0].message.content
        return content
        

def predict_openai(
      prompt, 
      system_description,
      use_temperature_samping=False,
      model_name="gpt-4"

):
    temperature = 0.5 if not use_temperature_samping else random.uniform(0, 0.5)

    response = openai_client.chat.completions.create(
    model=model_name,
    temperature=temperature,
    top_p=1,
    response_format = { "type": "json_object"},
    messages = [
            {"role": "system", "content": system_description},
            {"role": "user", "content": prompt},
        ],
    )
    content = response.choices[0].message.content
    return content

def predict_togetherai(
        input_prompt, 
        system_description, 
        use_temperature_samping=False,
        model_name="mistralai/Mistral-7B-Instruct-v0.3"
    ):

    temperature = 0.5 if not use_temperature_samping else random.uniform(0, 0.5)

    response = togetherai_client.chat.completions.create(
        model=model_name,
        messages=[
            {"role": "system", "content": system_description},
            {"role": "user", "content": input_prompt}
        ],
        temperature=temperature,
        max_tokens=1024,
        top_p=1,
        frequency_penalty=0,
        presence_penalty=0,
        stop=None,
        logprobs=False,
        n=1
    )

    return response.choices[0].message.content