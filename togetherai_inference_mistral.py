import os
from together import Together
import pandas as pd
import json
import numpy as np
import ast
import re
from prompts import relevance_score_prompt_mistral, coherence_score_prompt_mistral, aggressiveness_score_prompt_mistral, suitableness_score_prompt_mistral, common_input_prompt
from tqdm import tqdm
os.environ['http_proxy'] = "http://proxy61.iitd.ac.in:3128"
os.environ['https_proxy'] = "http://proxy61.iitd.ac.in:3128"

# os.environ['TOGETHER_API_KEY'] = "4c7e2342b2a27509999bb6dc9d7afc4aeca46bb9861d48a649b279354b69db40"
# os.environ['TOGETHER_API_KEY'] = "caf5789914abb0b7d6fd76fec967706e5357000ecabdbdb37e4648a6f29b2f07"
os.environ['TOGETHER_API_KEY'] =  "90e216b11fbd62f9b0ce3a9666cbcb134a116857b3b62f95b6643bde91feb364"

client = Together(api_key=os.environ.get("TOGETHER_API_KEY"))
TEMPERATURE_SCALE = 0.5

def predict(input_prompt, system_description):

    response = client.chat.completions.create(
        model=model_name,
        messages=[
            {"role": "system", "content": system_description},
            {"role": "user", "content": input_prompt}
        ],
        temperature=TEMPERATURE_SCALE,
        max_tokens=32,
        top_p=1,
        frequency_penalty=0,
        presence_penalty=0,
        stop=None,
        logprobs=False,
        n=1
    )

    return response

model_name = "mistralai/Mistral-7B-Instruct-v0.3"

# For gold labels, comment otherwise
# keep_cols = ['hatespeech', 'counterspeech','predicted_counterspeech', 'uuid']
# input_df = pd.read_csv('/home/amey/depository/cs-eval/data/annotations/dataset_metrics_calculated.csv')
# input_df = input_df[(input_df['Mistral-7B-Instruct-v03_relevance_score'] == -1) & (input_df['predicted_counterspeech'].notna())]
# if 'level_0' in input_df.columns:
#     input_df = input_df.drop(columns=['level_0'])
# input_df = input_df.reset_index()
# print(f"Total data: {input_df.shape}")

# # input_df = input_df[keep_cols]
# # input_df = input_df.reset_index()
# print(input_df.shape)
# input_df.head(1)
# print(input_df.isna().sum())

# input_df[f'prediction_{model_name}_relevance_score'] = None
# input_df[f'prediction_{model_name}_coherence_score'] = None
# input_df[f'prediction_{model_name}_suitableness_score'] = None
# input_df[f'prediction_{model_name}_aggressiveness_score'] = None

model_name = "mistralai/Mistral-7B-Instruct-v0.3"

# For gold labels, comment otherwise
keep_cols = ['hatespeech', 'counterspeech', 'predicted_counterspeech', 'uuid']
input_df = pd.read_csv('/home/amey/depository/cs-eval/data/annotations/dataset_metrics_calculated.csv')
input_df = input_df[keep_cols]
# input_df = input_df.reset_index()
print(input_df.shape)
input_df.head(1)
print(input_df.isna().sum())

input_df[f'prediction_{model_name}_relevance_score'] = None
input_df[f'prediction_{model_name}_coherence_score'] = None
input_df[f'prediction_{model_name}_suitableness_score'] = None
input_df[f'prediction_{model_name}_aggressiveness_score'] = None

input_df = input_df[:1300]

print(f"Starting Run")

for i, row in tqdm(input_df.iterrows(), desc="Getting Predictions"):
    # if i > 1000: # Compute inference for first 2000 datapoints to start with
    #     print(f"Finishing run")
    #     break
        
    if row['predicted_counterspeech'] == None or not isinstance(row['predicted_counterspeech'], str) or len(row['predicted_counterspeech']) < 1: # Skip instances where there is no predicted_counterspeech
        print(f"skipping data point: {i}")
        continue

    if not pd.isna(row[f'prediction_{model_name}_relevance_score']):
        print(f"prediction already there, skipping data point: {i}")
        continue

    prompt = common_input_prompt(row['hatespeech'], row['predicted_counterspeech'])
    
    system_description = relevance_score_prompt_mistral
    content = predict(prompt, system_description)
    input_df.at[i,f'prediction_{model_name}_relevance_score'] = content

    system_description = aggressiveness_score_prompt_mistral
    content = predict(prompt, system_description)  
    input_df.at[i,f'prediction_{model_name}_aggressiveness_score'] = content

    system_description = coherence_score_prompt_mistral
    content = predict(prompt, system_description)  
    input_df.at[i,f'prediction_{model_name}_coherence_score'] = content
    
    system_description = suitableness_score_prompt_mistral
    content = predict(prompt, system_description)  
    input_df.at[i,f'prediction_{model_name}_suitableness_score'] = content
        
    if i % 100 == 0:
        print('-'*20)
        print(f'Saving Results till -- {i}')
        fname = model_name.replace('/','-')
        input_df.to_pickle(f'/home/amey/depository/cs-eval/predictions_rebuttal/{fname}_temperature_{TEMPERATURE_SCALE}.pkl')
        input_df.to_csv(f'/home/amey/depository/cs-eval/predictions_rebuttal/{fname}_temperature_{TEMPERATURE_SCALE}.csv', index=False)
        print('-'*20)

print(f"Finished! Saving final results!")
fname = model_name.replace('/','-')
input_df.to_pickle(f'/home/amey/depository/cs-eval/predictions_rebuttal/{fname}_temperature_{TEMPERATURE_SCALE}.pkl')
input_df.to_csv(f'/home/amey/depository/cs-eval/predictions_rebuttal/{fname}_temperature_{TEMPERATURE_SCALE}.csv', index=False)