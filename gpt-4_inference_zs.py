import pandas as pd
import json
import numpy as np
import tqdm as tqdm
import ast
import re
from openai import OpenAI
import os
from prompts import relevance_score_prompt_gpt4, coherence_score_prompt_gpt4, aggressiveness_score_prompt_gpt4, suitableness_score_prompt_gpt4, common_input_prompt
from tqdm import tqdm


model_name = 'gpt-4o'

# For gold labels, comment otherwise
# keep_cols = ['hatespeech', 'counterspeech','predicted_counterspeech', 'uuid']
# For gold labels, comment otherwise
keep_cols = ['hatespeech', 'counterspeech','predicted_counterspeech', 'uuid']
input_df = pd.read_csv('/home/amey/depository/cs-eval/data/annotations/dataset_metrics_calculated.csv')

input_df = input_df[keep_cols]
# input_df = input_df.reset_index()
print(input_df.shape)
input_df.head(1)
print(input_df.isna().sum())

input_df = input_df[:1300]

input_df[f'prediction_{model_name}_relevance_score'] = None
input_df[f'prediction_{model_name}_coherence_score'] = None
input_df[f'prediction_{model_name}_suitableness_score'] = None
input_df[f'prediction_{model_name}_aggressiveness_score'] = None

client = OpenAI(
  # api_key=os.environ['OPENAI_API_KEY'],
  # api_key = "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",
  # api_key="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",
#   api_key = "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",
  # api_key= "sk-proj-QRL4wrtuEYN7QUH9d7bwT3BlbkFJqkVbpLOhyiz9TiG6lAUF"
  api_key="sk-proj-SWPkbjlG7ZO9tAKANL69T3BlbkFJ3iiUUZHszD9AuQnEjhLE"
)

def openai_response(prompt, system_description):
  response = client.chat.completions.create(
  model=model_name,
  temperature=1,
  top_p=1,
  response_format = { "type": "json_object"},
  messages = [
        {"role": "system", "content": system_description},
        {"role": "user", "content": prompt},
    ],
  )
  content = response.choices[0].message.content
  return content

print(f"Starting Run")

for i, row in tqdm(input_df.iterrows(), desc="Getting Predictions"):
    if row['predicted_counterspeech'] == None or not isinstance(row['predicted_counterspeech'], str) or len(row['predicted_counterspeech']) < 1: # Skip instances where there is no predicted_counterspeech
        continue

    # if not pd.isna(row[f'gpt-4_relevance_score']):
    #     print(row[f'gpt-4_relevance_score'])
    #     print(f"skipping data point: {i}")
    #     continue

    prompt = common_input_prompt(row['hatespeech'], row['predicted_counterspeech'])
    
    system_description = relevance_score_prompt_gpt4
    content = openai_response(prompt, system_description)
    input_df.at[i,f'prediction_{model_name}_relevance_score'] = content

    system_description = aggressiveness_score_prompt_gpt4
    content = openai_response(prompt, system_description)  
    input_df.at[i,f'prediction_{model_name}_aggressiveness_score'] = content

    system_description = coherence_score_prompt_gpt4
    content = openai_response(prompt, system_description)  
    input_df.at[i,f'prediction_{model_name}_coherence_score'] = content
    
    system_description = suitableness_score_prompt_gpt4
    content = openai_response(prompt, system_description)  
    input_df.at[i,f'prediction_{model_name}_suitableness_score'] = content

    if i % 100 == 0:
        print('-'*20)
        print(f'Saving Results till -- {i}')
        fname = model_name.replace('/','-')
        input_df.to_pickle(f'/home/amey/depository/cs-eval/predictions_rebuttal/{fname}.pkl')
        input_df.to_csv(f'/home/amey/depository/cs-eval/predictions_rebuttal/{fname}.csv', index=False)
        print('-'*20)

print(f"Finished! Saving final results!")
fname = model_name.replace('/','-')
input_df.to_pickle(f'/home/amey/depository/cs-eval/predictions_rebuttal/{fname}.pkl')
input_df.to_csv(f'/home/amey/depository/cs-eval/predictions_rebuttal/{fname}.csv', index=False)