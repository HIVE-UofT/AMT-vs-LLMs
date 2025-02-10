!pip install openai==0.28

import os
import pandas as pd
import openai
import time
from google.colab import userdata

openai.api_key = userdata.get('API_KEY')

def mapping(label):
    label = label.strip().upper()
    if label == 'YES':
        return 1
    elif label == 'NO':
        return -1
    else:
        return 0

def labelling_pipeline(instruction_prompt, tweets):
    response = openai.ChatCompletion.create(
        model="gpt-4-turbo-preview",
        messages=[{"role": "system", "content": instruction_prompt},
                  {"role": "user", "content": tweets}]
    )
    labels_str = response.choices[0].message.content
    return labels_str, response.usage  

folder_path = 'drive/MyDrive/data/tweets/sedentary'
prompt_path = 'drive/MyDrive/data/prompts/sedentary.txt'
res_path = 'drive/MyDrive/data/results/sedentary'
log_path = 'drive/MyDrive/data/usage log/'

with open(prompt_path, 'r') as f:
    prompt = f.read()

dfs = []
costs = []

for filename in os.listdir(folder_path):
    if filename.endswith('.txt'):
        file_path = os.path.join(folder_path, filename)

        with open(file_path, 'r') as f:
            file_content = f.read()

        original_tweet_ids = set([line.split(':')[0].strip()[1:] for line in file_content.split("\n") if ":" in line])

        max_retries = 5
        retry_count = 0
        valid_response = False

        while retry_count < max_retries and not valid_response:
            labels_str, usage_stats = labelling_pipeline(prompt, file_content)
            result = labels_str.strip().split("\n")

            try:
                df = pd.DataFrame(columns=['QID', 'label'])
                processed_tweet_ids = set()

                for i, line in enumerate(result):
                    try:
                        Id, label = line.split(':', 1)
                        Id = Id.strip()[1:]
                        df.loc[i] = [Id, mapping(label)]
                        processed_tweet_ids.add(Id)
                    except ValueError:
                        raise Exception(f"Malformed line: {line}")

                if processed_tweet_ids == original_tweet_ids:
                    valid_response = True
                else:
                    raise Exception("Mismatch in tweet IDs. Retrying...")

            except Exception as e:
                retry_count += 1
                print(f"Error processing response (Attempt {retry_count}/{max_retries}): {e}")
                time.sleep(2) 

        if not valid_response:
            print(f"Skipping file {filename} after {max_retries} failed attempts.")
            continue

        with open(os.path.join(res_path, filename), 'w') as f:
            f.write(labels_str)
        dfs.append(df)

        # Store cost usage
        costs.append({
            "file": filename,
            "prompt_tokens": usage_stats["prompt_tokens"],
            "completion_tokens": usage_stats["completion_tokens"],
            "total_tokens": usage_stats["total_tokens"]
        })

result_df = pd.concat(dfs, ignore_index=True)
result_df.to_csv('drive/MyDrive/data/sedentary_res.csv', index=False)

cost_df = pd.DataFrame(costs)
cost_df.to_csv(os.path.join(log_path, 'sedentary_cost.csv'), index=False)

print("Processing complete. Results and cost logs saved.")
