import json
import numpy as np
import pandas as pd
import argparse
import time
import os
from io import StringIO
from model_inference.gpt import *
from utils.table_utils import *
from eval.livesum_eval import *

# Argument parser to take model directory and model name as input
parser = argparse.ArgumentParser(description="Generate tables using GPT model")
parser.add_argument("--model_dir", type=str, required=True, help="Path to the model directory")
parser.add_argument("--model_name", type=str, required=True, help="Name of the GPT model")
args = parser.parse_args()

path = './data/LiveSum/test.json'
df = pd.read_json(path)
full_length = len(df)

print('Starting...')
for idx in range(full_length):
    formatted_output = "\n".join(paragraph_to_numbered_sentences(paragraph=df['text'][idx]))
    time.sleep(4)
    
    header_path = f"model_outputs/Livesum/{args.model_dir}/Headers/{idx}.txt"
    with open(header_path, "r") as f:
        header_out = f.read()
    
    input_text = f"Table Schema:\n{header_out}\nStatements:\n{formatted_output}"
    output_table = ask_chatgpt(
        text=input_text,
        prompt_path="prompts/Livesum/Our_Method/fill_table_gemini.txt",
        model_name=args.model_name,
    )
    
    output_dir = f"model_outputs/Livesum/{args.model_dir}/TabGen_1step_from_OG"
    os.makedirs(output_dir, exist_ok=True)
    
    output_path = f"{output_dir}/{idx}.txt"
    with open(output_path, 'w') as f:
        f.write(output_table)
    
    print(f'Saved results for idx {idx}')
    print("*****************************")
