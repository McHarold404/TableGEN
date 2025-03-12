import json
import numpy as np
import pandas as pd
import argparse
import time
import os
from io import StringIO
from model_inference.gemini import *
from utils.table_utils import *
from eval.livesum_eval import *

# Argument parser to take model directory and model name as input
parser = argparse.ArgumentParser(description="Generate tables using Gemini model")
parser.add_argument("--model_dir", type=str, required=True, help="Path to the model directory")
parser.add_argument("--model_name", type=str, required=True, help="Name of the Gemini model")
args = parser.parse_args()

path = './data/LiveSum/test.json'
df = pd.read_json(path)
full_length = len(df)

print('Starting...')
for idx in range(full_length):
    key = 11 if idx % 2 == 1 else 12
    superset = paragraph_to_numbered_sentences(paragraph=df['text'][idx])
    sz = len(superset)
    set1 = superset[:int(sz / 2)]
    set2 = superset[int(sz / 2):]
    formatted_output_1 = "\n".join(set1)
    formatted_output_2 = "\n".join(set2)
    
    # Ensure directories exist
    header_dir = f"model_outputs/Livesum/{args.model_dir}/Headers"
    step1_dir = f"model_outputs/Livesum/{args.model_dir}/TabGen_2step_from_OG/step1"
    step2_dir = f"model_outputs/Livesum/{args.model_dir}/TabGen_2step_from_OG/step2"
    os.makedirs(step1_dir, exist_ok=True)
    os.makedirs(step2_dir, exist_ok=True)
    
    header_path = f"{header_dir}/{idx}.txt"
    with open(header_path, "r") as f:
        header_out = f.read()
    
    input_text = f"Table Schema:\n{header_out}\nStatements:\n{formatted_output_1}"
    output_table = ask_gemini(
        text=input_text,
        prompt_path="prompts/Livesum/Current/fill_table_gemini.txt",
        model_name=args.model_name,
        key=key
    )
    
    step1_output_path = f"{step1_dir}/{idx}.txt"
    with open(step1_output_path, 'w') as f:
        f.write(output_table)
    
    try:
        dfa = extract_table(output_table)
    except:
        print("First step bugged out, skipping this index")
        continue
    
    table_input = [dfa.columns.tolist()] + dfa.values.tolist()
    input_str_pipe = "\n".join(["|".join(map(str, row)) for row in table_input])
    
    input_text = f"Given Table:\n{input_str_pipe}\nStatements:\n{formatted_output_2}"
    final_output = ask_gemini(
        text=input_text,
        prompt_path="prompts/Livesum/Current/fill_table_2_gemini.txt",
        key=key
    )
    
    step2_output_path = f"{step2_dir}/{idx}.txt"
    with open(step2_output_path, 'w') as f:
        f.write(final_output)
    
    print(f'Saved results for idx {idx}')
    print("*****************************")
    time.sleep(5)
