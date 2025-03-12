import os
import json
import numpy as np
import pandas as pd
from io import StringIO
import textwrap
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from model_inference.llama import ask_llama
from utils.table_utils import *

# Path to the input JSON file
path = './data/LiveSum/test.json'
try:
    df = pd.read_json(path)
except Exception as e:
    print(f"Error reading JSON file '{path}': {e}")
    exit(1)

full_length = 754
model_dir = "Llama3.3"

# Define the output directory and ensure it exists
output_dir = f"./model_outputs/Livesum/{model_dir}/T3/Zero_Shot/Direct/"
try:
    os.makedirs(output_dir, exist_ok=True)
except Exception as e:
    print(f"Error creating output directory '{output_dir}': {e}")
    exit(1)

print('Starting parallel processing...')

def process_sample(idx):
    # Define the realigned file path to check if sample has been processed already
    fname = f"./model_outputs/Livesum/{model_dir}/T3/Zero_Shot/Direct/realigned/{idx}.txt"
    if os.path.exists(fname):
        print(f"Skipping {idx}, already processed.")
        return None

    # fname = f"./model_outputs/Livesum/{model_dir}/T3/Zero_Shot/Direct/{idx}.txt"
    # if os.path.exists(fname):
    #     print(f"Skipping {idx}, already processed.")
    #     return None

    # Retrieve text for the sample
    try:
        text = df['text'][idx]
    except Exception as e:
        print(f"Error retrieving text for sample {idx}: {e}")
        return None

    # Call the ask_llama API with the prompt
    try:
        atomic_out = ask_llama(
            text=text, 
            prompt_path="prompts/Livesum/Current/T3_one_shot_direct.txt",
            key = 2
        )
    except Exception as e:
        print(f"Error during ask_llama for sample {idx}: {e}")
        return None

    # Save the output to file
    output_file = os.path.join(output_dir, f"{idx}.txt")
    try:
        with open(output_file, "w") as f:
            f.write(atomic_out)
    except Exception as e:
        print(f"Error writing output for sample {idx} to file '{output_file}': {e}")
        return None

    print(f"Processed sample {idx}")
    return idx

# Use ThreadPoolExecutor to process samples concurrently
num_threads = 200
with ThreadPoolExecutor(max_workers=num_threads) as executor:
    futures = {executor.submit(process_sample, idx): idx for idx in range(full_length)}
    for future in as_completed(futures):
        try:
            result = future.result()  # result is either the idx or None
        except Exception as e:
            print(f"Error processing sample {futures[future]}: {e}")

print("Processed all samples")
