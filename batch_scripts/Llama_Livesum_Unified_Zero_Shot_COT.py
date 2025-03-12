import os
import json
import numpy as np
import pandas as pd
import argparse
import time
from io import StringIO
import textwrap
from concurrent.futures import ThreadPoolExecutor, as_completed
from model_inference.llama import ask_llama
from utils.table_utils import *

def process_atomic_generation(model_dir, key, full_length=754):
    path = './data/LiveSum/test.json'
    try:
        df = pd.read_json(path)
    except Exception as e:
        print(f"Error reading JSON file '{path}': {e}")
        return

    print('Starting...')

    # Ensure the output directory exists
    output_path = f"./model_outputs/Livesum/{model_dir}/TabGen_Unified_Zero_Shot_COT/"
    try:
        os.makedirs(output_path, exist_ok=True)
    except Exception as e:
        print(f"Error creating output directory '{output_path}': {e}")
        return

    # Set the prompt path (constant for all samples)
    prompt_path = "prompts/Livesum/Current/tabgen_unified_zero_shot_COT.txt"

    # Function to process each sample
    def process_sample(idx):
        output_file_check = f"{output_path}/realigned/{idx}.txt"
        if os.path.exists(output_file_check):
            print(f"Skipping {idx}, already processed.")
            return 
        try:
            atomic_out = ask_llama(text=df['text'][idx], prompt_path=prompt_path, key=key)
        except Exception as e:
            print(f"Error during ask_llama for index {idx}: {e}")
            return
        output_file = os.path.join(output_path, f"{idx}.txt")
        try:
            with open(output_file, "w") as f:
                f.write(atomic_out)
        except Exception as e:
            print(f"Error writing output for index {idx} to file '{output_file}': {e}")
            return
        print(f"Finished processing index {idx}")

    # Process all samples concurrently using 200 threads
    with ThreadPoolExecutor(max_workers=200) as executor:
        futures = {executor.submit(process_sample, idx): idx for idx in range(full_length)}
        for future in as_completed(futures):
            idx = futures[future]
            try:
                future.result()
            except Exception as e:
                print(f"Error processing index {idx}: {e}")

    print("Saved all results.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Script for schema + atomic generation using Gemini models.")
    parser.add_argument("--model_dir", type=str, required=True, help="Directory name of the model outputs.")
    parser.add_argument("--key", type=int, required=True, help="API key number for the model.")

    args = parser.parse_args()
    process_atomic_generation(args.model_dir, args.key)
