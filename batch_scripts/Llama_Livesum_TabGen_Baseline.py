import json
import numpy as np
import pandas as pd
import time
import argparse
import os
from io import StringIO
from model_inference.llama import *
from utils.table_utils import *
from eval.livesum_eval import *


def main(model_dir, exp_name, prompt_path,key):
    path = './data/LiveSum/test.json'
    
    # Read the JSON file with error handling.
    try:
        df = pd.read_json(path)
    except Exception as e:
        print(f"Error reading JSON file '{path}': {e}")
        return

    full_length = len(df)
    print('Starting...')

    for idx in range(full_length):
        # Retrieve text for the current index.
        fname = f"./model_outputs/Livesum/{model_dir}/{exp_name}/realigned/{idx}.txt"
        if os.path.exists(fname):
            continue
        try:
            text = df['text'][idx]
        except Exception as e:
            print(f"Error retrieving text at index {idx}: {e}")
            continue

        # Call ask_llama and catch any errors.
        try:
            baseline_table = ask_llama(
                text=text, 
                prompt_path=f"prompts/Livesum/Current/{prompt_path}.txt", 
                key = key
            )
        except Exception as e:
            print(f"Error during ask_llama for index {idx}: {e}")
            continue
        
        # Ensure output directory exists.
        output_dir = f"./model_outputs/Livesum/{model_dir}/{exp_name}/"
        try:
            os.makedirs(output_dir, exist_ok=True)
        except Exception as e:
            print(f"Error creating directory '{output_dir}': {e}")
            continue
        
        # Write the output file with error handling.
        output_path = os.path.join(output_dir, f"{idx}.txt")
        try:
            with open(output_path, "w") as f:
                f.write(baseline_table)
        except Exception as e:
            print(f"Error writing to file '{output_path}': {e}")
            continue
        
        print(f'Saved results for idx {idx}')
        print("*****************************")
        


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run Gemini model inference and save results.")
    parser.add_argument("--model_dir", type=str, required=True, help="Directory to save model outputs")
    parser.add_argument("--exp_name", type=str, required=True, help="Experiment name")
    parser.add_argument("--prompt_path", type=str, required=True, help="Prompt file name (without extension)")
    parser.add_argument("--key", type=int, required=True, help="Prompt file name (without extension)")
    
    args = parser.parse_args()
    
    try:
        main(args.model_dir, args.exp_name, args.prompt_path,args.key)
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
