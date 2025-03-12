import json
import numpy as np
import pandas as pd
import argparse
import time
from io import StringIO
import textwrap
from model_inference.llama import *
from utils.table_utils import *

def process_atomic_generation(model_dir,key,full_length=754):
    path = './data/LiveSum/test.json'
    df = pd.read_json(path)

    print('Starting...')

    # Ensure the output directory exists
    output_path = f"./model_outputs/Livesum/{model_dir}/TabGen_Unified_Zero_Shot_COT/"
    os.makedirs(output_path, exist_ok=True)

    for idx in range(full_length):
        prompt_path = "prompts/Livesum/Current/tabgen_unified_zero_shot_COT.txt"

        # Call Gemini model
        atomic_out = ask_llama(text=df['text'][idx], prompt_path=prompt_path, key=key)

        # Save the output
        output_file = os.path.join(output_path, f"{idx}.txt")
        with open(output_file, "w") as f:
            f.write(atomic_out)

        time.sleep(5)  # Avoid excessive API calls
        print(f"Finished processing index {idx}")

    print("Saved all results.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Script for schema + atomic generation using Gemini models.")
    parser.add_argument("--model_dir", type=str, required=True, help="Directory name of the model outputs.")
    parser.add_argument("--key", type=int, required=True, help="Name of the Gemini model being used.")

    args = parser.parse_args()

    process_atomic_generation(args.model_dir, args.key)
