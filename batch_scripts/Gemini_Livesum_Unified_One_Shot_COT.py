import json
import numpy as np
import pandas as pd
import argparse
import time
from io import StringIO
import textwrap
from model_inference.gemini import *
from utils.table_utils import *

def process_atomic_generation(model_dir, model_name, full_length=100):
    path = './data/LiveSum/test.json'
    df = pd.read_json(path)

    print('Starting...')

    # Ensure the output directory exists
    output_path = f"./model_outputs/Livesum/{model_dir}/TabGen_Unified_One_Shot_COT_Weak/"
    os.makedirs(output_path, exist_ok=True)

    for idx in range(full_length):
        key = 1
        prompt_path = "prompts/Livesum/Current/tabgen_unified_one_shot_COT.txt"

        # Call Gemini model
        atomic_out = ask_gemini(text=df['text'][idx], prompt_path=prompt_path, key=key, model_name=model_name)

        # Save the output
        output_file = os.path.join(output_path, f"{idx}.txt")
        with open(output_file, "w") as f:
            f.write(atomic_out)

        if idx %10 == 0:
            time.sleep(10)
        print(f"Finished processing index {idx}")

    print("Saved all results.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Script for schema + atomic generation using Gemini models.")
    parser.add_argument("--model_dir", type=str, required=True, help="Directory name of the model outputs.")
    parser.add_argument("--model_name", type=str, required=True, help="Name of the Gemini model being used.")

    args = parser.parse_args()

    process_atomic_generation(args.model_dir, args.model_name)
