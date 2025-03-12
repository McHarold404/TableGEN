import json
import numpy as np
import pandas as pd
import argparse
import os
from io import StringIO
import textwrap
import time
from model_inference.gpt import *
from utils.table_utils import *
from eval.livesum_eval import *

def process_tables(model_dir, model_name, full_length=754):
    path = './data/LiveSum/test.json'
    df = pd.read_json(path)

    print('Starting...')

    for idx in range(580,full_length):
        try:
            with open(f"model_outputs/Livesum/{model_dir}/Atomic/{idx}.txt", 'r') as f:
                sentences = f.readlines()
        except FileNotFoundError:
            print(f"File model_outputs/Livesum/{model_dir}/Atomic/{idx}.txt not found. Skipping...")
            continue

        formatted_output = "\n".join(f"{i}:{line.strip()}" for i, line in enumerate(sentences))

        try:
            with open(f"model_outputs/Livesum/{model_dir}/Headers/{idx}.txt", "r") as f:
                header_out = f.read()
        except FileNotFoundError:
            print(f"File model_outputs/Livesum/{model_dir}/Headers/{idx}.txt not found. Skipping...")
            continue

        input_text = f"Table Schema: \n{header_out}\nStatements: \n{formatted_output}"

        output_table = ask_chatgpt(text=input_text, prompt_path="prompts/Livesum/Current/fill_table_gemini.txt", model_name=model_name)

        os.makedirs(f"model_outputs/Livesum/{model_dir}/TabGen_1step_from_Atomic/", exist_ok=True)

        with open(f"model_outputs/Livesum/{model_dir}/TabGen_1step_from_Atomic/{idx}.txt", 'w') as f:
            f.write(output_table)

        print(f"Saved results for idx {idx}")
        print("*****************************")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Script to generate tables from statements using GPT models.")
    parser.add_argument("--model_dir", type=str, required=True, help="Directory name of the model outputs.")
    parser.add_argument("--model_name", type=str, required=True, help="Name of the model being used.")

    args = parser.parse_args()

    process_tables(args.model_dir, args.model_name)
