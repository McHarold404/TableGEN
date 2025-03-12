import argparse
import os
import csv
import re
import pandas as pd
import numpy as np
from io import StringIO
from model_inference.gemini import *
from model_inference.gpt import *
from eval.livesum_eval import *

def process_tables(exp_name, model_dir):
    path = 'data/LiveSum/test.json'
    start = 214
    end = 754
    data = pd.read_json(path)
    outputs = []
    gold = []

    for idx in range(start, end):
        fname = f"model_outputs/Livesum/{model_dir}/{exp_name}/{idx}.txt"

        fname_2 = f"./model_outputs/Livesum/{model_dir}/{exp_name}/realigned/{idx}.txt"
        if os.path.exists(fname_2):
            continue
        if not os.path.exists(fname):
            print(fname, "not found")
            continue

        with open(fname, "r") as f:
            file_content = f.read()

        try:
            dfa = extract_table(file_content)
        except:
            print("Table extraction failed for file:", fname)
            continue

        table_input = [dfa.columns.tolist()] + dfa.values.tolist()
        input_str_pipe = "\n".join(["|".join(map(str, row)) for row in table_input])

        realigned_table = ask_gemini(text=input_str_pipe, prompt_path='prompts/Livesum/Our_Method/realign.txt',key = 1)
        time.sleep(6)
        realigned_fname = f"model_outputs/Livesum/{model_dir}/{exp_name}/realigned/{idx}.txt"

        os.makedirs(os.path.dirname(realigned_fname), exist_ok=True)
        with open(realigned_fname, "w") as f:
            f.write(realigned_table)

        try:
            dfa = extract_table(realigned_table)
        except:
            print('Table extraction failed after realignment. Idx:', idx)
            continue

        outputs.append(dfa)

        table_string = data['table'][idx].replace('<NEWLINE>', '\n')
        df_table = pd.read_csv(StringIO(table_string))
        gold.append(df_table)
        print(idx, dfa.shape)

    print('Done')

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp_name", type=str, required=True, help="Experiment name")
    parser.add_argument("--model_dir", type=str, required=True, help="Model directory")
    args = parser.parse_args()

    process_tables(args.exp_name, args.model_dir)
