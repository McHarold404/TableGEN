import argparse
import os
import pandas as pd
import numpy as np
from io import StringIO
from model_inference.gemini import ask_gemini
from utils.utils import read_file, write_to_file 
from eval.livesum_eval import *
import time

def get_table_string(df):
    if df.empty:
        return ''
    df = df.where(pd.notnull(df), None)
    header = '|'.join(df.columns.astype(str))
    data_rows = []
    for r in df.to_numpy():
        temp = '|'.join(str(None) if item is None else str(item) for item in r)
        data_rows.append(temp)
    table_string = '\n'.join([header] + data_rows)
    return table_string

def get_table_dataframe(df):
    df = df.replace('<NEWLINE>', '\n')
    table_string_io = StringIO(df)
    df_table = pd.read_csv(table_string_io)
    return df_table

def unroll_tables(model_dir, exp_name, full_length=100):
    input_dir = f"model_outputs/Livesum/{model_dir}/{exp_name}"
    output_dir = f"model_outputs/Livesum/{model_dir}/{exp_name}/Unrolled_Statements"
    
    os.makedirs(output_dir, exist_ok=True)

    print("Starting table unrolling...")

    for idx in range(full_length):
        input_file = os.path.join(input_dir, f"{idx}.txt")

        if not os.path.exists(input_file):
            print(f"{input_file} not found. Skipping...")
            continue

        try:
            gold_table = read_file(input_file)
        except Exception as e:
            print(f"Error reading {input_file}: {e}")
            continue

        try:
            df = extract_table(gold_table)
            ## except red cards whatever columns have both rows as zero, remove them as they were added during realignment.
            # Identify columns to drop (all-zero columns, except 'Red Cards')
            print(df.columns,df.shape)
            columns_to_drop = [
                col for col in df.columns
                if col!= "Team" and col != "Red Cards" and df[col].astype(float).sum() == 0  # Convert to float and sum
            ]
            # Drop the identified columns
            df.drop(columns=columns_to_drop, inplace=True)
            print(df.shape)
            table_string = get_table_string(df)
            time.sleep(10)
            unrolled_statements = ask_gemini(text=table_string, prompt_path='eval/adobe_eval/tabunroll_adobe_eval.txt',key = 7,model_name = "gemini-2.0-flash-exp")

            output_file = os.path.join(output_dir, f"{idx}.txt")
            write_to_file(content=unrolled_statements, file_path=output_file)

            print(f"Table {idx} unrolled and saved.")
        except Exception as e:
            print(f"Error processing table {idx}: {e}")

    print("All tables processed.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Script to process tables and unroll statements")
    parser.add_argument("--model_dir", type=str, required=True, help="Name of the model directory")
    parser.add_argument("--exp_name", type=str, required=True, help="Name of the experiment")

    args = parser.parse_args()
    unroll_tables(args.model_dir, args.exp_name)
