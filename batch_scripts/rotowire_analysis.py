import sys
import os
from dotenv import *
# Add the project root directory to sys.path
from post_processing.post_processing import *
from utils.utils import *
from utils.table_utils import *
#from eval.eval import *
from model_inference.gpt import *
import json
from model_inference.gemini import *
#from model_inference.llama import *
from utils.rotowire_utils import *
import pprint

def parse_table(text):
    """
    Parses markdown-style tables from text by scanning line-by-line.

    A table is expected to be preceded by a header line starting with "###"
    (followed by the table name) and then consecutive lines starting with "|" that
    constitute the table. Empty cells (i.e. cells that are empty after stripping) are
    replaced with None.

    It also skips separator rows (e.g. those containing mostly dashes).

    Returns a dictionary mapping table names to pandas DataFrames.
    """
    tables = {}
    lines = text.splitlines()
    i = 0
    while i < len(lines):
        line = lines[i].strip()
        # Look for a table header line (e.g., "### Team" or "### Player")
        header_match = re.match(r"^###\s+(.+)", line)
        if header_match:
            table_name = header_match.group(1).strip()
            table_lines = []
            i += 1
            # Skip any blank lines immediately following the table header.
            while i < len(lines) and not lines[i].strip():
                i += 1
            # Collect lines that start with a pipe
            while i < len(lines) and lines[i].strip().startswith("|"):
                table_lines.append(lines[i])
                i += 1

            if table_lines:
                # Remove any separator rows (rows that are made mostly of dashes, colons, or pipes)
                content_lines = [ln for ln in table_lines if not re.match(r"^\s*\|[\s\-\|:]+\|\s*$", ln)]
                if not content_lines:
                    continue

                # Process the header row: empty cells become None.
                header_line = content_lines[0]
                headers = [
                    cell.strip() if cell.strip() != "" else None
                    for cell in header_line.strip().strip('|').split('|')
                ]
                
                data = []
                # Process the remaining lines as data rows.
                for row_line in content_lines[1:]:
                    row_cells = [
                        cell.strip() if cell.strip() != "" else None
                        for cell in row_line.strip().strip('|').split('|')
                    ]
                    # Ensure that each row has the same number of cells as headers.
                    if len(row_cells) < len(headers):
                        row_cells.extend([None] * (len(headers) - len(row_cells)))
                    elif len(row_cells) > len(headers):
                        row_cells = row_cells[:len(headers)]
                    data.append(row_cells)
                
                if data:
                    df = pd.DataFrame(data, columns=headers, dtype=object)
                    df.replace('None', np.nan, inplace=True)
                    df = df[~df.iloc[:, 1:].isna().all(axis=1)]
                    df.dropna(axis=1, how='all', inplace=True)
                    df.replace("None", np.nan, inplace=True)
                    tables[table_name] = df
        else:
            i += 1

    return tables

def get_table_formatted(table):
    if table is None:
        return ''
    elif isinstance(table, pd.DataFrame) and table.empty:
        return ''
    
    columns = table.columns.tolist()
    #columns[0] = ''
    table.columns = columns

    table.replace('False', False, inplace=True)
    table.replace('True', True, inplace=True)

    return table.to_markdown(index=False)


import re
import pandas as pd
import random
import numpy as np

full_length = 728
idx = 0
fname_text = "data/rotowire/test.text"
gold_table_player = pd.read_csv(f"data/rotowire_corrected_full/player/{idx}.csv")
gold_table_team = pd.read_csv(f"data/rotowire_corrected/team/{idx}.csv")
table_type_fname = 'player' ## or player
# table_type_list = ['Player','Team'] ## or Player
# exp_names = ["Baseline_1shot_COT","Method_1shot_COT"]
table_type_list = ['Team'] ## or Player
exp_names = ["Method_1shot_COT"]



for table_type in table_type_list:
    table_type_fname = 'player' if table_type == 'Player' else 'team'
    for exp_name in exp_names:
        with open(fname_text, "r") as f:
            all_text = f.readlines()
        key = 1
        for idx in range(0,full_length):
            gold_text = all_text[idx]
            try:
                gold_table = pd.read_csv(f"data/rotowire_corrected_full/{table_type_fname}/{idx}.csv")
            except:
                print(f"Error: gold {table_type} not found for sample {idx}")
                # If either CSV fails to load, skip this iteration.
                continue
            
            gold_table.rename(columns = {'Unnamed: 0': table_type},inplace = True)    
            # Fix: Use .values.tolist() for the row headers, and .columns.tolist() for column headers
            row_headers = gold_table[table_type].values.tolist()
            column_headers = gold_table.columns.tolist()

            target_schema = (
                f"[ \n row_headers : {row_headers} \n "
                f"columns_headers : {column_headers} ]"
            )

            # Read the file containing the markdown table
            try:
                with open(f"model_outputs/Rotowire/Gemini2.0/{exp_name}/{idx}.txt", "r") as f:
                    tables = f.read()
                input_text = parse_table(tables)[table_type]    
                input_table = get_table_formatted(input_text)        
            except:
                print(f"Error: {table_type}, {exp_name},{idx} not found or didn't extract table")
                continue
            final_text = f"Given table:\n{input_table}\n\nTarget Schema:\n{target_schema}"
            response  = ask_gemini(text = final_text, prompt_path="prompts/Rotowire/Current/pp_rotowire_player.txt", key = key)
            time.sleep(5)
            print("Finished idx: ", idx)
            with open(f"model_outputs/Rotowire/Gemini2.0/Analysis/{exp_name}/{table_type}/{idx}.txt","w") as f:
                f.write(response)
            continue
        print(f"{table_type} done for {exp_name}")
    