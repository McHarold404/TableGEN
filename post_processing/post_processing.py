from utils.utils import *
from utils.table_utils import *
import json 
from tqdm import tqdm
import re
import numpy as np
import pandas as pd
import time
import pprint
import os

def process_rotowire_output(output,gold_label_path = "data/rotowire/test.data",eval_rows= None):
    ## unprocessed output of llms (json dictionary consisting of data point number, input and response)

    print("Starting table extraction...")
    extracted_tables = extract_tables_from_responses(output)
    print("Table extraction complete.")


    # if reading and extractind directly from gpt model.  
    # if model responses are saved and read from a file, helpful for testing and multiple evaluations.
    output_tables = generate_output_lines(output_path = gold_label_path,end_line= eval_rows) ### gold tables
    #print(extracted_tables)

    # Now you can process each extracted table

    gold_table_player = []
    gold_table_team = []
    model_output_team = []
    model_output_player = []

    for model_output,gold_label in zip(extracted_tables,output_tables):
        #data_point = tables['data_point']

        gold_tables = separate_tables(gold_label)
        try: 
            model_output_tt = prune_table(model_output['Team'])
        except:
            model_output_tt = ""
        try:
            model_output_pt = prune_table(model_output['Player'])
        except:
            model_output_pt = ""
        model_output_team.append(model_output_tt)
        model_output_player.append(model_output_pt) 
        gold_table_player.append( gold_tables['Player'])
        gold_table_team.append(gold_tables['Team'])
    return model_output_team,model_output_player,gold_table_team,gold_table_player


def process_rotowire_corrected_output(output_path=None, gold_label_path="data/rotowire_corrected", eval_rows=None):
    """
    Optimized function for processing rotowire corrected outputs, appending headers in memory without modifying files,
    and ensuring gold_table_team and gold_table_player contain NumPy arrays.
    """

    import os
    import numpy as np
    from tqdm import tqdm
    from io import StringIO

    print("Starting table extraction...")
    extracted_tables = extract_tables_from_responses_path(output_path)
    print("Table extraction complete.")

    gold_table_player = []
    gold_table_team = []
    model_output_team = []
    model_output_player = []

    for idx, model_output in enumerate(tqdm(extracted_tables, desc="Processing Samples")):
        # Define file paths
        print("Line:",idx)
        team_path = f"{gold_label_path}/team/{idx}.csv"
        player_path = f"{gold_label_path}/player/{idx}.csv"

        # Check if files exist
        if not os.path.exists(team_path) or not os.path.exists(player_path):
            print(f"Error: Missing table files for index {idx}")
            gold_table_team.append(np.array([]))  # Append empty array for missing files
            gold_table_player.append(np.array([]))  # Append empty array for missing files
            continue

        # Handle empty or non-empty team file in memory
        if os.stat(team_path).st_size == 0:
            print(f"Warning: Empty team file for index {idx}")
            gold_team = np.array([])  # Create an empty array
        else:
            # Read file content and append "Team" header in memory
            with open(team_path, "r") as f:
                lines = f.readlines()

            if lines[0].strip() == "":
                lines[0] = "Team" + lines[0]  # Add header in memory

            # Convert the updated content to a NumPy array
            gold_team = np.genfromtxt(StringIO("".join(lines)), delimiter=",", dtype=None, encoding="utf-8", filling_values="")

        # Handle empty or non-empty player file in memory
        if os.stat(player_path).st_size == 0:
            print(f"Warning: Empty player file for index {idx}")
            gold_player = np.array([])  # Create an empty array
        else:
            # Read file content and append "Player" header in memory
            with open(player_path, "r") as f:
                lines = f.readlines()

            if lines[0].strip() == "":
                lines[0] = "Player" + lines[0]  # Add header in memory

            # Convert the updated content to a NumPy array
            gold_player = np.genfromtxt(StringIO("".join(lines)), delimiter=",", dtype=None, encoding="utf-8", filling_values="")

        # Process model outputs with pruning
        model_output_tt = prune_table(model_output.get("Team", ""))
        model_output_pt = prune_table(model_output.get("Player", ""))

        # Append results to lists
        model_output_team.append(model_output_tt)
        model_output_player.append(model_output_pt)
        gold_table_team.append(gold_team)
        gold_table_player.append(gold_player)

    return model_output_team, model_output_player, gold_table_team, gold_table_player




def prune_table(table_str: str) -> str:
    # Split the table into rows
    if table_str == '':
        return ""
    rows = table_str.strip().split('\n')
    # Extract headers and data rows
    headers = rows[0].split('|')
    data_rows = [row.split('|') for row in rows[1:]]
    
    # Ensure all data rows have the same length as headers
    for i in range(len(data_rows)):
        if len(data_rows[i]) < len(headers):
            # Add trailing empty strings to make row lengths equal
            data_rows[i].extend([''] * (len(headers) - len(data_rows[i])))
    
    # Identify non-empty columns (include column 0 for player names)
    non_empty_columns = [i for i in range(len(headers)) if i == 0 or any(row[i].strip() for row in data_rows)]
    
    # Prune headers and data rows by keeping only non-empty columns
    pruned_headers = [headers[i] for i in non_empty_columns]
    pruned_data_rows = [[row[i] for i in non_empty_columns] for row in data_rows]
    
    # Recreate the table with pruned columns
    pruned_table = '|'.join(pruned_headers) + '\n' + '\n'.join(['|'.join(row) for row in pruned_data_rows])
    return pruned_table
