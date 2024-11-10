from utils.utils import *
import json 
import os
from utils.table_utils import *

def process_rotowire_output(output,gold_label_path = "data/rotowire/test.data",eval_rows= None):
    ## unprocessed output of llms (json dictionary consisting of data point number, input and response)
    ## gold_label_path = path of gold labels of the test data
    extracted_tables = extract_tables_from_responses(output) 
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
        #model_tables = separate_tables(model_output.strip(['Final Answer','\n',':')])
        #print(model_tables)
        # print(model_tables['Player'])
        # print("--------------------------")
        # print(model_tables['Team'])
        #print("---------------------------")
        #print(gold_tables['Player'])
        #print("---------------------------")
        #print(gold_tables['Team'])
        #print("---------------------------")
    
        model_output_team.append(model_output_tt)
        model_output_player.append(model_output_pt) 
        gold_table_player.append( gold_tables['Player'])
        gold_table_team.append(gold_tables['Team'])
    return model_output_team,model_output_player,gold_table_team,gold_table_player

import re

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

