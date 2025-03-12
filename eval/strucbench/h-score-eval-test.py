import Levenshtein
import difflib
import pandas as pd
import os
import json
import numpy as np

def parse_table(input_text):
    sections = input_text.strip().split('###')
    dataframes = {}
    for section in sections:
        if section.strip():
            lines = section.strip().split('\n')
            table_name = lines[0].strip()
            headers = lines[1].strip('|').split('|')
            rows = [line.strip('|').split('|') for line in lines[2:]]

            headers = [''] + headers # remove this line if table[0][0] present
            rows = [row for row in rows if len(row) == len(headers)]
            
            df = pd.DataFrame(rows, columns=headers, dtype=object)

            df.replace('None', np.nan, inplace=True)

            df.dropna(axis=1, how='all', inplace=True)
            
            dataframes[table_name] = df

    return dataframes


def levenshtein_similarity(pred, gold):
    if not gold:
        return 0.0
    return 1 - Levenshtein.distance(pred, gold) / (2 * len(gold))

def difflib_similarity(pred, gold):
    sm = difflib.SequenceMatcher(None, pred, gold)
    return sm.ratio()

def flatten_table(table):
    return [x for row in table for x in row ]

def get_table_artifacts(table):
    if table is None:
        return {
            "num_rows": 0,
            "num_cols": 0,
            "column_names": [],
            "data_rows": []
        }
    elif isinstance(table, pd.DataFrame) and table.empty:
        return {
            "num_rows": 0,
            "num_cols": 0,
            "column_names": [],
            "data_rows": []
        }

    columns = table.columns.to_list()
    columns[0] = ''

    return {
        "num_rows": table.shape[0],
        "num_cols": table.shape[1],
        "column_names": columns,
        "data_rows": table.to_numpy().tolist()
    }


def score_artifacts(artifacts_1, artifacts_2):
    sub_scores = {}
    sub_scores["num_rows_match"] = (artifacts_1["num_rows"] == artifacts_2["num_rows"]) * 1.0
    sub_scores["num_cols_match"] = (artifacts_1["num_cols"] == artifacts_2["num_cols"]) * 1.0
    sub_scores["columns_levenshtein_score"] = levenshtein_similarity(
        artifacts_1["column_names"],
        artifacts_2["column_names"],
    )
    sub_scores["columns_difflib_score"] = difflib_similarity(
        artifacts_1["column_names"],
        artifacts_2["column_names"],
    )
    sub_scores["data_levenshtein_score"] = levenshtein_similarity(
        flatten_table(artifacts_1["data_rows"]),
        flatten_table(artifacts_2["data_rows"]),
    )
    sub_scores["data_difflib_score"] = difflib_similarity(
        flatten_table(artifacts_1["data_rows"]),
        flatten_table(artifacts_2["data_rows"]),
    )
    return sub_scores

def score_tables(gold_team, gold_player, method_team, method_player):

    team_table_artifacts_pred = get_table_artifacts(method_team)
    player_table_artifacts_pred = get_table_artifacts(method_player)
    team_table_artifacts_gold = get_table_artifacts(gold_team)
    player_table_artifacts_gold = get_table_artifacts(gold_player)

    team_artifacts = score_artifacts(team_table_artifacts_pred, team_table_artifacts_gold)
    player_artifacts = score_artifacts(player_table_artifacts_pred, player_table_artifacts_gold)

    return pd.DataFrame({
        "team": team_artifacts,
        "player": player_artifacts,
    }).mean(1).to_dict()

#strucbench score
content_scores = 0.0
format_scores = 0.0

with open('./outputs/gpt-4o/baseline_direct_0shot_cot.json', 'r') as f:
    method_output = json.load(f)
    
gold_player_path = '../data/rotowire_corrected_full/player/'
gold_team_path = '../data/rotowire_corrected_full/team/'

for res_key, res_val in method_output.items():
    # if int(res_key) != 1:
    #     continue
    # print('Idx: ', res_key)
    method_tables = parse_table(res_val)

    if os.path.getsize(os.path.join(gold_player_path, f'{res_key}.csv')) == 0:
        gold_player = pd.DataFrame()
    else:
        gold_player = pd.read_csv(gold_player_path + f'{int(res_key)}.csv', keep_default_na=False, na_values=[''],engine='python', dtype=object)
    
    if os.path.getsize(os.path.join(gold_team_path, f'{res_key}.csv')) == 0:
        gold_team = pd.DataFrame()
    else:
        gold_team = pd.read_csv(gold_team_path + f'{int(res_key)}.csv', keep_default_na=False, na_values=[''],engine='python', dtype=object)
    
    # print(get_table_artifacts(method_tables.get('Player')))
    # print(get_table_artifacts(gold_player))

    our_score = score_tables(gold_team, gold_player, method_tables.get('Team'), method_tables.get('Player'))
    content_scores = content_scores + our_score["data_levenshtein_score"] + our_score["data_difflib_score"]
    format_scores = format_scores + our_score["num_rows_match"] + our_score["num_cols_match"] + our_score["columns_levenshtein_score"] + our_score["columns_difflib_score"]

    # if int(res_key) == 500:
    #     break


content_scores = content_scores / len(method_output)
format_scores = format_scores / len(method_output)

print("---------------------")
print("Strucbench score")
print("Content_score: ", content_scores)
print("Structure_score: ", format_scores)
print("---------------------")