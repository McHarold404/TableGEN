import re
import pandas as pd
import numpy as np



import re

import re

def extract_operations(file_contents: str) -> str:
    """
    Extracts the Updates and Revised State from soccer game statements,
    considering only the region starting from "1." and ending two lines 
    after the last statement block.

    Args:
        file_contents (str): The complete text containing numbered statements.

    Returns:
        str: A formatted string containing only the Updates and Revised State
             for each statement within the defined region.
    """
    # Step 1: Find the start of the region from the first occurrence of "1."
    start_index = file_contents.find("1.")
    if start_index == -1:
        return ""  # If "1." is not found, return empty string
    region = file_contents[start_index:]
    
    # Step 2: Define regex pattern to capture each numbered block
    # This pattern will be applied on the trimmed region.
    pattern = re.compile(
        r"(\d+)\.\s*\n"                          # Capture the statement number (e.g., "1.")
        r"Statement:.*?\n"                       # Match the Statement line (non-capturing)
        r"Updates:\s*(.*?)\n"                    # Capture the Updates line
        r"Revised State:\s*(.*?)(?=\n\d+\.|$)",  # Capture the Revised State line
        re.DOTALL | re.MULTILINE                  # Flags to allow matching across lines
    )

    # Step 3: Find all matches within the region
    matches = list(pattern.finditer(region))
    if not matches:
        return ""
    
    # Determine the end of the last block:
    # Use the end of the last match from the regex, which should be two lines after the last "Revised State"
    last_match_end = matches[-1].end()
    
    # Slice the region to include all content up to two lines after the last match
    relevant_region = region[:last_match_end]
    
    # Apply findall on the relevant region to capture all blocks cleanly
    matches = pattern.findall(relevant_region)

    # Initialize a list to hold the formatted output lines
    output_lines = []

    # Iterate over each match and format the output
    for match in matches:
        number, updates, revised_state = match
        updates = updates.strip()
        revised_state = revised_state.strip()
        
        output_lines.append(f"{number}. update: {updates}")
        output_lines.append(f"revised state: {revised_state}")
        output_lines.append("")  # Separation line

    formatted_output = "\n".join(output_lines).strip()
    return formatted_output



def extract_table(file_contents: str) -> pd.DataFrame:
    """
    Extracts the final table from the given file contents and returns it as a pandas DataFrame.
    
    The function can handle tables where the first cell is "Team" or is empty.
    
    Parameters:
    - file_contents (str): The complete content of the file as a string.
    
    Returns:
    - pd.DataFrame: The extracted table as a DataFrame.
    
    Raises:
    - ValueError: If the "### Final Table" section is not found or if there are mismatches in the table structure.
    """
    # Locate the start of the final table section using regex
    start_match = re.search(r"###\s*Final\s*Table", file_contents, re.IGNORECASE)
    if not start_match:
        raise ValueError("No '### Final Table' heading found.")
        
    # Extract the substring starting from "### Final Table"
    final_table_start = start_match.end()
    text_after_final_table = file_contents[final_table_start:].strip()
    
    # Split the content into lines and filter lines containing '<NEWLINE>'
    lines = [line.strip() for line in text_after_final_table.splitlines() if any(c.isalnum() for c in line) and "|" in line]
    
    #
    if not lines:
        raise ValueError("No table lines found in the final table section.")
    
    # Extract and process the header line
    header_line = lines[0]
    # Strip leading and trailing '|' and split by '|'
    raw_columns = [col.strip() for col in header_line.strip('|').split('|') if col.strip() != '<NEWLINE>']
    
    # Determine if the first column is 'Team' or empty
    if raw_columns and raw_columns[0].lower() == 'team':
        columns = raw_columns
    elif raw_columns and raw_columns[0] == '':
        # Replace the first empty string with 'Team'
        raw_columns[0] = 'Team'
        columns = raw_columns
    else:
        raise ValueError(
            f"First column is neither 'Team' nor empty. Found: '{raw_columns[0] if raw_columns else 'None'}'. "
            f"Columns: {raw_columns}"
        )
    
    # Process data rows
    data = []
    for idx, line in enumerate(lines[1:], start=2):  # Start at 2 to account for header
        # Strip leading and trailing '|' and split by '|'
        parts = [part.strip() for part in line.strip('|').split('|') if part.strip() != '<NEWLINE>']
        
        # If the first element is empty, remove it (optional based on data)
        if parts and parts[0] == '':
            parts.pop(0)
        
        # Check if the number of parts matches the number of columns
        if len(parts) != len(columns):
            raise ValueError(
                f"Row length mismatch at line {idx}! Expected {len(columns)} columns, but got {len(parts)}. "
                f"Columns: {columns} \n Row: {parts}"
            )
        
        # Replace 'Not found' with 0
        processed_parts = [0 if part.lower() == 'not found' else part for part in parts]
        data.append(processed_parts)
    
    # Create the DataFrame
    df = pd.DataFrame(data, columns=columns)
    
    # Convert numeric columns (excluding 'Team') to appropriate types
    numeric_columns = [col for col in df.columns if col.lower() != 'team']
    df[numeric_columns] = df[numeric_columns].apply(pd.to_numeric, errors='coerce').fillna(0)
    
    return df

import pandas as pd

def extract_markdown_table(markdown: str) -> pd.DataFrame:
    """
    Extracts a table from a Markdown-formatted string and returns it as a pandas DataFrame.
    
    Parameters:
    - markdown (str): The Markdown string containing the table.
    
    Returns:
    - pd.DataFrame: The extracted table as a DataFrame.
    
    Raises:
    - ValueError: If no valid table is found in the input string.
    """
    # Use pandas to read the Markdown table
    try:
        # Pandas can read Markdown tables using read_csv with sep='|'
        from io import StringIO
        
        # Split the markdown into lines
        lines = markdown.strip().split('\n')
        
        # Remove empty lines
        lines = [line for line in lines if line.strip()]
        
        # Ensure there's at least a header and a separator
        if len(lines) < 2:
            raise ValueError("Markdown table must have at least a header and a separator line.")
        
        # Join the lines and read with pandas
        table_str = '\n'.join(lines)
        df = pd.read_csv(StringIO(table_str), sep='|', engine='python')
        
        # Clean column names by stripping whitespace
        df.columns = [col.strip() for col in df.columns]
        
        # Remove unnamed columns if present (from leading/trailing '|')
        df = df.loc[:, ~df.columns.str.contains('^Unnamed')]
        
        # Strip whitespace from string columns
        df = df.applymap(lambda x: x.strip() if isinstance(x, str) else x)
        
        # Convert numeric columns (excluding 'Team') to appropriate types
        if 'Team' in df.columns:
            numeric_columns = [col for col in df.columns if col.lower() != 'team']
        else:
            numeric_columns = [col for col in df.columns]
        
        df[numeric_columns] = df[numeric_columns].apply(pd.to_numeric, errors='coerce').fillna(0)
        
        return df
    except Exception as e:
        raise ValueError(f"Failed to extract table from Markdown: {e}")


# def compute_rmse_and_er_cell_by_cell(df_true: pd.DataFrame, df_pred: pd.DataFrame) -> (float, float):
#     """
#     Computes RMSE and Error Rate between two DataFrames for "Home Team" and "Away Team",
#     ensuring the same number of columns, renaming 'Shots Taken' to 'Shots' in df_pred,
#     aligning columns by name, and calculating the metrics.
    
#     Parameters:
#     - df_true (pd.DataFrame): The ground truth DataFrame.
#     - df_pred (pd.DataFrame): The predicted DataFrame.
    
#     Returns:
#     - Tuple[float, float]: RMSE and Error Rate (%).
    
#     Raises:
#     - ValueError: If column counts differ or required teams are missing.
#     """
#     # Set 'Team' as index and drop any duplicate 'Team' columns
#     if df_pred.shape != df_true.shape:
#         return 1000,1000
#     for df in [df_true, df_pred]:
#         if 'Team' in df.columns:
#             df.set_index('Team', inplace=True)
#         df.index.name = 'Team'
#         if 'Team' in df.columns:
#             df.drop(columns=['Team'], inplace=True)

#     # Rename 'Shots Taken' to 'Shots' in df_pred
#     if 'Shots Taken' in df_pred.columns:
#         df_pred.rename(columns={'Shots Taken': 'Shots'}, inplace=True)
#     if 'Corners' in df_pred.columns:
#         df_pred.rename(columns={'Corners': 'Corner Kicks'}, inplace=True)
#     if 'Freekicks' in df_pred.columns:
#         df_pred.rename(columns={'Freekicks': 'Free Kicks'}, inplace=True)
    
#     # Check if both DataFrames have the same number of columns
#     if df_true.shape[1] != df_pred.shape[1]:
#         raise ValueError(f"Column count mismatch: df_true has {df_true.shape[1]} columns, "
#                          f"df_pred has {df_pred.shape[1]} columns.")
    
#     # Ensure both DataFrames contain 'Home Team' and 'Away Team'
#     expected_teams = {"Home Team", "Away Team"}
#     if not expected_teams.issubset(df_true.index) or not expected_teams.issubset(df_pred.index):
#         missing_true = expected_teams - set(df_true.index)
#         missing_pred = expected_teams - set(df_pred.index)
#         raise ValueError(f"Missing teams - df_true: {missing_true}, df_pred: {missing_pred}")
    
#     # Reorder DataFrames to have teams in the same order
#     df_true = df_true.loc[["Home Team", "Away Team"]]
#     df_pred = df_pred.loc[["Home Team", "Away Team"]]
    
#     # Align columns by name
#     if not set(df_true.columns) == set(df_pred.columns):
#         print(df_true.columns)
#         print(df_pred.columns)
#         raise ValueError("Column names do not match between df_true and df_pred.")
    
#     df_pred = df_pred[df_true.columns]
    
#     # Ensure all columns are numeric
#     if not all(pd.api.types.is_numeric_dtype(df_true[col]) for col in df_true.columns) or \
#        not all(pd.api.types.is_numeric_dtype(df_pred[col]) for col in df_pred.columns):
#         raise ValueError("All columns must be numeric.")
    
#     # Compute RMSE
#     mse = ((df_true - df_pred) ** 2).values.mean()
#     rmse = np.sqrt(mse)
    
#     # Compute Error Rate
#     exact_matches = (df_true == df_pred).values
#     error_count = np.size(exact_matches) - np.count_nonzero(exact_matches)
#     total_cells = exact_matches.size
#     er = (error_count / total_cells) * 100
    
#     return rmse, er

import pandas as pd
import numpy as np

import pandas as pd
import numpy as np

def compute_rmse_and_er_cell_by_cell(df_true: pd.DataFrame, df_pred: pd.DataFrame) -> dict:
    """
    Computes RMSE and Error Rate between two DataFrames for "Home Team" and "Away Team",
    categorizing columns into 'easy', 'hard', 'medium', and 'average'.

    Parameters:
    - df_true (pd.DataFrame): The ground truth DataFrame.
    - df_pred (pd.DataFrame): The predicted DataFrame.

    Returns:
    - dict: Dictionary with keys 'easy', 'hard', 'medium', 'average', each containing [RMSE, Error Rate].
    
    Raises:
    - ValueError: If column counts differ or required teams are missing.
    """
    
    #df_true.drop(columns = ['Red Cards'],inplace=True)
    if 'Red Cards' not in df_pred.columns:
        rcards = [0,0]
        df_pred['Red Cards'] = rcards
    print(df_pred.shape,df_true.shape)
    # 1. Check if shapes match
    if df_pred.shape != df_true.shape:
        return {'easy': [0,0], 'hard': [0,0], 'medium': [0,0], 'average': [1000,1000]}
    
    # 2. Set 'Team' as index and drop duplicate 'Team' columns
    for df in [df_true, df_pred]:
        if 'Team' in df.columns:
            df.set_index('Team', inplace=True)
        df.index.name = 'Team'
        if 'Team' in df.columns:
            df.drop(columns=['Team'], inplace=True, errors='ignore')
    
    # 3. Rename columns in df_pred
    df_pred.rename(columns={
        'Shots Taken': 'Shots',
        'Corners': 'Corner Kicks',
        'Freekicks': 'Free Kicks'
    }, inplace=True)
    
    # 4. Verify column counts
    if df_true.shape[1] != df_pred.shape[1]:
        raise ValueError(f"Column count mismatch: df_true has {df_true.shape[1]} columns, "
                         f"df_pred has {df_pred.shape[1]} columns.")
    
    # 5. Ensure both DataFrames contain 'Home Team' and 'Away Team'
    expected_teams = {"Home Team", "Away Team"}
    missing_true = expected_teams - set(df_true.index)
    missing_pred = expected_teams - set(df_pred.index)
    if missing_true or missing_pred:
        raise ValueError(f"Missing teams - df_true: {missing_true}, df_pred: {missing_pred}")
    
    # 6. Reorder DataFrames
    df_true = df_true.loc[["Home Team", "Away Team"]]
    df_pred = df_pred.loc[["Home Team", "Away Team"]]
    
    # 7. Align columns by name
    if set(df_true.columns) != set(df_pred.columns):
        print("df_true columns:", df_true.columns)
        print("df_pred columns:", df_pred.columns)
        raise ValueError("Column names do not match between df_true and df_pred.")
    df_pred = df_pred[df_true.columns]
    
    # 8. Ensure all columns are numeric
    if not all(pd.api.types.is_numeric_dtype(df_true[col]) for col in df_true.columns) or \
       not all(pd.api.types.is_numeric_dtype(df_pred[col]) for col in df_pred.columns):
        raise ValueError("All columns must be numeric.")
    
    # 9. Define categories
    categories = {
        'easy': ['Goals', 'Red Cards'],
        'hard': ['Shots', 'Fouls'],
        'medium': ['Yellow Cards', 'Free Kicks', 'Corner Kicks', 'Offsides']
    }
    
    results = {}
    
    # 10. Compute metrics for each category
    for category, cols in categories.items():
        if not set(cols).issubset(df_true.columns):
            raise ValueError(f"Missing columns for category '{category}': {set(cols) - set(df_true.columns)}")
        
        df_true_cat = df_true[cols]
        df_pred_cat = df_pred[cols]
        
        # Compute RMSE
        mse = ((df_true_cat - df_pred_cat) ** 2).values.mean()
        rmse = np.sqrt(mse)
        
        # Compute Error Rate
        exact_matches = (df_true_cat == df_pred_cat).values
        error_count = (~exact_matches).sum()
        total_cells = exact_matches.size
        er = (error_count / total_cells) * 100
        
        results[category] = [rmse, er]
    
    # 11. Compute average RMSE and Error Rate
    mse_avg = ((df_true - df_pred) ** 2).values.mean()
    rmse_avg = np.sqrt(mse_avg)
    
    exact_matches_avg = (df_true == df_pred).values
    error_count_avg = (~exact_matches_avg).sum()
    total_cells_avg = exact_matches_avg.size
    er_avg = (error_count_avg / total_cells_avg) * 100
    
    results['average'] = [rmse_avg, er_avg]
    
    return results


def compute_rmse_and_er_w_missing_cols(df_true: pd.DataFrame, df_pred: pd.DataFrame) -> dict:
    """
    Computes RMSE and Error Rate between two DataFrames for "Home Team" and "Away Team",
    categorizing columns into 'easy', 'hard', 'medium', and 'average'.

    Parameters:
    - df_true (pd.DataFrame): The ground truth DataFrame.
    - df_pred (pd.DataFrame): The predicted DataFrame.

    Returns:
    - dict: Dictionary with keys 'easy', 'hard', 'medium', 'average', each containing [RMSE, Error Rate].
    
    Raises:
    - ValueError: If column counts differ or required teams are missing.
    """
    # 1. Check if shapes match

    # if df_pred.shape != df_true.shape:
    #     return {'easy': [0,0], 'hard': [0,0], 'medium': [0,0], 'average': [1000,1000], "missing_cols": missing_cols}
    
    # 2. Set 'Team' as index and drop duplicate 'Team' columns
    for df in [df_true, df_pred]:
        if 'Team' in df.columns:
            df.set_index('Team', inplace=True)
        df.index.name = 'Team'
        if 'Team' in df.columns:
            df.drop(columns=['Team'], inplace=True, errors='ignore')
    
    if 'Red Cards' not in df_pred.columns:
        red_cards = [0,0]
        df_pred['Red Cards'] = red_cards
    
    # # 3. Rename columns in df_pred
    # df_pred.rename(columns={
    #     'Shots Taken': 'Shots',
    #     'Corners': 'Corner Kicks',
    #     'Freekicks': 'Free Kicks'
    # }, inplace=True)
    for col in list(df_pred.columns):
        if col not in df_true.columns:
            df_pred.drop(columns = col,inplace=True)
    assert(set(df_true.columns) == set(df_pred.columns))
    missing_cols = 0
    for col in list(df_pred.columns):
        if col != 'Red Cards':
            if df_pred[col].sum() == 0:
                df_pred.drop(columns=col,inplace=True)
                df_true.drop(columns = col,inplace=True)
                missing_cols +=1

    #print(df_pred.shape,df_true.shape)
    # 4. Verify column counts
    if df_true.shape[1] != df_pred.shape[1]:
        raise ValueError(f"Column count mismatch: df_true has {df_true.shape[1]} columns, "
                         f"df_pred has {df_pred.shape[1]} columns.")
    
    # 5. Ensure both DataFrames contain 'Home Team' and 'Away Team'
    expected_teams = {"Home Team", "Away Team"}
    missing_true = expected_teams - set(df_true.index)
    missing_pred = expected_teams - set(df_pred.index)
    if missing_true or missing_pred:
        raise ValueError(f"Missing teams - df_true: {missing_true}, df_pred: {missing_pred}")
    
    # 6. Reorder DataFrames
    df_true = df_true.loc[["Home Team", "Away Team"]]
    df_pred = df_pred.loc[["Home Team", "Away Team"]]
    
    # 7. Align columns by name
    if set(df_true.columns) != set(df_pred.columns):
        print("df_true columns:", df_true.columns)
        print("df_pred columns:", df_pred.columns)
        raise ValueError("Column names do not match between df_true and df_pred.")
    df_pred = df_pred[df_true.columns]
    
    # 8. Ensure all columns are numeric
    if not all(pd.api.types.is_numeric_dtype(df_true[col]) for col in df_true.columns) or \
       not all(pd.api.types.is_numeric_dtype(df_pred[col]) for col in df_pred.columns):
        raise ValueError("All columns must be numeric.")
    
    # 9. Define categories
    categories = {
        'easy': ['Goals', 'Red Cards'],
        'hard': ['Shots', 'Fouls'],
        'medium': ['Yellow Cards', 'Free Kicks', 'Corner Kicks', 'Offsides']
    }
    
    results = {"missing columns": missing_cols}
    # 10. Compute metrics for each category
    for category, cols in categories.items():
        # if not set(df_true.columns).issubset(cols):
        #     raise ValueError(f"Missing columns for category '{category}': {set(cols) - set(df_true.columns)}")
        available_cols = list(set(cols).intersection(set(df_true.columns)))
        if not available_cols:
            continue
        df_true_cat = df_true[available_cols]
        df_pred_cat = df_pred[available_cols]
        # Compute RMSE
        mse = ((df_true_cat - df_pred_cat) ** 2).values.mean()
        rmse = np.sqrt(mse)
        
        # Compute Error Rate
        try:
            exact_matches = (df_true_cat == df_pred_cat).values.astype(bool)
        except:
            print(df_true_cat.columns,df_pred_cat.columns)
        error_count = (~exact_matches).sum()
        total_cells = exact_matches.size
        er = (error_count / total_cells) * 100
        
        results[category] = [rmse, er]
    
    # 11. Compute average RMSE and Error Rate
    mse_avg = ((df_true - df_pred) ** 2).values.mean()
    rmse_avg = np.sqrt(mse_avg)
    
    exact_matches_avg = (df_true == df_pred).values
    error_count_avg = (~exact_matches_avg).sum()
    total_cells_avg = exact_matches_avg.size
    er_avg = (error_count_avg / total_cells_avg) * 100
    
    results['average'] = [rmse_avg, er_avg]
    # print("Medium:", results['medium'][0],results['medium'][1])
    # print("Hard:", results['hard'][0],results['hard'][1])

    return results




import numpy as np
import pandas as pd
from io import StringIO


import numpy as np
import pandas as pd
from io import StringIO

import numpy as np
import pandas as pd
from io import StringIO

def compute_rmse_over_under(df_true: pd.DataFrame, df_pred: pd.DataFrame) -> dict:
    """
    Computes overcounted and undercounted RMSE between two DataFrames for 
    'Home Team' and 'Away Team', categorizing columns into 'easy', 'hard', 'medium', 
    plus an 'average' across all columns.
    
    In addition to RMSE, this function also reports error rates (percentage of cells 
    that are under- or over-predicted) for each category.
    
    Returns a dictionary of the form:
      {
        "missing columns": <int>,
        "easy":    [<under_rmse>, <over_rmse>, <under_error_rate>, <over_error_rate>],
        "hard":    [<under_rmse>, <over_rmse>, <under_error_rate>, <over_error_rate>],
        "medium":  [<under_rmse>, <over_rmse>, <under_error_rate>, <over_error_rate>],
        "average": [<under_rmse>, <over_rmse>, <under_error_rate>, <over_error_rate>]
      }
    
    Raises ValueError if:
      - column counts differ (after dropping zero-sum predictions) or
      - 'Home Team' / 'Away Team' are missing, or
      - columns are non-numeric.
    """

    # 1. Ensure "Team" is index, drop duplicates
    for df in [df_true, df_pred]:
        if 'Team' in df.columns:
            df.set_index('Team', inplace=True)
        df.index.name = 'Team'
        if 'Team' in df.columns:
            df.drop(columns=['Team'], inplace=True, errors='ignore')

    # 2. If "Red Cards" is expected but missing, add it:
    if 'Red Cards' not in df_pred.columns:
        df_pred['Red Cards'] = [0, 0]

    # 3. Optionally drop columns with zero-sum predictions
    missing_cols = 0
    for col in list(df_pred.columns):  # list to avoid issues while iterating
        if col != 'Red Cards' and df_pred[col].sum() == 0:
            df_pred.drop(columns=col, inplace=True)
            if col in df_true.columns:
                df_true.drop(columns=col, inplace=True)
            missing_cols += 1

    # 4. Confirm the columns match after any drops
    if set(df_true.columns) != set(df_pred.columns):
        raise ValueError("Column mismatch between df_true and df_pred after dropping zero-sum columns.")

    # 5. Ensure 'Home Team' and 'Away Team' rows exist
    expected_teams = {"Home Team", "Away Team"}
    missing_true = expected_teams - set(df_true.index)
    missing_pred = expected_teams - set(df_pred.index)
    if missing_true or missing_pred:
        raise ValueError(f"Missing teams - df_true: {missing_true}, df_pred: {missing_pred}")

    # 6. Reorder rows in both dataframes (Home first, Away second)
    df_true = df_true.loc[["Home Team", "Away Team"]]
    df_pred = df_pred.loc[["Home Team", "Away Team"]]

    # 7. Align columns in the same order
    df_pred = df_pred[df_true.columns]

    # 8. Verify numeric columns
    for col in df_true.columns:
        if not (pd.api.types.is_numeric_dtype(df_true[col]) and pd.api.types.is_numeric_dtype(df_pred[col])):
            raise ValueError(f"Non-numeric column detected: {col}")

    # 9. Define categories
    categories = {
        'easy':   ['Goals', 'Red Cards'],
        'hard':   ['Shots', 'Fouls'],
        'medium': ['Yellow Cards', 'Free Kicks', 'Corner Kicks', 'Offsides']
    }

    # 10. Helper function to compute the under/over RMSE and error rates for a slice.
    def _compute_over_under_metrics(df_gt: pd.DataFrame, df_pd: pd.DataFrame) -> (float, float, float, float):
        # Convert to numpy arrays for clean masking.
        gt_vals = df_gt.to_numpy()
        pd_vals = df_pd.to_numpy()
        diff = pd_vals - gt_vals

        # Create boolean masks:
        under_mask = diff < 0  # prediction is too low
        over_mask  = diff > 0  # prediction is too high

        total_cells = gt_vals.size

        # Compute RMSE for underpredictions:
        under_vals = (gt_vals - pd_vals)[under_mask]
        if under_vals.size > 0:
            under_rmse = np.sqrt((under_vals ** 2).mean())
        else:
            under_rmse = 0.0

        # Compute RMSE for overpredictions:
        over_vals = (pd_vals - gt_vals)[over_mask]
        if over_vals.size > 0:
            over_rmse = np.sqrt((over_vals ** 2).mean())
        else:
            over_rmse = 0.0

        # Compute error rates (percentage of cells that are under/over predicted):
        under_count = np.sum(under_mask)
        over_count  = np.sum(over_mask)
        under_er = (under_count / total_cells) * 100
        over_er  = (over_count / total_cells) * 100

        return under_rmse, over_rmse, under_er, over_er

    # 11. Compute metrics for each category
    results = {"missing columns": missing_cols}
    for cat, cols in categories.items():
        # Determine which columns in this category exist in df_true.
        available_cols = list(set(cols).intersection(df_true.columns))
        if len(available_cols) == 0:
            results[cat] = [0.0, 0.0, 0.0, 0.0]
            continue

        sub_true = df_true[available_cols]
        sub_pred = df_pred[available_cols]
        under_rmse, over_rmse, under_er, over_er = _compute_over_under_metrics(sub_true, sub_pred)
        results[cat] = [under_rmse, over_rmse, under_er, over_er]

    # 12. Compute overall "average" metrics across all columns.
    under_rmse_all, over_rmse_all, under_er_all, over_er_all = _compute_over_under_metrics(df_true, df_pred)
    results['average'] = [under_rmse_all, over_rmse_all, under_er_all, over_er_all]

    return results

# === Aggregation Code ===
# This code loops through multiple tables, aggregates the metrics for each category,
# and computes both the average and standard deviation for RMSE and error rates.
