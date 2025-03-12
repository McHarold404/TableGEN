import re
import pandas as pd
import numpy as np

def extract_final_table(file_contents: str) -> pd.DataFrame:
    """
    Extracts the final table from the provided file contents and returns it as a pandas DataFrame.
    
    The function handles tables where the first header cell is either empty or contains "Team",
    and ensures that "Team" is always a column in the resulting DataFrame.
    
    Args:
        file_contents (str): The content of the file as a single string.
        
    Returns:
        pd.DataFrame: The extracted final table as a DataFrame.
        
    Raises:
        ValueError: If the '### Final Table' section or table lines are not found.
    """
    
    # Locate the start of the final table section using a regex search
    start_match = re.search(r"###\s*Final\s*Table", file_contents, re.IGNORECASE)
    if not start_match:
        raise ValueError("No '### Final Table' heading found.")
        
    # Extract the substring starting from the end of the '### Final Table' heading
    final_table_start = start_match.end()
    text_after_final_table = file_contents[final_table_start:].strip()
    
    # Split the remaining text into individual lines
    lines = text_after_final_table.split('\n')
    
    # Clean lines by stripping whitespace and ignoring empty lines
    lines = [line.strip() for line in lines if line.strip()]
    
    # Identify all table lines:
    # These lines start with '|' and contain '<NEWLINE>'
    table_lines = [line for line in lines if line.startswith('|') and '<NEWLINE>' in line]
    
    if not table_lines:
        raise ValueError("No table lines found in the final table section.")
    
    # The first line is assumed to be the header
    header_line = table_lines[0]
    data_lines = table_lines[1:]
    
    # Split the header line by '|' and clean each column name
    columns = [col.strip() for col in header_line.split('|') if col.strip() and col.strip() != '<NEWLINE>']
    
    # If the first header cell is empty, set it to 'Team'
    if columns and columns[0] == '':
        columns[0] = 'Team'
    elif columns and columns[0].lower() != 'team':
        # If the first header is not 'Team', assume it needs to be added
        columns.insert(0, 'Team')
        # Adjust data accordingly
        for row in data_lines:
            # Prepend an empty string if 'Team' column was not present
            row = '| ' + row
    # Process each data line similarly
    data = []
    for line in data_lines:
        # Split by '|' and clean each cell
        parts = [part.strip() for part in line.split('|') if part.strip() and part.strip() != '<NEWLINE>']
        # Replace 'Not found' with 0
        parts = [0 if p == 'Not found' else p for p in parts]
        data.append(parts)
    
    # Extract team names (first element of each row) and data values
    team_names = [row[0] for row in data]
    data_values = [row[1:] for row in data]
    
    # If the first column was empty and set to 'Team', include 'Team' as a column
    if columns and columns[0] == 'Team':
        data_columns = columns  # Include 'Team' as a column
    else:
        # If the first column is not 'Team', insert 'Team' as the first column
        columns.insert(0, 'Team')
        data_columns = columns
    
    # Create the DataFrame with 'Team' as a column
    df = pd.DataFrame(data, columns=data_columns)
    
    # Replace any remaining "Not found" with 0 just in case
    df.replace("Not found", 0, inplace=True)
    
    # Convert all columns except 'Team' to numeric types, coercing errors to NaN and then filling with 0
    for col in df.columns:
        if col.lower() != 'team':
            df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)
    
    # Ensure 'Team' is a column, not an index
    if df.index.name == 'Team':
        df.reset_index(inplace=True)
    
    return df


import numpy as np
import pandas as pd

def compute_rmse_and_er_cell_by_cell(df_true: pd.DataFrame, df_pred: pd.DataFrame) -> (float, float):
    """
    Compute Root Mean Squared Error (RMSE) and Error Rate (ER) between two DataFrames
    by comparing cell-by-cell numerical values, aligning based on 'Team' column or index.

    Args:
        df_true (pd.DataFrame): The ground truth DataFrame, must contain 'Team' column or index.
        df_pred (pd.DataFrame): The predicted DataFrame, must contain 'Team' column or index.

    Returns:
        tuple: (rmse, er) where rmse is the Root Mean Squared Error,
               and er is the Error Rate percentage.

    Raises:
        ValueError: If necessary conditions for comparison are not met.
    """
    
    # Ensure 'Team' is a column in both DataFrames
    if 'Team' not in df_true.columns and 'Team' in df_true.index.names:
        df_true = df_true.reset_index()
    if 'Team' not in df_pred.columns and 'Team' in df_pred.index.names:
        df_pred = df_pred.reset_index()
    
    if 'Team' not in df_true.columns:
        raise ValueError("The true DataFrame must contain a 'Team' column.")
    if 'Team' not in df_pred.columns:
        raise ValueError("The predicted DataFrame must contain a 'Team' column.")
    
    # Rename 'Shots Taken' to 'Shots' in the predicted DataFrame if present
    if 'Shots Taken' in df_pred.columns:
        df_pred = df_pred.rename(columns={'Shots Taken': 'Shots'})
    
    # Define expected teams
    expected_teams = {"Home Team", "Away Team"}
    
    # Extract teams present in both DataFrames
    true_teams = set(df_true['Team'])
    pred_teams = set(df_pred['Team'])
    
    # Verify that both DataFrames contain the expected teams
    if not expected_teams.issubset(true_teams):
        missing = expected_teams - true_teams
        raise ValueError(f"The true DataFrame does not contain the following expected teams: {missing}")
    if not expected_teams.issubset(pred_teams):
        missing = expected_teams - pred_teams
        raise ValueError(f"The predicted DataFrame does not contain the following expected teams: {missing}")
    
    # Set 'Team' as index to facilitate alignment
    df_true_indexed = df_true.set_index('Team')
    df_pred_indexed = df_pred.set_index('Team')
    
    # Ensure that both DataFrames have the same numerical columns
    if not df_true_indexed.columns.equals(df_pred_indexed.columns):
        raise ValueError("Both DataFrames must have the same numerical columns.")
    
    # Reorder df_pred to match the order of df_true
    df_pred_indexed = df_pred_indexed.loc[df_true_indexed.index]
    
    # Ensure that after reindexing, there are no missing teams
    if df_pred_indexed.isnull().values.any():
        raise ValueError("Mismatch in teams between true and predicted DataFrames after alignment.")
    
    # Compute the difference
    diff = df_true_indexed - df_pred_indexed
    
    # Compute RMSE
    squared_diff = diff ** 2
    mse = squared_diff.mean().mean()  # Mean of squared differences
    rmse = np.sqrt(mse)
    
    # Compute Error Rate: percentage of cells where true != pred
    error_cells = (df_true_indexed != df_pred_indexed).sum().sum()
    total_cells = df_true_indexed.size
    er = (error_cells / total_cells) * 100
    
    return rmse, er