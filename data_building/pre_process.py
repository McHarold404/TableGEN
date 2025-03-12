import pandas as pd

def create_row_tuples(df, row_header_column):
    """
    Creates a 2D list of tuples (row_header, col_header, value) for non-empty values in a DataFrame.
    Each sublist represents a row's tuples.

    Parameters:
    - df (DataFrame): Input DataFrame.
    - row_header_column (str): The column name to be used as row headers.

    Returns:
    - result (list of list): A list where each sublist contains tuples for non-empty values of a row.
    """
    result = []  # Final 2D list of row-wise tuples
    
    # Extract row headers and remaining columns
    row_headers = df[row_header_column]
    col_headers = [col for col in df.columns if col != row_header_column]
    
    for idx, row in df.iterrows():
        row_tuples = []  # Tuples for this row
        row_header = row[row_header_column]  # Get row header
        for col in col_headers:
            value = row[col]
            if value != '':  # Only include non-empty (non-NaN) values
                row_tuples.append((row_header, col, value))
        result.append(row_tuples)
    
    return result


import re

def extract_step_content(text, step_number=2):
    """
    Extracts the content of a specific step section from the input text.

    Parameters:
        text (str): The input string containing the step-by-step content.
        step_number (int): The step number to identify and extract content from.

    Returns:
        str: The extracted step content or a message if the step is not found.
    """
    # Updated pattern to match '###Step{number}' headers
    step_pattern = rf'###Step{step_number}\s*\n(.*?)(\n###|$)'
    
    # Search for the pattern with DOTALL to include newline characters
    match = re.search(step_pattern, text, re.DOTALL | re.IGNORECASE)
    
    try:
        # Return the captured group, stripping any leading/trailing whitespace
        return match.group(1).strip()
    
    # Return a message if the step is not found
    except Exception as e:
        return e
    
def convert_to_iostr(tuples):
    import io
    # Format the 2D list as strings preserving tuple structure
    formatted_string = "\n".join(
        ", ".join(str(item) for item in row) for row in tuples
    )

    # Convert to StringIO for LLM input
    string_io = io.StringIO(formatted_string)

    return string_io.getvalue()


import pandas as pd
import numpy as np
import random

def sparsify_table(table, mandatory_cols, sparsity_fraction=0.3, seed=None):
    """
    Takes a smaller table and introduces sparsity by randomly setting some cell values to NaN.
    Ensures:
    - Every row and column has at least one non-NaN value.
    - No values from the mandatory columns are omitted.

    Parameters:
    - table (pd.DataFrame): The smaller generated table to be sparsified.
    - mandatory_cols (list): List of mandatory columns whose values must not be omitted.
    - sparsity_fraction (float): Fraction of cells to randomly set as NaN.
    - seed (int, optional): Random seed for reproducibility.

    Returns:
    - pd.DataFrame: The sparsified table.
    """
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)

    table = table.copy()  # Work on a copy to avoid modifying the original table
    original_table = table.copy()  # Preserve original values for restoration

    # Exclude mandatory columns from sparsification
    non_mandatory_cols = [col for col in table.columns if col not in mandatory_cols]

    # Calculate total number of cells to sparsify (only for non-mandatory columns)
    total_cells = len(table) * len(non_mandatory_cols)
    cells_to_sparsify = int(total_cells * sparsity_fraction)

    # Ensure that cells_to_sparsify does not exceed the number of available cells
    cells_to_sparsify = min(cells_to_sparsify, len(non_mandatory_cols) * len(table))

    # Get all row-column index pairs for non-mandatory columns
    all_indices = [(row, col) for row in table.index for col in non_mandatory_cols]

    # Randomly select unique indices to sparsify
    sparse_indices = random.sample(all_indices, cells_to_sparsify) if cells_to_sparsify <= len(all_indices) else all_indices

    # Set the selected indices to NaN
    for row, col in sparse_indices:
        table.loc[row, col] = np.nan

    # Ensure every column has at least one non-NaN value
    for col in table.columns:
        if table[col].isnull().all():
            # Choose a random row to restore the original value
            non_nan_row = random.choice(table.index)
            original_value = original_table.loc[non_nan_row, col]
            if pd.notnull(original_value):
                table.loc[non_nan_row, col] = original_value
            else:
                # If the original value is also NaN, search for a non-NaN value in the column
                non_nan_values = original_table[col].dropna().unique()
                if len(non_nan_values) > 0:
                    table.loc[non_nan_row, col] = random.choice(non_nan_values)
                else:
                    # If all original values are NaN, set to a default placeholder or keep as NaN
                    table.loc[non_nan_row, col] = "Default_Value"  # Or keep as np.nan

    # Ensure every row has at least one non-NaN value
    for row in table.index:
        if table.loc[row].isnull().all():
            # Restore a value from a mandatory column
            # Since mandatory columns are not sparsified, they must have non-NaN values
            # Select a random mandatory column
            if len(mandatory_cols) == 0:
                raise ValueError("No mandatory columns provided. Cannot ensure rows have non-NaN values.")
            non_nan_col = random.choice(mandatory_cols)
            original_value = original_table.loc[row, non_nan_col]
            table.loc[row, non_nan_col] = original_value

    return table

import os
import pandas as pd
import random
import math
import numpy as np

def split_category_into_subtables(
    df,
    mandatory_cols,
    optional_cols,
    product_type_col,
    total_subtables,
    base_dir="Smaller_Tables/",
    row_mean=7,
    row_std=2,
    row_min=2,
    row_max=11,
    col_mean=9,
    col_std=3,
    col_min=7,
    col_max=14,
):
    """
    Splits a large DataFrame into smaller CSV files based on product types with randomized rows and columns.
    
    Parameters:
    - df (pd.DataFrame): The input DataFrame.
    - mandatory_cols (list): Columns to always include.
    - optional_cols (list): Columns to subsample.
    - product_type_col (str): Column representing the product type.
    - total_subtables (int): Total number of smaller tables to generate.
    - base_dir (str): Directory to save smaller tables.
    - row_mean (int): Mean number of rows per subtable.
    - row_std (int): Standard deviation for rows.
    - row_min (int): Minimum number of rows.
    - row_max (int): Maximum number of rows.
    - col_mean (int): Mean number of total columns per subtable (includes mandatory columns).
    - col_std (int): Standard deviation for columns.
    - col_min (int): Minimum number of total columns.
    - col_max (int): Maximum number of total columns.
    
    Returns:
    - None
    """
    os.makedirs(base_dir, exist_ok=True)
    
    # Calculate allocation based on row counts
    value_counts = df[product_type_col].value_counts()
    total_rows = value_counts.sum()
    allocation = {ptype: math.floor((count / total_rows) * total_subtables) 
                 for ptype, count in value_counts.items()}
    
    # Adjust allocation to match total_subtables
    allocated = sum(allocation.values())
    while allocated < total_subtables:
        ptype = value_counts.idxmax()
        allocation[ptype] += 1
        allocated += 1
    
    table_num = 1
    
    for product_type, num_tables in allocation.items():
        subset = df[df[product_type_col] == product_type].copy()
        subset = subset.sample(frac=1, random_state=42).reset_index(drop=True)
        total_rows_pt = subset.shape[0]
        
        for _ in range(num_tables):
            # Randomize number of rows
            rows = int(np.random.normal(row_mean, row_std))
            rows = max(row_min, min(rows, row_max, total_rows_pt))
            
            # Randomize number of columns
            cols = int(np.random.normal(col_mean, col_std))
            cols = max(col_min, min(cols, col_max, len(optional_cols) + len(mandatory_cols)))
            
            # Debugging: Print the sampled number of rows and columns
            print(f"Product Type: {product_type}, Table #{table_num}, Rows: {rows}, Columns: {cols}")
            
            # Select rows first
            if rows > subset.shape[0]:
                sampled_rows = subset.copy()
            else:
                sampled_rows = subset.sample(n=rows, replace=False, random_state=random.randint(0, 10000)).copy()
            
            # Identify optional columns that are not entirely empty in the sampled rows
            non_empty_optional = [
                col for col in optional_cols
                if not sampled_rows[col].isnull().all() and not (sampled_rows[col] == '').all()
            ]
            
            # Determine how many optional columns to select
            optional_needed = max(1, cols - len(mandatory_cols))
            optional_needed = min(optional_needed, len(non_empty_optional))  # Adjust if not enough
            
            # Select optional columns
            if optional_needed > 0:
                selected_optional = random.sample(non_empty_optional, k=optional_needed)
            else:
                selected_optional = []
            
            selected_cols = mandatory_cols + selected_optional
            selected_cols = [col for col in selected_cols if col in subset.columns]
            
            # Create subtable
            subtable = sampled_rows[selected_cols]
            
            # Save CSV
            csv_path = os.path.join(base_dir, f"{table_num}.csv")
            subtable.to_csv(csv_path, index=False,sep="|")
            print(f"Saved {csv_path} for Product Type: {product_type} with {subtable.shape[0]} rows and {subtable.shape[1]} columns.")
            
            table_num += 1
            #subset = subset.drop(sampled_rows.index)  # Uncomment if you want to prevent reuse
    
    print(f"Total tables created: {table_num - 1}")