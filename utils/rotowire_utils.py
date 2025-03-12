import re
import json
import pandas as pd


def parse_text(summary):
    """
    Parses the input text with sections separated by '###' and extracts mappings and summary.

    Args:
        summary (str): Input text containing player mappings, team mappings, and a summary.

    Returns:
        dict: A dictionary containing player mappings, team mappings, and the updated summary.
    """
    # Split sections by "###"
    sections = summary.split("###")
    if len(sections) < 3:
        raise ValueError("The input text does not have the required sections separated by '###'.")

    # Extract the Summary section
    summary_text = None
    for section in sections:
        if section.strip().startswith("Summary"):
            summary_text = section.split("Summary", 1)[1].strip()
            break

    if not summary_text:
        raise ValueError("The 'Summary' section is missing from the input text.")

    # Extract the Mapping section
    mapping_text = None
    for section in sections:
        if section.strip().startswith("Mapping"):
            mapping_text = section.split("Mapping:", 1)[1].strip()
            break

    if not mapping_text:
        raise ValueError("The 'Mapping' section is missing from the input text.")

    # Clean and process the mappings
    mapping_lines = mapping_text.splitlines()
    cleaned_mapping_text = "\n".join(
        line for line in mapping_lines if line.strip() and not line.startswith("```")
    )
    mappings = json.loads(cleaned_mapping_text)
    
    # Extract player and team mappings
    player_mappings = mappings.get("Player Mappings", {})
    team_mappings = mappings.get("Team Mappings", {})

    # # Replace mapped names in the summary
    # parsed_summary = summary_text
    # for mapped, original in player_mappings.items():
    #     parsed_summary = parsed_summary.replace(original, mapped)
    # for mapped, original in team_mappings.items():
    #     parsed_summary = parsed_summary.replace(original, mapped)

    # Return the parsed data
    return {
        "Player Mappings": player_mappings,
        "Team Mappings": team_mappings,
        "Parsed Summary": summary_text
    }
    
    

import pandas as pd
import re

def parse_table(section):
    """
    Parses a table section and returns it as a DataFrame.

    Args:
        section (str): The section text starting with the table header.

    Returns:
        DataFrame: A pandas DataFrame representing rows of the table.
    """
    # Split the section into lines and filter out lines without alphanumeric characters
    lines = [line.strip() for line in section.split('\n') if line.strip() and any(char.isalnum() for char in line)]


    # Extract headers
    headers_line = lines[0]
    headers = [col.strip() for col in headers_line.strip('|').split('|')]

    # Process rows starting after the separator
    data_lines = lines[1:]
    rows = []
    for line in data_lines:
        # Split each data line into values
        values = [val.strip() for val in line.strip('|').split('|')]
        # Only consider rows that have the correct number of columns
        if len(values) == len(headers):
            rows.append(values)
        else:
            # If row length mismatch, skip this row or handle accordingly
            print(f"Skipping misaligned row: {line}")

    return pd.DataFrame(rows, columns=headers)

def extract_team_and_player_tables(file_content):
    """
    Extracts team and player tables as DataFrames from the provided content.

    Args:
        file_content (str): The input content containing team and player tables.

    Returns:
        tuple: Two pandas DataFrames, one for the team table and one for the player table.
    """
    # Split the content to process tables separately
    try:
        team_section = file_content.split("### Team Table:")[1].split("### Player Table:")[0].strip()
        player_section = file_content.split("### Player Table:")[1].strip()
    except IndexError:
        raise ValueError("Could not find both team and player sections in the provided content.")

    # Parse each section into a DataFrame

    team_table_df = parse_table(team_section)
    player_table_df = parse_table(player_section)

    return team_table_df, player_table_df



def extract_atomic_statements(text):
    """
    Extracts the section below '### FINAL STATEMENTS' from the given text.

    Parameters:
        text (str): The input text containing various sections.

    Returns:
        str: The extracted final statements section, or None if not found.
    """
    # Define a regex pattern to match '### FINAL STATEMENTS' followed by the statements
    pattern = r'### Atomic Statements:\s*\n([\s\S]+)'

    # Search for the pattern in the text
    match = re.search(pattern, text, re.IGNORECASE)

    # If a match is found, return the captured group, stripped of leading/trailing whitespace
    if match:
        return match.group(1).strip()
    
    # If no match is found, return None
    return None


import re

def extract_schema(text):
    """
    Extracts the JSON schema portion from the input text after '###Final Schema:'.

    Args:
        text (str): The input text containing the schema.

    Returns:
        str or None: The extracted JSON schema as a string, or None if not found.
    """
    # Define the regex pattern
    pattern = r"### Final Schema:\s*(\{[\s\S]*\})"

    # Search for the pattern in the text
    match = re.search(pattern, text)

    # If a match is found, return the captured group (the JSON schema)
    if match:
        return match.group(1).strip()
    
    # If no match is found, return None
    return None

# Example usage:
# input_text = """... your provided text ..."""
# schema = extract_final_schema(input_text)
# print(schema)



def extract_final_tables(text):
    """
    Extracts the final Team and Player tables from the provided text after the '### Final Tables:' section
    and returns them as Pandas DataFrames.

    Parameters:
    - text (str): The input text containing the tables.

    Returns:
    - team_df (pd.DataFrame): DataFrame containing the Team table.
    - player_df (pd.DataFrame): DataFrame containing the Player table.
    """

    # Step 1: Locate the '### Final Tables:' section
    final_tables_pattern = re.compile(r'### Final Table:', re.MULTILINE)
    final_tables_match = final_tables_pattern.search(text)
    if not final_tables_match:
        raise ValueError("The '### Final Tables:' section was not found in the provided text.")

    # Extract the text after '### Final Tables:'
    post_final_tables_text = text[final_tables_match.end():]

    # Step 2: Define regex patterns to capture Team and Player tables after '### Final Tables:'
    team_pattern = re.compile(
        r'### Team:\n((?:\|.*\|<NEWLINE>\n?)+)<TABLE END>', re.MULTILINE)
    player_pattern = re.compile(
        r'### Player:\n((?:\|.*\|<NEWLINE>\n?)+)<TABLE END>', re.MULTILINE)

    # Step 3: Search for Team table in the post_final_tables_text
    team_match = team_pattern.search(post_final_tables_text)
    if not team_match:
        print("Team table not found")
        team_table_text = None
    else:
        team_table_text = team_match.group(1).strip()
    # Step 4: Search for Player table in the post_final_tables_text
    player_match = player_pattern.search(post_final_tables_text)
    if not player_match :
        print("Player table not found")
        player_table_text = None
    else: 
        player_table_text = player_match.group(1).strip()
    #print(team_table_text,player_table_text)

    def parse_table(table_text):
        """
        Parses the table text and converts it into a Pandas DataFrame.

        Parameters:
        - table_text (str): The raw table text.

        Returns:
        - df (pd.DataFrame): The parsed DataFrame.
        """
        if table_text is None:
            return None
        # Split the table into lines based on <NEWLINE>
        lines = table_text.strip().split('<NEWLINE>')

        # Remove any empty lines and strip whitespace
        lines = [line.strip() for line in lines if line.strip()]

        if not lines:
            raise ValueError("Empty table found.")

        # Extract headers
        header_line = lines[0]
        headers = [header.strip() for header in header_line.strip('|').split('|')]

        # Initialize list to hold row data
        data = []

        # Iterate over the remaining lines to extract row data
        for line in lines[1:]:
            # Ensure the line starts and ends with '|'
            if line.startswith('|') and line.endswith('|'):
                # Split the line into individual cell values
                cells = [cell.strip() for cell in line.strip('|').split('|')]
                data.append(cells)
            else:
                # Handle unexpected line formats if necessary
                continue

        # Create DataFrame
        df = pd.DataFrame(data, columns=headers)

        # Replace 'None' strings with actual None (NaN in DataFrame)
        df.replace('None', pd.NA, inplace=True)

        return df

    # Step 5: Parse both tables
    try:
        team_df = parse_table(team_table_text)
    except:
        team_df = None
    try:
        player_df = parse_table(player_table_text)
    except:
        player_df = None

    return team_df, player_df
