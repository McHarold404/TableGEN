import os
import json
import pandas as pd
import time
import io
import contextlib
import re

from model_inference.llama import *
from utils.table_utils import *

def ensure_dir(directory):
    """Create directory if it does not exist."""
    if not os.path.exists(directory):
        os.makedirs(directory)

def extract_code(text):
    """
    Extracts the code from the given text by removing markdown delimiters.
    If a markdown code block is found, returns the content between triple backticks.
    Otherwise, removes any standalone triple backticks.
    """
    pattern = re.compile(r"```(?:python)?\s*\n(.*?)\n```", re.DOTALL)
    match = pattern.search(text)
    if match:
        return match.group(1).strip()
    # Fall back: remove any lines that are only triple backticks
    lines = text.splitlines()
    filtered_lines = [line for line in lines if line.strip() != "```"]
    return "\n".join(filtered_lines).strip()

# Define output directories
model_dir = "Llama3.3"
base_output_path = f"./model_outputs/Livesum/{model_dir}/T3/One_Shot"
output_dirs = {
    "text_tuple": os.path.join(base_output_path, "Text-Tuple"),
    "tuple_integrated": os.path.join(base_output_path, "Tuple-Integrated"),
    "final_table": os.path.join(base_output_path, "Final-Table")
}
for key, directory in output_dirs.items():
    ensure_dir(directory)

# Load live text data
path = './data/LiveSum/test.json'
try:
    df = pd.read_json(path)
except Exception as e:
    print(f"Error loading JSON from {path}: {e}")
    exit(1)

full_length = 754
model_name = "gpt-4o"
key = 1
print("Starting the prompting pipeline...")
for idx in range(0,full_length):
    try:
        # ---------------------------
        # Step 1: Text-to-Tuple Prompt
        # ---------------------------
        try:
            text_tuples = ask_llama(
                text=df['text'][idx],
                prompt_path="prompts/Livesum/Current/T3/T3_text-tuple.txt",
                key = key
            )
        except Exception as e:
            print(f"Error during ask_chatgpt for text-tuple at idx {idx}: {e}")
            continue
        
        text_tuple_path = os.path.join(output_dirs["text_tuple"], f"{idx}.txt")
        try:
            with open(text_tuple_path, "w") as f:
                f.write(text_tuples)
        except Exception as e:
            print(f"Error writing text-tuple output at idx {idx}: {e}")
            continue

        # ----------------------------------------------
        # Step 2: Generate Python Code to Consolidate Tuples
        # ----------------------------------------------
        try:
            integration_code = ask_llama(
                text=text_tuples,
                prompt_path="prompts/Livesum/Current/T3/T3_tuple-integrate.txt",
                key = key
            )
        except Exception as e:
            print(f"Error during ask_chatgpt for tuple-integrate at idx {idx}: {e}")
            continue
        
        integrated_code_path = os.path.join(output_dirs["tuple_integrated"], f"{idx}.txt")
        try:
            with open(integrated_code_path, "w") as f:
                f.write(integration_code)
        except Exception as e:
            print(f"Error writing integrated code at idx {idx}: {e}")
            continue

        # ------------------------------------------------
        # Step 3: Execute the Consolidation Code Using exec
        # ------------------------------------------------
        # Extract the raw Python code in case there are markdown delimiters.
        code_to_exec = extract_code(integration_code)
        consolidated_output_stream = io.StringIO()
        try:
            with contextlib.redirect_stdout(consolidated_output_stream):
                exec(code_to_exec, {})  # Execute the code in an isolated namespace.
        except Exception as e:
            print(f"Error executing consolidation code at idx {idx}: {e}")
            print("Extracted code to execute:")
            print(code_to_exec)
            continue

        # Capture the printed output (expected to be the consolidated dictionary).
        consolidated_output = consolidated_output_stream.getvalue().strip()
        # Optionally, log the consolidated output
        consolidated_output_path = os.path.join(output_dirs["tuple_integrated"], f"{idx}_output.txt")
        try:
            with open(consolidated_output_path, "w") as f:
                f.write(consolidated_output)
        except Exception as e:
            print(f"Error writing consolidated output at idx {idx}: {e}")
            continue

        # -----------------------------------------------
        # Step 4: Generate the Final Table Using tuple-table Prompt
        # -----------------------------------------------
        try:
            final_table = ask_llama(
                text=consolidated_output,
                prompt_path="prompts/Livesum/Current/T3/T3_tuple-table.txt",
                key = key
            )
        except Exception as e:
            print(f"Error during ask_chatgpt for tuple-table at idx {idx}: {e}")
            continue
        
        final_table_path = os.path.join(output_dirs["final_table"], f"{idx}.txt")
        try:
            with open(final_table_path, "w") as f:
                f.write(final_table)
        except Exception as e:
            print(f"Error writing final table output at idx {idx}: {e}")
            continue
        
        print(f"Finished idx {idx}")
    except Exception as e:
        print(f"Unhandled error at idx {idx}: {e}")

print("Saved all results for the prompting pipeline.")
