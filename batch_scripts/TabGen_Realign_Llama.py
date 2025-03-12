import argparse
import os
import csv
import re
import pandas as pd
import numpy as np
import time
import threading
from io import StringIO
from concurrent.futures import ThreadPoolExecutor, as_completed
from model_inference.llama import ask_llama
from eval.livesum_eval import extract_table

# Global request counter and lock
request_count = 0
request_lock = threading.Lock()

# Function to reset request counter every 60 seconds
def reset_request_count():
    global request_count
    while True:
        time.sleep(60)
        with request_lock:
            request_count = 0

# Start background thread for rate limit reset
reset_thread = threading.Thread(target=reset_request_count, daemon=True)
reset_thread.start()

# Function to process a batch of table realignments
def process_sample(idx, exp_name, model_dir, data):
    global request_count

    try:
        fname = f"model_outputs/Livesum/{model_dir}/{exp_name}/{idx}.txt"
        realigned_fname = f"model_outputs/Livesum/{model_dir}/{exp_name}/realigned/{idx}.txt"

        # Skip if realigned file already exists
        if os.path.exists(realigned_fname):
            print(f"Skipping {idx}, already processed.")
            return None

        # Skip if original file is missing
        if not os.path.exists(fname):
            print(f"File not found: {fname}")
            return None

        # Read table content
        with open(fname, "r") as f:
            file_content = f.read()

        # Extract table (with error handling)
        try:
            dfa = extract_table(file_content)
        except Exception as e:
            print(f"Table extraction failed for file: {fname}, Error: {e}")
            return None

        # Convert table to input format
        table_input = [dfa.columns.tolist()] + dfa.values.tolist()
        input_str_pipe = "\n".join(["|".join(map(str, row)) for row in table_input])

        # Rate limiting: Ensure no more than 300 requests per minute
        while True:
            with request_lock:
                if request_count < 300:
                    request_count += 1
                    break  # Proceed if under limit
            time.sleep(1)  # Wait before checking again

        # Query Gemini API for realignment (with retry mechanism)
        for attempt in range(3):  # Retry up to 3 times
            try:
                realigned_table = ask_llama(text=input_str_pipe, prompt_path='prompts/Livesum/Our_Method/realign.txt', key=1)
                break  # Exit retry loop on success
            except Exception as e:
                print(f"Attempt {attempt + 1} failed for idx {idx}. Error: {e}")
                time.sleep(5)  # Small delay before retrying
        else:
            print(f"Max retries reached for idx {idx}. Skipping...")
            return None

        # Save realigned table
        os.makedirs(os.path.dirname(realigned_fname), exist_ok=True)
        with open(realigned_fname, "w") as f:
            f.write(realigned_table)

        # Verify extracted table after realignment
        try:
            dfa = extract_table(realigned_table)
        except Exception as e:
            print(f"Table extraction failed after realignment. Idx: {idx}, Error: {e}")
            return None

        # Load ground truth table from dataset
        table_string = data['table'][idx].replace('<NEWLINE>', '\n')
        df_table = pd.read_csv(StringIO(table_string))

        print(f"Processed idx {idx}: Realigned table shape {dfa.shape}")

        return dfa  # Returning dfa for optional further processing

    except Exception as e:
        print(f"Unexpected error for idx {idx}: {e}")
        return None

# Multi-threaded function to process tables in parallel
def process_tables(exp_name, model_dir):
    path = 'data/LiveSum/test.json'
    start, end = 0,754
    data = pd.read_json(path)

    num_threads = min(50, (end - start))  # Use up to 50 threads or fewer if not enough samples
    indices = list(range(start, end))

    print(f"Starting parallel execution with {num_threads} threads...")

    # Execute in parallel
    with ThreadPoolExecutor(max_workers=num_threads) as executor:
        futures = {executor.submit(process_sample, idx, exp_name, model_dir, data): idx for idx in indices}

        for future in as_completed(futures):
            try:
                result = future.result()  # Process results if needed
            except Exception as e:
                print(f"Error processing idx {futures[future]}: {e}")

    print("Processing complete.")

# Main execution
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp_name", type=str, required=True, help="Experiment name")
    parser.add_argument("--model_dir", type=str, required=True, help="Model directory")
    args = parser.parse_args()

    process_tables(args.exp_name, args.model_dir)
