import os
import pandas as pd
from concurrent.futures import ThreadPoolExecutor
from model_inference.llama import ask_llama

# Configuration
model_dir = "Llama3.3"
output_atomic_dir = f"model_outputs/Livesum/{model_dir}/Atomic"
output_headers_dir = f"model_outputs/Livesum/{model_dir}/Headers"
output_table_dir = f"model_outputs/Livesum/{model_dir}/TabGen_1step_from_Atomic"
os.makedirs(output_table_dir, exist_ok=True)

# Get the number of samples available
num_samples = len([f for f in os.listdir(output_atomic_dir) if f.endswith('.txt')])
num_threads = min(150, num_samples)  # Use at most 150 threads, or less if fewer samples

# Function to process a batch of indices
def process_batch(indices):
    for idx in indices:
        key = 1

        atomic_file = f"{output_atomic_dir}/{idx}.txt"
        header_file = f"{output_headers_dir}/{idx}.txt"
        output_file = f"{output_table_dir}/{idx}.txt"
        output_file_check = f"{output_table_dir}/realigned/{idx}.txt"
        # Skip processing if output already exists
        if os.path.exists(output_file_check):
            print(f"Skipping {idx}, already processed.")
            continue
        
        # Ensure required input files exist
        if not os.path.exists(atomic_file) or not os.path.exists(header_file):
            print(f"Skipping {idx}, missing input files.")
            continue

        # Read atomic statements
        with open(atomic_file, 'r') as f:
            sentences = f.readlines()
        formatted_output = "\n".join(f"{i}:{line.strip()}" for i, line in enumerate(sentences))

        # Read header
        # with open(header_file, 'r') as f:
        #     header_out = f.read()
        header_out = '''{
            "row_header" : ["Home Team","Away Team"],
            "columns_headers" : ["Goals","Shots", "Fouls","Yellow Cards","Red Cards","Corner Kicks","Free Kicks","Offsides"]
            }
            '''

        # Construct input text
        input_text = f"Table Schema:\n{header_out}\nStatements:\n{formatted_output}"
        
        # Generate table output
        output_table = ask_llama(text=input_text, prompt_path="prompts/Livesum/Current/fill_table_gemini.txt", key=key)

        # Save output
        with open(output_file, 'w') as f:
            f.write(output_table)

        print(f"Saved results for idx {idx}")
        print("*****************************")

# Distribute indices across threads
chunk_size = num_samples // num_threads + (num_samples % num_threads > 0)
indices_list = [list(range(i, min(i + chunk_size, num_samples))) for i in range(0, num_samples, chunk_size)]

print(f"Starting parallel execution with {num_threads} threads...")

# Run multithreaded execution
try:
    with ThreadPoolExecutor(max_workers=num_threads) as executor:
        executor.map(process_batch, indices_list)
except KeyboardInterrupt:
    print("Execution interrupted. Safely exiting...")

print("Saved all results for table generation.")

    
    