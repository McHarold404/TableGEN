# import json
# import numpy as np
# import pandas as pd
# from io import StringIO
# import textwrap
# from model_inference.llama import *
# from utils.table_utils import *
# path = './data/LiveSum/test.json'
# df = pd.read_json(path)
# full_length = 754
# import time
# model_dir = "Llama3.3"
# ### Schema + Atomic Generation
# print('Starting...')
# for idx in range(0,754):
#     key = 1
#     atomic_out = ask_llama(text=df['text'][idx],prompt_path=f"prompts/Livesum/Current/livesum_atomic.txt",key = key)
#     with open(f"./model_outputs/Livesum/{model_dir}/Atomic/{idx}.txt","w") as f:
#         f.write(atomic_out)
#     # header_out = ask_llama(text=atomic_out,prompt_path="prompts/Livesum/Current/livesum_header.txt", key = key )
#     # with open(f"./model_outputs/Livesum/{model_dir}/Headers/{idx}.txt", 'w') as f:
#     #     f.write(header_out)

#     print(f"Finished generating statements and headers for sample {idx}")
# print("Saved all results for headers and atomic statements.")
import json
import numpy as np
import pandas as pd
import os
from concurrent.futures import ThreadPoolExecutor
from model_inference.llama import ask_llama
from utils.table_utils import *
import time

# Load Data
path = './data/LiveSum/test.json'
df = pd.read_json(path)
full_length = len(df)
model_dir = "Llama3.3"
output_atomic_dir = f"./model_outputs/Livesum/{model_dir}/Atomic"
os.makedirs(output_atomic_dir, exist_ok=True)

# Define number of threads
num_threads = 150  
chunk_size = full_length // num_threads + (full_length % num_threads > 0)  # Distribute indices across threads
indices_list = [list(range(i, min(i + chunk_size, full_length))) for i in range(0, full_length, chunk_size)]

# Function to process a batch of samples
def process_batch(indices):
    for idx in indices:
        output_file = f"{output_atomic_dir}/{idx}.txt"

        # Skip already processed files (prevents reprocessing after cancellation)
        if os.path.exists(output_file):
            print(f"Skipping {idx}, already processed.")
            continue

        key = 1
        atomic_out = ask_llama(text=df['text'][idx], prompt_path=f"prompts/Livesum/Current/livesum_atomic.txt", key=key)

        # Save output
        with open(output_file, "w") as f:
            f.write(atomic_out)

        print(f"Finished generating atomic statements for sample {idx}")

print(f"Starting parallel execution with {num_threads} threads...")

try:
    with ThreadPoolExecutor(max_workers=num_threads) as executor:
        executor.map(process_batch, indices_list)  # Each thread gets a list of indices to process
except KeyboardInterrupt:
    print("Execution interrupted. Safely exiting...")

print("Saved all results for atomic statements.")
