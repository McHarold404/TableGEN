import json
import numpy as np
import pandas as pd
from io import StringIO
import textwrap
from model_inference.gpt import *
from utils.table_utils import *
path = './data/LiveSum/test.json'
df = pd.read_json(path)
full_length = 754
import time
model_dir = "GPT-4o-mini"
model_name = "gpt-4o-mini"
### Schema + Atomic Generation
print('Starting...')
for idx in range(0,754):
    key = 1 if idx%2 == 0 else 2
    print(ask_chatgpt(text = "Yo",prompt_path=f"prompts/Livesum/Current/livesum_atomic.txt"))
    atomic_out = ask_chatgpt(text=df['text'][idx],prompt_path=f"prompts/Livesum/Current/livesum_atomic.txt",model_name = model_name)
    with open(f"./model_outputs/Livesum/{model_dir}/Atomic/{idx}.txt","w") as f:
        f.write(atomic_out)
    header_out = ask_chatgpt(text=atomic_out,prompt_path="prompts/Livesum/Current/livesum_header.txt", model_name=model_name)
    with open(f"./model_outputs/Livesum/{model_dir}/Headers/{idx}.txt", 'w') as f:
        f.write(header_out)
    time.sleep(10)
    print(f"Finished generating statements and headers for sample {idx}")
print("Saved all results for headers and atomic statements.")