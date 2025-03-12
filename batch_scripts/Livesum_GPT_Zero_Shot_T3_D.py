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
model_dir = "GPT-4o"
model_name = "gpt-4o"
### Schema + Atomic Generation
print('Starting...')
for idx in range(0,full_length):
    atomic_out = ask_chatgpt(text=df['text'][idx],prompt_path=f"prompts/Livesum/Current/T3_zero_shot_direct.txt",model_name = model_name)
    with open(f"./model_outputs/Livesum/{model_dir}/T3/Zero_Shot/Direct/{idx}.txt","w") as f:
        f.write(atomic_out)
    print(f"Finished idx {idx}")

print("Saved all results for headers and atomic statements.")