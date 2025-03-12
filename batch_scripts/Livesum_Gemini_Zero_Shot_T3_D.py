import json
import numpy as np
import pandas as pd
from io import StringIO
import textwrap
from model_inference.gemini import *
from utils.table_utils import *
path = './data/LiveSum/test.json'
df = pd.read_json(path)
full_length = 754
import time
model_dir = "Gemini-2.0-flash-exp"
model_name = "gemini-2.0-flash-exp"
### Schema + Atomic Generation
print('Starting...')
for idx in range(100,full_length):
    key = 13
    atomic_out = ask_gemini(text=df['text'][idx],prompt_path=f"prompts/Livesum/Current/T3_zero_shot_direct.txt",model_name = model_name, key = key)
    with open(f"./model_outputs/Livesum/{model_dir}/T3/Zero_Shot/Direct/{idx}.txt","w") as f:
        f.write(atomic_out)
    time.sleep(10)
    print(f"processed sample {idx}")
print("Processed all samples")