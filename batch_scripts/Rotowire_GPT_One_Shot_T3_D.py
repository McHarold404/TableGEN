import json
import numpy as np
import pandas as pd
from io import StringIO
import textwrap
from model_inference.gpt import *
from model_inference.gemini import *
from utils.table_utils import *
from utils.rotowire_utils import *
import pandas as pd
import numpy as np

full_length = 728
output_dir = "GPT4o"
fname_text = f"data/rotowire/test.text"
with open(fname_text,"r") as f:
    all_text = f.readlines()

print('Starting...')
for idx in range(0,full_length):
    ### Generate 
    gold_text = all_text[idx]
    response  = ask_chatgpt(text = gold_text, prompt_path="prompts/Rotowire/Current/T3_one_shot_direct.txt",model_name = "gpt-4o")
    with open(f"model_outputs/Rotowire/{output_dir}/T3/One_Shot/Direct/{idx}.txt","w") as f:
        f.write(response)       
    print('Saved results for idx',idx)
    print("***********")