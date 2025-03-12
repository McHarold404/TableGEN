
import json
import numpy as np
import pandas as pd
from io import StringIO
import textwrap
from model_inference.gpt import *
from model_inference.gemini import *
from utils.table_utils import *

from eval.livesum_eval import *
import pandas as pd
import numpy as np
path = './data/LiveSum/test.json'
df = pd.read_json(path)
full_length = 754
#### Dont run for 100-250 for 1.5 flash.
model_dir = "Gemini1.5"
model_name = "gemini-1.5-flash"

print('Starting...')
for idx in range(0,754):
    key = 5 if idx % 2 == 1 else 6
    with open(f"model_outputs/Livesum/{model_dir}/Atomic/{idx}.txt",'r') as f:
        sentences = f.readlines()
    formatted_output = "\n".join(f"{i}:{line.strip('\n')}" for i,line in enumerate(sentences))
    time.sleep(4)
    with open(f"model_outputs/Livesum/{model_dir}/Headers/{idx}.txt","r") as f:
        header_out = f.read()
    input_text = "Table Schema: \n" + header_out + '\n' + "Statements: \n" + formatted_output
    output_table = ask_gemini(text=input_text,prompt_path="prompts/Livesum/Current/fill_table_gemini.txt",key = key,model_name=model_name)
    with open(f"model_outputs/Livesum/{model_dir}/TabGen_1step_from_Atomic/{idx}.txt",'w') as f:
       f.write(output_table)
    print('Saved results for idx',idx)
    print("*****************************")
    
    