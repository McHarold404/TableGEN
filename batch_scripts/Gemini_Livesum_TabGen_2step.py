
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



print('Starting...')
for idx in range(0,full_length):
    superset = paragraph_to_numbered_sentences(paragraph= df['text'][idx])
    sz = len(superset)
    set1 = superset[:int(sz/2)]
    set2 = superset[int(sz/2):]
    formatted_output_1 = "\n".join(set1)
    formatted_output_2 = "\n".join(set2)
    ## Call 1
    with open(f"model_outputs/Livesum/Gemini1.5/Headers/{idx}.txt","r") as f:
        header_out = f.read()
    input_text = "Table Schema: \n" + header_out + '\n' + "Statements: \n" + formatted_output_1
    output_table = ask_gemini(text=input_text,prompt_path="prompts/Livesum/Our_Method/fill_table_gemini.txt",key = 5)
    with open(f"model_outputs/Livesum/Gemini1.5/TabGen_2step/step1/{idx}.txt",'w') as f:
       f.write(output_table)
    try:
        dfa = extract_table(output_table)
    except:
        print("First step bugged out,skipping this index")
    #print(dfa)
    ## Prep table as str for call 2
    table_input = [dfa.columns.tolist()] + dfa.values.tolist()
    input_table = table_input
    # columns = input_table[0]  # Extract column names from the first row
    # data = input_table[1:]    # Remove the header row from the data
    # dfa = pd.DataFrame(data,columns=columns)
    input_str_pipe = "\n".join(["|".join(map(str, row)) for row in table_input])
    ## Call 2
    input_text = "Given Table: \n" + input_str_pipe + 'Statements: \n' + formatted_output_2 
    final_output = ask_gemini(text=input_text,prompt_path="prompts/Livesum/Our_Method/fill_table_2_gemini.txt", key = 6)
    with open(f"model_outputs/Livesum/Gemini1.5/TabGen_2step/step2/{idx}.txt",'w') as f:
        f.write(final_output)
    print('Saved results for idx',idx)
    print("*****************************")
    
    
