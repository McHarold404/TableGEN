import json
import numpy as np
import pandas as pd
import time
import argparse
from io import StringIO
from model_inference.gpt import *
from model_inference.gemini import *
from utils.table_utils import *
from eval.livesum_eval import *


def main(model_dir, model_name, exp_name, prompt_path):
    path = './data/LiveSum/test.json'
    df = pd.read_json(path)
    full_length = len(df)
    
    print('Starting...')
    for idx in range(full_length):
        key = 11 if idx % 2 == 1 else 12
        baseline_table = ask_gemini(
            text=df['text'][idx], 
            prompt_path=f"prompts/Livesum/Current/{prompt_path}.txt", 
            key=key, 
            model_name=model_name
        )
        
        output_path = f"./model_outputs/Livesum/{model_dir}/{exp_name}/{idx}.txt"
        with open(output_path, "w") as f:
            f.write(baseline_table)        
        print(f'Saved results for idx {idx}')
        print("*****************************")
        time.sleep(5)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run Gemini model inference and save results.")
    parser.add_argument("--model_dir", type=str, required=True, help="Directory to save model outputs")
    parser.add_argument("--model_name", type=str, required=True, help="Name of the model to use")
    parser.add_argument("--exp_name", type=str, required=True, help="Experiment name")
    parser.add_argument("--prompt_path",type = str, required = True,help ="Prompt file name")
    
    
    args = parser.parse_args()
    main(args.model_dir, args.model_name, args.exp_name,args.prompt_path)
