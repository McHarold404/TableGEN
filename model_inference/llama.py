import os
from datetime import datetime
import json
from groq import Groq
from dotenv import load_dotenv
from dotenv import *

class LlamaBot:
    def __init__(self, api_key=None, model="llama3-70b-8192", 
                 data_path=None,
                 start_line = None,
                 end_line = None,
                 limit_rows = None,
                 meta_data = None,
                 file_name = None,
                 output_path = None,
                 data_string = None,
                 prompt_path=None):
        self.api_key = api_key or os.getenv("GROQ_API_KEY")
        if not self.api_key:
            raise ValueError("API key is required. Set it in the constructor or as an environment variable 'GROQ_API_KEY'")
        self.client = Groq(api_key=self.api_key)
        self.model = model
        self.limit_rows = limit_rows
        self.file_name = file_name
        self.data_string = data_string
        self.start_line = start_line
        self.end_line = end_line
        self.meta_data = meta_data
        '''
        output folder config
        '''
        os.makedirs(output_path, exist_ok=True)
        # Define the final output path using "file_name"
        self.output_dir = output_path
        self.output_path = os.path.join(output_path, file_name)
        #self.output_path = os.path.join(os.path.dirname(output_path), f"llama_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{os.path.basename(output_path)}")
        self.data_path = data_path
        self.prompt_path = prompt_path
    
    def get_api_response(self, prompt = None, user_message = None):
        try:
            chat_completion = self.client.chat.completions.create(
                messages=[
                    {
                        "role" : "system",
                        "content" : prompt,
                    },
                    {
                        "role": "user",
                        "content": user_message,
                    }
                    
                ],
                model=self.model
            )
            return chat_completion.choices[0].message.content
        except Exception as e:
            return f"An error occurred: {str(e)}"
        
    def load_prompt(self):
        try:
            with open(self.prompt_path, 'r') as file:
                return file.read().strip()
        except FileNotFoundError:
            print(f"Prompt file not found at {self.prompt_path}")
            return ""
        except Exception as e:
            print(f"Error reading prompt file: {str(e)}")
            return ""

    def load_data(self):
        try:
            with open(self.data_path, 'r') as file:
                return [line.strip() for line in file if line.strip()]
        except FileNotFoundError:
            if self.data_string is not None:
                print('Data file not found, returning string used')
                return [self.data_string]
            else:
                print(f"Data file not found at {self.data_path}")
                return []
        except Exception as e:
            print(f"Error reading data file: {str(e)}")
            return []

    def load_meta_data(self):
        if self.meta_data is None:
            print(f"Error reading meta data, file not found")
            return []
        try:
            with open(self.meta_data,"r") as json_file:
                meta = json.load(json_file)
            return [x['response'] for x in meta]
        
        except Exception as e:
            print(f"Error reading data file: {str(e)}")
            return []


    def run_inference(self):
        print("Starting inference")
        data = self.load_data()
        prompt = self.load_prompt()
        meta = self.load_meta_data()
        
        if not data and not meta:
            print("No data to process. Exiting.")
            return
        
        if not prompt:
            print("No prompt loaded. Exiting.")
            return
        
        results = []
        final_data = []
        if not data:
            final_data = meta
        elif not meta:
            final_data = data
        else:
            assert(len(meta) == len(data))
            for i in range(len(data)):
                final_data.append(f"{meta[i]}:\n{data[i]}")
            

        for i, point in enumerate(final_data):
            if self.start_line is not None:
                if i < self.start_line:
                    continue
            if self.end_line is not None:
                if i > self.end_line:
                    continue
            if( self.limit_rows is not None and i >= self.limit_rows):
                break
            #if len(meta) == 0:
            message = f"{prompt}:\n{point.strip('\n')}"
            final_input = message
            response = self.get_api_response(final_input)
            results.append({'data_point': i,'input': point,'response': response})
            print(f"Data point {i}:")
            #print(f"Input: {point}")
            #print(f"AI Response: {response}")
            #print("-" * 50)
        ## Store outputs
        with open(self.output_path,"w") as json_file:
            json.dump(results,json_file,indent=2)

        ## Store prompts
        with open(f"{self.output_dir}/{os.path.basename(self.prompt_path)}","w") as f:
            f.writelines(prompt)

        print("Finished inference")
        print("Stored Prompts at",f"{self.output_dir}/{os.path.basename(self.prompt_path)}")
        print("Model output stored at", self.output_path)
        print("---------------------")
        return results
       

import os
from dotenv import load_dotenv
import requests

def load_api_key(key):
    """Load the API key from the environment variable."""
    load_dotenv()
    if key == 1:
        return os.getenv("NOVITA_API_KEY_1")
    else:
        return os.getenv("NOVITA_API_KEY_2")
def load_prompt(prompt_path):
    """Load the prompt text from the file."""
    try:
        with open(prompt_path, 'r') as file:
            return file.read().strip()
    except FileNotFoundError:
        raise ValueError(f"Prompt file not found: {prompt_path}")

def ask_llama(text, key, prompt_path=None):
    """Send a prompt and text to the Llama 3.3-70B model and return the response."""
    if not text:
        return "Error: Text missing"

    if prompt_path:
        prompt = load_prompt(prompt_path)
    else:
        return "Error: Prompt or text missing"

    api_key = load_api_key(key = key)
    if not api_key:
        return "Error: API key not found"

    # Define the API endpoint and header
    
    from openai import OpenAI

    client = OpenAI(
        base_url="https://api.novita.ai/v3/openai",
        # Get the Novita AI API Key by referring to: https://novita.ai/docs/get-started/quickstart.html#_2-manage-api-key.
        api_key= api_key
    )

    model = "meta-llama/llama-3.1-8b-instruct"
    stream = False
    max_tokens = 10000

    chat_completion_res = client.chat.completions.create(
        model=model,
        messages=[
            {
                "role": "system",
                "content": prompt,
            },
            {
                "role": "user",
                "content": text,
            }
        ],
        stream=stream,
        max_tokens=max_tokens,
        top_p= 0.1,
        temperature= 0.1
    )
    return chat_completion_res.choices[0].message.content




 