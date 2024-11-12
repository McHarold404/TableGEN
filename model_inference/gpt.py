import os
from openai import OpenAI
from openai import OpenAI
import json
from datetime import datetime
from dotenv import load_dotenv
from dotenv import *


class GPT4MiniBot:
    def __init__(self, api_key="", model="gpt-4o-mini", 
                 data_path=None, prompt_path=None, output_path=None, file_name=None, 
                 meta_data=None, limit_rows=None, start_line=None, end_line=None, data_string=None):
        load_dotenv()
        self.client = OpenAI(api_key = os.getenv("OPENAI_API_KEY"))
        #self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        #if not self.api_key:
        #    raise ValueError("API key is required. Set it in the constructor or as an environment variable 'OPENAI_API_KEY'")
        self.model = model
        self.limit_rows = limit_rows
        self.start_line = start_line
        self.end_line = end_line
        self.data_string = data_string
        self.meta_data = meta_data

        # Output folder configuration
        try:
            os.makedirs(output_path, exist_ok=True)
            self.output_dir = output_path
            self.output_path = os.path.join(output_path, file_name)
        except:
            pass
        self.data_path = data_path
        self.prompt_path = prompt_path

    def get_api_response(self, prompt = None, message= None):
        try:
            response = self.client.chat.completions.create(model=self.model,
            messages=[
                {'role' : "system" , "content" : prompt},
                {'role': "user" , "content": message},
                ],
            temperature=0.1,
            top_p=0.1
            )
            return response.choices[0].message.content
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
                print('Data file not found, using provided data string.')
                return [self.data_string]
            else:
                print(f"Data file not found at {self.data_path}")
                return []
        except Exception as e:
            print(f"Error reading data file: {str(e)}")
            return []

    def load_meta_data(self):
        if self.meta_data is None:
            print(f"No meta data provided.")
            return []
        try:
            with open(self.meta_data, "r") as json_file:
                meta = json.load(json_file)
            return [x['response'].strip('\n') for x in meta]
        except Exception as e:
            print(f"Error reading meta data file: {str(e)}")
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
            assert len(meta) == len(data), "Meta data and data length mismatch."
            for i in range(len(data)):
                final_data.append(f"{meta[i]}:\n{data[i]}")

        for i, point in enumerate(final_data):
            if self.start_line is not None and i < self.start_line:
                continue
            if self.end_line is not None and i > self.end_line:
                break
            if self.limit_rows is not None and i >= self.limit_rows:
                break

            response = self.get_api_response(prompt =prompt,message=point)
            results.append({'data_point': i, 'input': point, 'response': response})
            print(f"Processed data point {i}")

        # Store outputs
        try:
            with open(self.output_path, "w") as json_file:
                json.dump(results, json_file, indent=2)
            print("Model output stored at", self.output_path)

        except Exception as e:
            print(f"Error saving results: {str(e)}")

        # Store prompts
        try:
            prompt_filename = os.path.basename(self.prompt_path)
            with open(os.path.join(self.output_dir, prompt_filename), "w") as f:
                f.write(prompt)
            print("Stored prompts at", os.path.join(self.output_dir, prompt_filename))
        except Exception as e:
            print(f"Error saving prompt: {str(e)}")

        print("Finished inference")
        print("----------'")
        return results


def ask_chatgpt(text: str, prompt_path=None):
    # Check if a prompt path is provided and read prompt text
    load_dotenv()
    api_key =os.getenv("OPENAI_API_KEY")
    client = OpenAI(api_key = api_key)
    if prompt_path and text:
        with open(prompt_path, 'r') as file:
            prompt = file.read().strip()
    else:
        return "Error: no data given"

    # Model configuration - replace 'gpt-4' with the specific model if needed
    model_name = "gpt-4o-mini"  # or "gpt-3.5-turbo" if you want a different model
    # Send the prompt to the model
    response = client.chat.completions.create(
    model=model_name,
    messages=[
        {'role' : "system" , "content" : prompt },
        {'role': "user" , "content": text}
    ],
    temperature=0.1,
    top_p=0.1
    )
        
    # Return the response content
    return response.choices[0].message.content
