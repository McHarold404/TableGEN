import os
import google.generativeai as genai
import time 
from datetime import datetime
import json
import os
import json
from datetime import datetime
import time  # For rate limiting
from dotenv import load_dotenv
from dotenv import *
import google.generativeai as genai  # Assumed to be the client library for Gemini AI

class GeminiBot:
    def __init__(self, api_key=None, model_name="gemini-1.5-flash", 
                 data_path=None, prompt_path=None, meta_data=None, file_name=None, 
                 limit_rows=None, output_path=None, start_line=None, end_line=None, data_string=None):
        self.api_key = api_key or os.getenv("GEMINI_KEY")
        if not self.api_key:
            raise ValueError("API key is required. Set it in the constructor or as an environment variable 'GEMINI_KEY'")
        genai.configure(api_key=self.api_key)
        self.model_name = model_name
        self.data_path = data_path
        self.prompt_path = prompt_path
        self.meta_data = meta_data
        self.file_name = file_name
        self.limit_rows = limit_rows
        self.output_path = output_path
        self.start_line = start_line
        self.end_line = end_line
        self.data_string = data_string

        # Output folder configuration
        os.makedirs(output_path, exist_ok=True)
        self.output_dir = output_path
        self.output_path = os.path.join(output_path, file_name)

        self.generation_config = {
            "temperature": 1,
            "top_p": 0.95,
            "top_k": 64,
            "max_output_tokens": 8192,
            "response_mime_type": "text/plain",
        }
        self.model = self._create_model()
        self.chat_session = self.model.start_chat(history=[])
    
    def _create_model(self):
        return genai.GenerativeModel(
            model_name=self.model_name,
            system_instruction= self.load_prompt,
            generation_config=self.generation_config,
        )
    
    def get_response(self, user_message):
        try:
            response = self.chat_session.send_message(user_message)
            return response.text
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
            print("No meta data provided.")
            return []
        try:
            with open(self.meta_data, "r") as json_file:
                meta = json.load(json_file)
            return [x['response'] for x in meta]
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
            if i % 15 == 14:
                time.sleep(60)  # Pause to avoid rate limiting

            #message = f"{prompt}:\n{point}"
            response = self.get_response(point)
            results.append({'data_point': i, 'input': point, 'response': response})
            print(f"Processed data point {i}")

        # Store prompts
        try:
            prompt_filename = os.path.basename(self.prompt_path)
            with open(os.path.join(self.output_dir, prompt_filename), "w") as f:
                f.write(prompt)
        except Exception as e:
            print(f"Error saving prompt: {str(e)}")

        # Store outputs
        try:
            with open(self.output_path, "w") as json_file:
                json.dump(results, json_file, indent=2)
        except Exception as e:
            print(f"Error saving results: {str(e)}")

        print("Finished inference")
        print("Stored prompts at", os.path.join(self.output_dir, prompt_filename))
        print("Model output stored at", self.output_path)
        print("---------------------")

        return results


def ask_gemini(text : str, prompt_path : str =None, key = 1,model_name = "gemini-1.5-flash"):
    
    # Import necessary module for GenerativeModel if not already done
    load_dotenv()
    if key == 1:
        api_key = os.getenv("GEMINI_KEY")
    elif key == 2:
        api_key = os.getenv("GEMINI_KEY_2")
    elif key == 3:
        api_key = os.getenv("GEMINI_KEY_3")
    elif key == 4:
        api_key = os.getenv("GEMINI_KEY_4")
    elif key == 5:
        api_key = os.getenv("GEMINI_KEY_5")
    elif key == 6:
        api_key = os.getenv("GEMINI_KEY_6")
    elif key == 7:
        api_key = os.getenv("GEMINI_KEY_7")
    elif key == 8:
        api_key = os.getenv("GEMINI_KEY_8")
    elif key == 9:
        api_key = os.getenv("GEMINI_KEY_9")
    elif key == 10:
        api_key = os.getenv("GEMINI_KEY_10")
    elif key == 11:
        api_key = os.getenv("GEMINI_KEY_11")
    elif key == 12:
        api_key = os.getenv("GEMINI_KEY_12")
    elif key == 13:
        api_key = os.getenv("GEMINI_KEY_13")
    elif key == 14:
        api_key = os.getenv("GEMINI_KEY_14")
    else:
        raise ValueError("No key specified")
    # Check if a prompt path is provided and read prompt text
    if prompt_path and text:
        with open(prompt_path, 'r') as file:
            prompt = file.read().strip()
    else:
        return "Error: no data given"

    # Initialize the GenerativeModel with the required configuration
    generation_config = {
        "temperature": 0,
        "top_p": 1.0,
        "response_mime_type": "text/plain",
    }
    genai.configure(api_key=api_key)
    try:
        # Instantiate the model
        model = genai.GenerativeModel(
            model_name=model_name,  # replace with actual model name
            # The `generation_config` dictionary in the GeminiBot class and the ask_gemini function is
            # used to configure the generation settings for the GenerativeModel. It contains the
            # following key-value pairs:
            generation_config=generation_config,
            system_instruction= prompt
        )
        
        # Start a chat session
        chat_session = model.start_chat(history=[])
        # Get the response by sending the prompt
        response = chat_session.send_message(text)
        return response.text

    except Exception as e:
        return f"An error occurred: {str(e)}"
