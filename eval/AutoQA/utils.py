import re
import pandas as pd
import numpy as np


def extract_questions_answers(text):
    # Define regex to capture both formats: with and without spaces between "Question"/"Answer" and numbers
    qa_pattern = re.compile(
        r'\[Question\s?(\d+):\s*(.*?)\]\s*\[Answer\s?(\d+):\s*(.*?)\]', re.DOTALL
    )
    
    questions, answers = [], []
    
    # Find all matches in the text
    matches = qa_pattern.findall(text)
    
    for match in matches:
        question_num, question_text, answer_num, answer_text = match
        # Ensure that question and answer numbers match
        if question_num == answer_num:
            questions.append(question_text.strip())
            answers.append(answer_text.strip())
    
    return list(zip(questions, answers))