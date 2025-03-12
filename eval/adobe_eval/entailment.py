import json
import os
import argparse
import torch
from utils.utils import *
from transformers import AutoTokenizer, AutoModelForSequenceClassification

torch.cuda.empty_cache()

# Load pre-trained model and tokenizer
model_name = "roberta-large-mnli"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name)

# Check if a GPU is available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# Batched NLI function using RoBERTa
def nli_roberta_batch(pairs):
    sentences1, sentences2 = zip(*pairs)
    inputs = tokenizer(list(sentences1), list(sentences2), return_tensors="pt", truncation=True, padding=True)
    inputs = {key: value.to(device) for key, value in inputs.items()}

    with torch.no_grad():
        outputs = model(**inputs)

    logits = outputs.logits
    probabilities = torch.softmax(logits, dim=1).cpu().tolist()
    entailment_scores = [prob[-1] for prob in probabilities]
    
    return entailment_scores

# Precision, Recall, and F1 calculation
def precision_recall_f1(predicted_statements, ground_truth_statements):
    N = len(predicted_statements)
    M = len(ground_truth_statements)

    predicted_to_ground_truth_pairs = [(pi, gj) for pi in predicted_statements for gj in ground_truth_statements]
    ground_truth_to_predicted_pairs = [(gj, pi) for gj in ground_truth_statements for pi in predicted_statements]

    precision_scores = []
    recall_scores = []

    if predicted_to_ground_truth_pairs:
        precision_entailment_scores = nli_roberta_batch(predicted_to_ground_truth_pairs)
        for i in range(N):
            max_score = max(precision_entailment_scores[i * M:(i + 1) * M])
            precision_scores.append(max_score)
        precision = sum(precision_scores) / N
    else:
        precision = 0

    if ground_truth_to_predicted_pairs:
        recall_entailment_scores = nli_roberta_batch(ground_truth_to_predicted_pairs)
        for j in range(M):
            max_score = max(recall_entailment_scores[j * N:(j + 1) * N])
            recall_scores.append(max_score)
        recall = sum(recall_scores) / M
    else:
        recall = 0

    f1_score = 2 * (precision * recall) / (precision + recall) if precision + recall > 0 else 0
    return precision, recall, f1_score

def parse_atomic_statements(state):
    if state == '':
        return ''
    lines = state.splitlines()
    idx_s = idx_e = 0

    for i, line in enumerate(lines):
        if line.strip() == 'Statements:':
            idx_s = i
        elif line.strip() == 'Rows:':
            idx_e = i
            break
    statements = lines[idx_s+1:idx_e-1]
    return [s[3:].strip() for s in statements]

def calculate_average_metrics(metrics_dict):
    total_precision, total_recall, total_f1 = 0.0, 0.0, 0.0
    count = len(metrics_dict)

    for metrics in metrics_dict.values():
        total_precision += metrics.get("Precision", 0.0)
        total_recall += metrics.get("Recall", 0.0)
        total_f1 += metrics.get("F1-score", 0.0)

    if count == 0:
        return {"Precision": None, "Recall": None, "F1-score": None}

    return {
        "Precision": total_precision / count,
        "Recall": total_recall / count,
        "F1-score": total_f1 / count
    }

full_length = 10
def run_evaluation(model_dir, exp_name):
    adobe_eval_output = {}
    cnt = 0 

    for idx in range(full_length):
        print('Idx:', idx)
        try:
            pred_file = read_file(file_path=f"model_outputs/Livesum/{model_dir}/{exp_name}/Unrolled_Statements/{idx}.txt")
            gold_file = read_file(file_path=f"model_outputs/Livesum/Unrolled_Statements/{idx}.txt")
        except FileNotFoundError:
            print(f"File not found for idx {idx}")
            continue

        gold_atomic = parse_atomic_statements(gold_file)
        pred_atomic = parse_atomic_statements(pred_file)

        if len(gold_atomic) != len(pred_atomic) or not gold_atomic or not pred_atomic:
            print(f"Statements not found for idx {idx}")   
            continue

        cnt += 1
        pp, pr, pf1 = precision_recall_f1(pred_atomic, gold_atomic)
        adobe_eval_output[idx] = {'Precision': pp, 'Recall': pr, 'F1-score': pf1}

    print(f"Total {cnt} samples evaluated")

    output_path = f"model_outputs/Livesum/{model_dir}/{exp_name}/"
    os.makedirs(output_path, exist_ok=True)

    with open(f"{output_path}/adobe_eval.json", 'w') as f:
        json.dump(adobe_eval_output, f, indent=6)
    
    with open(f"{output_path}/adobe_eval_collated.json", 'w') as f:
        json.dump(calculate_average_metrics(adobe_eval_output), f, indent=6)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate statement unrolling using RoBERTa.")
    parser.add_argument("--model_dir", type=str, required=True, help="Directory name of the model outputs.")
    parser.add_argument("--exp_name", type=str, required=True, help="Name of the experiment.")

    args = parser.parse_args()
    run_evaluation(args.model_dir, args.exp_name)
