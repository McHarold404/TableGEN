import numpy as np
import bert_score
from sacrebleu import sentence_chrf
import torch

def evaluate_table(model_output, gold_label):
    """
    Evaluate the model output table against the gold label table.
    """
    # Initialize BERTScorer only once
    bert_scorer = bert_score.BERTScorer(model_type='roberta-large', lang='en', rescale_with_baseline=True)
    metric_cache = dict()  # cache similarities to avoid redundant computations

    # Helper function to convert data types to standard Python strings
    def to_python_str(s):
        if isinstance(s, np.ndarray):
            # Flatten array and join elements into a string
            return ' '.join(map(str, s.flatten()))
        elif isinstance(s, (np.str_, np.generic)):
            return s.item()
        else:
            return str(s)

    # Helper function to calculate similarity between two items
    def calc_similarity(tgt, pred, metric):
        # Convert to strings
        tgt_str = to_python_str(tgt)
        pred_str = to_python_str(pred)

        # Use string representations as cache keys
        cache_key = (tgt_str, pred_str, metric)
        if cache_key in metric_cache:
            return metric_cache[cache_key]

        # Debugging: Print types and values
        # print(f"tgt_str type: {type(tgt_str)}, value: {tgt_str}")
        # print(f"pred_str type: {type(pred_str)}, value: {pred_str}")

        if metric == 'exact_match':
            sim = float(tgt_str.strip() == pred_str.strip())
        elif metric == 'chrf':
            sim = sentence_chrf(pred_str, [tgt_str]).score / 100  # chrF score ranges from 0 to 1
        elif metric == 'BERT_score':
            #print(f"Calculating BERT_score between '{pred_str}' and '{tgt_str}'")
            P, R, F1 = bert_scorer.score([pred_str], [tgt_str])
            sim = F1.item()  # Scaled BERTScore
        else:
            raise ValueError(f"Unknown metric: {metric}")
        metric_cache[cache_key] = sim
        return sim

    # Function to extract data from the table
    def parse_table_to_data(table):
        """
        Extract row headers, column headers, and relations from the table.
        """
        #print(table)
        num_rows, num_cols = table.shape
        print("Rows:",num_rows,"Cols:",num_cols)
        row_headers = set(to_python_str(table[i, 0]) for i in range(1, num_rows)) if num_rows > 1 else set()
        col_headers = set(to_python_str(table[0, j]) for j in range(1, num_cols)) if num_cols > 1 else set()
        relations = set()
        for i in range(1, num_rows):
            for j in range(1, num_cols):
                cell_value = table[i, j]
                if cell_value != '':
                    row_header = to_python_str(table[i, 0]) if num_cols > 0 else ''
                    col_header = to_python_str(table[0, j]) if num_rows > 0 else ''
                    relations.add((row_header, col_header, to_python_str(cell_value)))
        return row_headers, col_headers, relations

    # Extract data from model_output and gold_label tables
    gold_row_headers, gold_col_headers, gold_relations = parse_table_to_data(gold_label)
    model_row_headers, model_col_headers, model_relations = parse_table_to_data(model_output)

    # Initialize metrics dictionary
    metrics = {}

    # List of metrics to compute
    metric_names = ['exact_match', 'BERT_score', 'chrf']

    # Evaluate cells (relations)
    cells_metrics = {}
    for metric_name in metric_names:
        precision, recall, f1 = metrics_by_sim(
            gold_relations, model_relations, metric_name, calc_similarity
        )
        cells_metrics[metric_name + '(%)'] = {
            'precision': precision * 100,
            'recall': recall * 100,
            'f1': f1 * 100,
        }
    metrics['cells'] = cells_metrics

    # Evaluate row headers
    row_header_metrics = {}
    for metric_name in metric_names:
        precision, recall, f1 = metrics_by_sim(
            gold_row_headers, model_row_headers, metric_name, calc_similarity
        )
        row_header_metrics[metric_name + '(%)'] = {
            'precision': precision * 100,
            'recall': recall * 100,
            'f1': f1 * 100,
        }
    metrics['row_header'] = row_header_metrics

    # Evaluate column headers
    col_header_metrics = {}
    for metric_name in metric_names:
        precision, recall, f1 = metrics_by_sim(
            gold_col_headers, model_col_headers, metric_name, calc_similarity
        )
        col_header_metrics[metric_name + '(%)'] = {
            'precision': precision * 100,
            'recall': recall * 100,
            'f1': f1 * 100,
        }
    metrics['col_header'] = col_header_metrics

    return metrics

def metrics_by_sim(tgt_data, pred_data, metric_name, calc_similarity_func):
    """
    Compute precision, recall, and F1 score based on similarity.
    """
    if not tgt_data and not pred_data:
        return 1.0, 1.0, 1.0  # Both are empty, perfect match
    if not pred_data:
        return 0.0, 0.0, 0.0  # No predictions
    if not tgt_data:
        return 0.0, 0.0, 0.0  # No target data

    # Build similarity matrix
    sim_matrix = np.zeros((len(tgt_data), len(pred_data)), dtype=float)
    for i, tgt_item in enumerate(tgt_data):
        for j, pred_item in enumerate(pred_data):
            sim = calc_similarity_func(tgt_item, pred_item, metric_name)
            sim_matrix[i, j] = sim

    # Precision: average of max similarities for each predicted item
    max_sim_pred = np.max(sim_matrix, axis=0)
    precision = np.mean(max_sim_pred)

    # Recall: average of max similarities for each target item
    max_sim_tgt = np.max(sim_matrix, axis=1)
    recall = np.mean(max_sim_tgt)

    if precision + recall == 0:
        f1 = 0.0
    else:
        f1 = 2 * precision * recall / (precision + recall)

    return precision, recall, f1



# Function to calculate the macro average for specific entity and subkey (cells, row_header, col_header)
def calculate_macro_avg_per_category(results, category):
    metrics = ['exact_match(%)', 'BERT_score(%)', 'chrf(%)']
    totals = {metric: {'precision': 0.0, 'recall': 0.0, 'f1': 0.0} for metric in metrics}
    counts = {metric: 0 for metric in metrics}
    
    # Loop through each test point
    for test_key, test_val in results.items():
        for entity, entity_val in test_val.items():
            if category in entity_val:  # Only process if the specified category exists
                for metric in metrics:
                    if metric in entity_val[category]:  # Only consider the available metrics
                        totals[metric]['precision'] += entity_val[category][metric]['precision']
                        totals[metric]['recall'] += entity_val[category][metric]['recall']
                        totals[metric]['f1'] += entity_val[category][metric]['f1']
                        counts[metric] += 1

    # Calculate the macro average by dividing the sum by the count
    macro_avg = {metric: {key: totals[metric][key] / counts[metric] for key in totals[metric]} for metric in metrics}
    return macro_avg

# Function to calculate macro averages for cells, row_header, and col_header for the entire dictionary
def calculate_all_macro_avgs(results):
    cells_avg = calculate_macro_avg_per_category(results, 'cells')
    row_header_avg = calculate_macro_avg_per_category(results, 'row_header')
    col_header_avg = calculate_macro_avg_per_category(results, 'col_header')

    # Return the results as a dictionary
    return {
        'Cells': cells_avg,
        'Row Header': row_header_avg,
        'Column Header': col_header_avg
    }

