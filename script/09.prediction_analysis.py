import csv
import itertools
import json
import os
import numpy as np
import pandas as pd

APPS = ['addressbook', 'claroline', 'ppma', 'mrbs', 'mantisbt', 'dimeshift', 'pagekit', 'phoenix', 'petclinic']
FEATURES = ['content_tags', 'content', 'tags']
base_path = '/Users/lgk/git/uni/web_test_generation/neural-embeddings-web-testing/'
setting = "across_apps"  # within_apps or across_apps
csv_output = f'{base_path}BERT-SAF_csv_results_table/model_predictions-{setting}.csv'

results = []

for app in APPS:
    pred_file = f'{base_path}model_predictions_ss/{setting}/{app}.csv'
    predictions = pd.read_csv(pred_file)

    for feature in FEATURES:
        # counters
        true_positives = 0
        false_positives = 0
        false_negatives = 0
        true_negatives = 0

        for i in range(len(predictions)):
            if predictions.loc[i, f'{feature}-PREDICTION'] == 1 and predictions.loc[i, 'HUMAN_CLASSIFICATION'] == 0:
                false_positives += 1
            elif predictions.loc[i, f'{feature}-PREDICTION'] == 1 and predictions.loc[i, 'HUMAN_CLASSIFICATION'] == 1:
                true_positives += 1
            elif predictions.loc[i, f'{feature}-PREDICTION'] == 0 and predictions.loc[i, 'HUMAN_CLASSIFICATION'] == 0:
                true_negatives += 1
            elif predictions.loc[i, f'{feature}-PREDICTION'] == 0 and predictions.loc[i, 'HUMAN_CLASSIFICATION'] == 1:
                false_negatives += 1

        # Calculate metrics
        accuracy = (true_positives + true_negatives) / (true_positives + true_negatives + false_positives + false_negatives)
        precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0
        recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0
        f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

        results.append({
            'App': app,
            'Feature': feature,
            'Accuracy': accuracy,
            'Precision': precision,
            'Recall': recall,
            'F1-Score': f1_score,
            'True Positives': true_positives,
            'False Positives': false_positives,
            'True Negatives': true_negatives,
            'False Negatives': false_negatives
        })

# Create DataFrame from results
df_results = pd.DataFrame(results)

# Save to CSV
df_results.to_csv(csv_output, index=False)

print(f"Results saved to {csv_output}")
