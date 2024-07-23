import csv
import itertools
import json
import os
import numpy as np
import pandas as pd

base_path = '/Users/lgk/git/uni/web_test_generation/neural-embeddings-web-testing/'

APPS = ['addressbook', 'claroline', 'dimeshift', 'mantisbt', 'mrbs', 'pagekit', 'petclinic', 'phoenix', 'ppma']
OUTPUT_CSV = True
ADJUSTED_CW = True # if True, use the trained models with the adjusted class weights
setting = "within_apps" # within_apps or across_apps
filename = f'{base_path}BERT-SAF_csv_results_table/{"CWAdj-" if ADJUSTED_CW else ""}rq2-ALT-{setting}.csv'

if __name__ == '__main__':
    os.chdir("..")

    for feature in ['content_tags', 'content', 'tags']:
        prediction_column = f'{feature}-PREDICTION' # 'HUMAN_CLASSIFICATION'
        if ADJUSTED_CW and feature not in ['content']:
            print(f"Skipping {feature} for adjusted class weights | not trained yet.")
            continue

        if OUTPUT_CSV:
            # Create CSV file to store the results if it doesn't exist
            if not os.path.exists(filename):
                header = ['Setting', 'App', 'Feature', 'Precision', 'Recall', 'F1']
                with open(filename, 'w', encoding='UTF8') as f:
                    writer = csv.writer(f)
                    writer.writerow(header)

        for app in APPS:
            print(app)
            cluster_file_name = f'output/{app}.json'
            pred_file = f'{base_path}model_predictions_ss/{setting}/{"CWAdj-" if ADJUSTED_CW else ""}{app}.csv'
            predictions = pd.read_csv(pred_file)

            model = [] # list of states that are included in model
            covered_bins = []
            redundant_states = []
            number_of_bins = 0
            total_number_of_states = 0

            with open(cluster_file_name, 'r') as f:
                data = json.load(f)
                for bin in data:
                    number_of_bins += 1
                    bin_covered = False
                    for state in data[bin]:
                        total_number_of_states += 1
                        if model == []: # if model is empty, add the first state
                            model.append(state)
                            bin_covered = True
                        else:
                            is_distinct = True
                            for ms in model: # for each state already in the model
                                matching_row = predictions.loc[(predictions['state1'] == ms) & (predictions['state2'] == state)]
                                if matching_row.empty:
                                    matching_row = predictions.loc[(predictions['state1'] == state) & (predictions['state2'] == ms)]

                                if not matching_row.empty:
                                    if matching_row[prediction_column].values[0] == 1: # current state is clone/ND to a state in the model => do not add to model
                                        redundant_states.append(state)
                                        is_distinct = False
                                        break
                                else:
                                    print(f"Missing: {ms} - {state}")

                            if is_distinct:
                                model.append(state)
                                bin_covered = True

                    if bin_covered:
                        covered_bins.append(bin)

            # Calculate Precision, Recall, F1 Score
            unique_states_in_model = len(covered_bins)
            precision = unique_states_in_model / len(model) if len(model) > 0 else 0
            recall = len(covered_bins) / number_of_bins if number_of_bins > 0 else 0
            f1_score = 2 * (precision * recall) / (precision + recall) if precision + recall > 0 else 0

            print(f"App: {app}")
            print(f"Covered bins: {len(covered_bins)}")
            print(f"Redundant states: {len(redundant_states)}")
            print(f"Number of bins: {number_of_bins}")
            print(f"Total number of states: {total_number_of_states}")
            print(f"Number of states in model: {len(model)}")
            print(f"Unique states in model: {unique_states_in_model}")
            print(f"Precision: {precision:.4f}")
            print(f"Recall: {recall:.4f}")
            print(f"F1 Score: {f1_score:.4f}")

            # Write results to CSV if needed
            if OUTPUT_CSV:
                with open(filename, 'a', encoding='UTF8') as f:
                    writer = csv.writer(f)
                    writer.writerow([setting, app, feature, precision, recall, f1_score])
