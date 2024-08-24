import csv
import itertools
import json
import os
import pickle

import numpy as np
import pandas as pd
from natsort import natsorted

base_path = '/Users/lgk/git/uni/web_test_generation/neural-embeddings-web-testing/'

APPS = ['addressbook', 'claroline', 'ppma', 'mrbs', 'mantisbt', 'dimeshift', 'pagekit', 'phoenix', 'petclinic']

OUTPUT_CSV = True
ADJUSTED_CW = True # if True, use the trained models with the adjusted class weights

setting = "within_apps" # within_apps or across_apps
filename = f'{base_path}0-BERT-SAF_csv_results_table/{"CWAdj-" if ADJUSTED_CW else ""}rq2-{setting}.csv'

if __name__ == '__main__':
    os.chdir("..")

    # APPS = ['pagekit']

    for feature in ['content_tags', 'content', 'tags']:
        if ADJUSTED_CW and feature not in ['content']:
            print(f"Skipping {feature} for adjusted class weights | not trained yet.")
            continue

        if OUTPUT_CSV:
            # create csv file to store the results
            if not os.path.exists(filename):
                header = ['Setting', 'App', 'Feature', 'Precision', 'Recall', 'F1']
                with open(filename, 'w', encoding='UTF8') as f:
                    writer = csv.writer(f)
                    # write the header
                    writer.writerow(header)

        for app in APPS:
            print(app)
            pred_file = f'{base_path}model_predictions_ss/{setting}/{"CWAdj-" if ADJUSTED_CW else ""}{app}.csv'
            predictions = pd.read_csv(pred_file)

            # predictions = predictions[['state1', 'state2', 'HUMAN_CLASSIFICATION']]
            predictions = predictions[['state1', 'state2', f'{feature}-PREDICTION']]

            # Convert predictions to a list of tuples
            tuples = [tuple(x) for x in predictions.to_numpy()]

            # Extract unique states and sort them
            items = natsorted(set.union(set([item[0] for item in tuples]), set([item[1] for item in tuples])))

            # Create a mapping of items to indices
            value = dict(zip(items, range(len(items))))
            dist_matrix = np.zeros((len(items), len(items)))

            # Fill the distance matrix using the predictions
            for tup in tuples:
                state1, state2, pred = tup
                dist_matrix[value[state1], value[state2]] = pred
                dist_matrix[value[state2], value[state1]] = pred

            # Convert the distance matrix to a DataFrame and save it
            new_ss = pd.DataFrame(dist_matrix, columns=items, index=items)
            new_ss.to_csv(f'{base_path}script/SS_as_distance_matrix.csv')

            # Build the dictionary of clones from the distance matrix | 'stateX' -> [clones]
            dictionary = {}
            for index, row in new_ss.iterrows():
                clones = []
                sel = new_ss.loc[new_ss[index] == 1.0]
                clones.append(sel[index].keys().tolist())
                dictionary[index] = clones

            # Load ground truth data
            with open(f'{base_path}output/' + app + '.json', 'r') as f:
                data = json.load(f)

            # Initialize counters
            number_in_common = 0
            number_gt = 0
            # number_d2v = len(dictionary.keys())  # Number of unique items with predictions
            number_d2v = 0 # this is not really making a huge difference for most apps as the number of unique items is not that high in comparison to the number of pairs

            # Compare model predictions to ground truth clusters
            for cluster in data:
                value = data[cluster]
                key = value[0]  # I treat the first item of the cluster as key
                value.remove(key)

                if len(value) == 0:  # empty cluster
                    continue

                # Generate all possible pairs of items in the cluster, not including the key???
                pairs = list(itertools.combinations(value, 2))

                number_gt += len(pairs)
                # Count the number of pairs in the ground truth (GT) clusters.

                for pair in pairs:
                    state1 = pair[0]
                    state2 = pair[1]

                    # For each pair of items in the cluster:
                    # print(f"Clones for {state1}: {dictionary.get(state1, [[]])[0]}")
                    if state2 in dictionary.get(state1, [[]])[0]:
                        number_in_common += 1
                    # else:
                    #     print(f"Pair {state1} - {state2} not in common for  {cluster}")

                    number_d2v += 1

            # Calculate precision, recall, and F1 score
            precision = number_in_common / number_d2v if number_d2v else 0 #  ratio of unique states (bins) covered by the model to the total number of states in the model (NDD2020)
            recall = number_in_common / number_gt if number_gt else 0 # the number of bins covered by the model to the total number of bins identified by humans (NDD2020)
            f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) else 0

            # Print the results
            # print(f"number of pairs in ground truth: {number_gt}")
            # print(f"number of pairs in common: {number_in_common}")
            # print(f"number of pairs: {number_d2v}")

            # print(f"precision: {precision:.2f}")
            # print(f"recall: {recall:.2f}")
            # print(f"f1: {f1:.2f}")

            # Write results to CSV if needed
            if OUTPUT_CSV:
                comparison_df = pd.read_csv(filename) if os.path.exists(filename) else pd.DataFrame(columns=['Setting', 'App', 'Feature', 'Classifier', 'Precision', 'Recall', 'F1'])
                d1 = pd.DataFrame({
                    'Setting': setting,
                    'App': app,
                    'Feature': feature,
                    'Precision': [precision],
                    'Recall': [recall],
                    'F1': [f1]
                })
                comparison_df = pd.concat([comparison_df, d1], ignore_index=True)
                comparison_df.to_csv(filename, index=False)
