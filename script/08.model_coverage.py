import csv
import glob
import itertools
import json
import os
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm


def compute_gs_models():
    recap_data = []
    recap_headers = ['appname', 'number_of_bins', 'number_of_states', 'number_of_singleton_bins',
                     'number_of_bins_with_two_states', 'number_of_bins_with_more_than_two_states']

    # iterate over all the gold-standard files
    for filepath in glob.iglob('raw_data/*.json'):

        appname = Path(filepath).stem  # filename with no extension
        with open(filepath, 'r') as file:
            data = json.load(file)
            gold_standard = {}  # start with empty dict
            number_of_states = 0

            # iterate over each state in the gold-standard data
            for state_name, state_data in data['states'].items():
                number_of_states = number_of_states + 1
                if not state_data['bin'] in gold_standard.keys():  # it's the first time the current bin is seen
                    gold_standard[state_data['bin']] = []  # initialize to empty list
                gold_standard[state_data['bin']].append(state_name)  # and then append current state

            with open(f'./output/{appname}.json', 'w+') as output:
                json.dump(gold_standard, output)

            # compute some per-app statistics
            number_of_states = 0
            number_of_singleton_bins = 0
            number_of_bins_with_two_states = 0
            number_of_bins_with_more_than_two_states = 0

            for bin, states in gold_standard.items():
                number_of_states = number_of_states + len(states)
                if len(states) == 1:
                    number_of_singleton_bins = number_of_singleton_bins + 1
                elif len(states) == 2:
                    number_of_bins_with_two_states = number_of_bins_with_two_states + 1
                else:
                    number_of_bins_with_more_than_two_states = number_of_bins_with_more_than_two_states + 1

            recap_row = [
                appname,  # app name
                len(gold_standard.keys()),  # number of bins
                number_of_states,  # number of states
                number_of_singleton_bins,
                number_of_bins_with_two_states,
                number_of_bins_with_more_than_two_states
            ]

        recap_data.append(recap_row)

    with open('output/recap.csv', 'w+') as output:
        writer = csv.writer(output)
        writer.writerow(recap_headers)
        writer.writerows(recap_data)


def is_clone(application, state1, state2, feature, setting, classifier):
    df = pd.read_csv('script/SS_threshold_set.csv')
    df = df.query("appname == @application")

    df = df[(df['state1'] == state1) & (df['state2'] == state2)]
    if feature == 'dom-rted':
        distance = df['DOM_RTED'].tolist()
    elif feature == 'visual-hyst':
        distance = df['VISUAL_Hyst'].tolist()
    else:
        distance = df[feature.replace('-', '_')].tolist()
    distance = np.array(distance).reshape(1, -1)

    # TODO: THIS PART IS DUPLICATE with main.py, refactor
    CLASSIFIER_PATH = 'trained_classifiers/'
    SETTING = setting
    CLASSIFIER = classifier
    EXT = '.sav'

    CLASSIFIER_USED = CLASSIFIER_PATH + SETTING + CLASSIFIER + feature.replace('_', '-') + EXT

    # print('DOC2VEC FEATURE = %s' % FEATURE)
    # print('CLASSIFIER = %s' % CLASSIFIER_USED)

    model = None
    try:
        model = pickle.load(open(CLASSIFIER_USED, 'rb'))
    except FileNotFoundError:
        print("Cannot find classifier %s" % CLASSIFIER_USED)
        exit()
    try:
        prediction = model.predict(distance)  # 0 = near-duplicates, 1 = distinct
    except ValueError:
        prediction = [0]

    if prediction == [0]:
        return True
    else:
        # print("states %s and %s are not clones, distance %.2f" % (state1, state2, distance))
        return False


APPS = ['addressbook', 'claroline', 'dimeshift', 'mantisbt', 'mrbs', 'pagekit', 'petclinic', 'phoenix', 'ppma']
# APPS = ['addressbook']

FEATURES = ['doc2vec-distance-tags', 'doc2vec-distance-content', 'doc2vec-distance-content_tags', 'doc2vec-distance-all',
            'dom-rted', 'visual-hyst']

SETTINGS = ["beyond-apps-"]

CLASSIFIERS = ['svm-rbf-', 'xgboost-']

if __name__ == '__main__':
    os.chdir("..")
    # compute_gs_models()

    OUTPUT_CSV = True
    filename = 'csv_results_table/rq3-beyond-apps.csv'

    if OUTPUT_CSV:
        # create csv file to store the results
        if not os.path.exists(filename):
            header = ['Setting', 'App', 'Feature', 'Classifier', 'Precision', 'Recall', 'F1']
            with open(filename, 'w', encoding='UTF8') as f:
                writer = csv.writer(f)
                # write the header
                writer.writerow(header)

    for setting in SETTINGS:
        comparison_df = None
        for app in APPS:
            print('app: %s' % app)
            for classifier in CLASSIFIERS:
                for feature in FEATURES:
                    print('feature: %s' % feature)

                    with open('output/' + app + '.json', 'r') as f:
                        data = json.load(f)

                    number_intra_pairs_gt = 0
                    number_intra_pairs_doc2vec = 0
                    number_intra_pairs_in_common = 0

                    for cluster in tqdm(data):
                        value = data[cluster]
                        if len(value) != 1:
                            pairs = list(itertools.combinations(value, 2))
                            number_intra_pairs_gt += len(pairs)
                            # print('cluster: %s\tpairs: %d\ttotal pairs: %d'
                            # % (cluster, len(pairs), number_intra_pairs_gt))
                            for pair in tqdm(pairs):
                                state1 = pair[0]
                                state2 = pair[1]
                                if is_clone(app, state1, state2, feature, setting, classifier):
                                    number_intra_pairs_doc2vec += 1
                                    number_intra_pairs_in_common += 1
                                else:
                                    number_intra_pairs_doc2vec += 1

                    # print("number_intra_pairs_in_common: %d" % number_intra_pairs_in_common)
                    # print("number_intra_pairs_doc2vec: %d" % number_intra_pairs_doc2vec)
                    # print("number_intra_pairs_gt: %d" % number_intra_pairs_gt)

                    precision = (number_intra_pairs_in_common / number_intra_pairs_doc2vec)
                    recall = (number_intra_pairs_in_common / number_intra_pairs_gt)
                    f1 = (2 * ((precision * recall) / (precision + recall)))
                    print("precision: %.2f" % precision)
                    print("recall: %.2f" % recall)
                    print("f1: %.2f" % f1)

                    if OUTPUT_CSV:
                        d1 = pd.DataFrame(
                            {'Setting': setting,
                             'App': app,
                             'Feature': feature,
                             'Classifier': classifier,
                             'Precision': [precision],
                             'Recall': [recall],
                             'F1': [f1]})

                        comparison_df = pd.concat([comparison_df, d1])

                comparison_df.to_csv(filename, index=False)
