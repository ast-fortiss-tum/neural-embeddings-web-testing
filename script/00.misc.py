import csv
import itertools
import json
import os
import pickle
import numpy as np
import pandas as pd
from natsort import natsorted

base_path = '/Users/lgk/git/uni/web_test_generation/neural-embeddings-web-testing/'
setting = "within_apps" # within_apps or across_apps

apps = ['addressbook', 'claroline', 'ppma', 'mrbs', 'mantisbt', 'dimeshift', 'pagekit', 'phoenix', 'petclinic']

pagekit_content = pd.read_csv(f'{base_path}model_predictions_ss/{setting}/pagekitContent.csv')

pagekit = pd.read_csv(f'{base_path}model_predictions_ss/{setting}/pagekit.csv')

c = 0
for index, row in pagekit.iterrows():
    matching_row = pagekit_content.loc[(pagekit_content['state1'] == row['state1']) & (pagekit_content['state2'] == row['state2'])]
    if not matching_row.empty:
        if matching_row['HUMAN_CLASSIFICATION'].values[0] != row['HUMAN_CLASSIFICATION']:
            c += 1
            pagekit.at[index, 'HUMAN_CLASSIFICATION'] = matching_row['HUMAN_CLASSIFICATION'].values[0]

pagekit.to_csv(f'{base_path}model_predictions_ss/{setting}/pagekit.csv', index=False)
print(f"Total: {c}")
