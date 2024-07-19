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
app = 'pagekit'

newPreds = pd.read_csv(f'{base_path}model_predictions_ss/{setting}/{app}_NEW.csv')
org = pd.read_csv(f'{base_path}model_predictions_ss/{setting}/{app}.csv')

for i in range(len(org)):
    # find matching row based on state1 and state2
    mask = (newPreds['state1'] == org.loc[i, 'state1']) & (newPreds['state2'] == org.loc[i, 'state2'])

    # Check if there's exactly one matching row
    if mask.sum() == 1:
        if org.loc[i, 'content-PREDICTION'] != newPreds.loc[mask, 'PREDICTION'].iloc[0]:
            org.loc[i, 'content-PREDICTION'] = newPreds.loc[mask, 'PREDICTION'].iloc[0]

    elif mask.sum() > 1:
        print(f"Multiple matches found for row {i}. Please check your data.")
    else:
        print(f"No match found for row {i}. Please check your data.")

org.to_csv(f'{base_path}model_predictions_ss/{setting}/{app}.csv', index=False)
