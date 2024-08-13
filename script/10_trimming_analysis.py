import csv
import json
import os
import pandas as pd
from transformers import AutoTokenizer
from abstract_function_python import utils

base_path = '/Users/lgk/Documents/uni/BA-Local/Data/WebEmbed-97k-state-pairs/'
APPS = ['addressbook', 'claroline', 'ppma', 'mrbs', 'mantisbt', 'dimeshift', 'pagekit', 'phoenix', 'petclinic']


def token_analysis(representation='content_tags'):
    tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")
    app_token_lengths = {}

    for app in APPS:
        path_to_folder = f'{base_path}/{app}/'
        total_tokens = 0
        file_count = 0

        for file in os.listdir(path_to_folder):
            if file.endswith(f'.{representation}'):
                with open(f'{path_to_folder}{file}', 'r') as f:
                    state = ' '.join(json.loads(f.read().strip()))
                    tokens = tokenizer.tokenize(state)
                    token_count = len(tokens)
                    total_tokens += token_count
                    file_count += 1

                    # write back the tokenized state into a .tokens file
                    with open(f'{path_to_folder}{file}.tokens', 'w') as f:
                        f.write(json.dumps(tokens))

        # Calculate and store the average token length per app
        if file_count > 0:
            avg_token_length = total_tokens / file_count
        else:
            avg_token_length = 0

        app_token_lengths[app] = avg_token_length
        # print(f'App: {app}, Avg Token Length: {avg_token_length}')
        print(avg_token_length)

    return app_token_lengths

def trimming_token_analysis(representation='content_tags', trimming_approach = 'char_by_char'):
    if representation not in ['content_tags', 'content', 'tags']:
        print(f'Invalid representation type: {representation}')
        return
    if trimming_approach not in ['char_by_char', 'body_only']:
        print(f'Invalid trimming approach: {trimming_approach}')
        return

    tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")
    ss_path = '/Users/lgk/Documents/uni/BA-Local/Data/SS.csv'
    ss = pd.read_csv(ss_path)

    app_token_lengths = {}

    for app in APPS:
        app_ss = ss[ss['appname'] == app]
        path_to_folder = os.path.join(base_path, app)

        total_files = 0
        total_trimmed_tokens = 0

        for index, row in app_ss.iterrows():
            state1 = row['state1']
            state2 = row['state2']

            file1_path = os.path.join(path_to_folder, f'{state1}.html.{representation}')
            file2_path = os.path.join(path_to_folder, f'{state2}.html.{representation}')

            if os.path.exists(file1_path) and os.path.exists(file2_path):
                # Read and process state1
                with open(file1_path, 'r') as f:
                    state1_representation = json.loads(f.read().strip())

                # Read and process state2
                with open(file2_path, 'r') as f:
                    state2_representation = json.loads(f.read().strip())

                trimmed_state1, trimmed_state2 = "", ""

                if trimming_approach == 'char_by_char':
                    # Trim the common parts using the trim_common_html function
                    trimmed_state1, trimmed_state2 = utils.trim_common_html(' '.join(state1_representation), ' '.join(state2_representation))

                elif trimming_approach == 'body_only':
                    trimmed_state1 = ' '.join(utils.trim_content_to_body(state1_representation, representation))
                    trimmed_state2 = ' '.join(utils.trim_content_to_body(state2_representation, representation))

                # Tokenize the trimmed states
                tokens_state1 = tokenizer.tokenize(trimmed_state1)
                tokens_state2 = tokenizer.tokenize(trimmed_state2)

                token_count_state1 = len(tokens_state1)
                token_count_state2 = len(tokens_state2)

                # Update totals
                total_files += 2  # Since we're processing two files (state1 and state2)
                total_trimmed_tokens += token_count_state1 + token_count_state2

        # Calculate the average number of tokens per app
        if total_files > 0:
            avg_tokens_per_file = total_trimmed_tokens / total_files
        else:
            avg_tokens_per_file = 0

        app_token_lengths[app] = avg_tokens_per_file
        print(avg_tokens_per_file)

    return app_token_lengths

def trim_analysis_body_only(representation='content_tags'):
    if representation not in ['content_tags']: #, 'content', 'tags'
        print(f'not yet implemented: {representation}')
        return
    for app in APPS:
        # print(app)
        path_to_folder = f'{base_path}/{app}/'

        total_files = 0
        total_length = 0
        total_trimmed_length = 0

        for file in os.listdir(path_to_folder):
            if file.endswith(f'.{representation}'):
                with open(f'{path_to_folder}{file}', 'r') as f:
                    content_tags = json.loads(f.read().strip())
                    length = len(content_tags)

                    trimmed_length = len(utils.trim_content_to_body(content_tags, representation='content_tags'))

                    total_files += 1
                    total_length += length
                    total_trimmed_length += trimmed_length

        # Calculate the average splitting ratio if there are files processed
        if total_files > 0:
            avg_length = total_length / total_files
            avg_trimmed_length = total_trimmed_length / total_files
            trimming_ratio = avg_trimmed_length / avg_length
            # print(f'Avg size of {app} after trimming: {trimming_ratio:.4f}')
            print(f'{trimming_ratio:.3f}')
        else:
            print(f'No .content_tags files found for {app}')

def trim_analysis_char_by_char_trimming(representation='content_tags'):
    if representation not in ['content_tags', 'content', 'tags']:
        print(f'Invalid representation type: {representation}')
        return

    ss_path = f'/Users/lgk/Documents/uni/BA-Local/Data/SS.csv'
    ss = pd.read_csv(ss_path)

    for app in APPS:
        app_ss = ss[ss['appname'] == app]
        path_to_folder = os.path.join(base_path, app)

        total_files = 0
        total_length = 0
        total_trimmed_length = 0

        for index, row in app_ss.iterrows():
            state1 = row['state1']
            state2 = row['state2']

            file1_path = os.path.join(path_to_folder, f'{state1}.html.{representation}')
            file2_path = os.path.join(path_to_folder, f'{state2}.html.{representation}')

            # Read and process state1
            with open(file1_path, 'r') as f:
                state1_representation = json.loads(f.read().strip())
                length_state1 = len(state1_representation)

            # Read and process state2
            with open(file2_path, 'r') as f:
                state2_representation = json.loads(f.read().strip())
                length_state2 = len(state2_representation)

            # Trim the common parts using the trim_common_html function
            trimmed_state1, trimmed_state2 = utils.trim_common_html(state1_representation, state2_representation)

            # Convert trimmed states back to arrays of strings
            trimmed_state1 = trimmed_state1.split(' ')
            trimmed_state2 = trimmed_state2.split(' ')

            # Record the lengths after trimming
            trimmed_length_state1 = len(trimmed_state1)
            trimmed_length_state2 = len(trimmed_state2)

            # Update totals
            total_files += 2  # Since we're processing two files (state1 and state2)
            total_length += length_state1 + length_state2
            total_trimmed_length += trimmed_length_state1 + trimmed_length_state2

        # Calculate the average trimming ratio if there are files processed
        if total_files > 0:
            avg_length = total_length / total_files
            avg_trimmed_length = total_trimmed_length / total_files
            trimming_ratio = avg_trimmed_length / avg_length

            print(f'{trimming_ratio:.3f}')
        else:
            print(f'No {representation} files found for {app}')


if __name__ == '__main__':
    # trim_analysis_body_only('content_tags')
    # token_analysis('content_tags')
    # trim_analysis_char_by_char_trimming('content_tags')
    trimming_token_analysis('content_tags', 'body_only') # trimming_token_analysis('tags', 'char_by_char')
