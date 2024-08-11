import csv
import json
import os
import pandas as pd
from transformers import AutoTokenizer

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


def trim_analysis(representation='content_tags'):
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

                    start_index = content_tags.index("<body>") + 1
                    end_index = content_tags.index("</body>")
                    trimmed_content_tags = content_tags[start_index:end_index]
                    trimmed_length = len(trimmed_content_tags)

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

if __name__ == '__main__':
    # trim_analysis('content_tags')
    token_analysis('tags')
