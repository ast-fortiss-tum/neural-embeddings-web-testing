from ast import excepthandler
from os.path import join
from os import mkdir, remove
import json
from random import random

all_html_path = '../dataset/all_html'
sets_path = '../dataset/sets'
output_corpus = '../dataset/train_model_corpus'

content_model_train_set = join(sets_path, 'content_model_train_set.json')
tags_model_train_set = join(sets_path, 'tags_model_train_set.json')
content_tags_model_train_set = join(sets_path, 'content_tags_model_train_set.json')

try:
    mkdir(output_corpus)
except:
    pass

def generate_line_sentences(set_path, output):
    try:
        remove(output)
    except:
        pass
    with open(set_path, 'r') as fp:
        set_metadata = json.load(fp)
    # extract data
    for app_name in set_metadata:
        for file_metadata in set_metadata[app_name]:
            file_path = join(all_html_path, file_metadata['path'])
            try:
                with open(file_path, 'r') as fp:
                    data = json.load(fp)
                # write each doc (html contet, html tag etc) as a single line
                with open(output, 'a+') as fp:
                    fp.write(' '.join(data))
                    fp.write('\n')
            except Exception as e:
                print(app_name, file_path, e)


content_model_train_set_path = join(output_corpus, 'content_model_train_set.line_sentence')
tags_model_train_set_path = join(output_corpus, 'tags_model_train_set.line_sentence')
content_tags_model_train_set_path = join(output_corpus, 'content_tags_model_train_set.line_sentence')

# all docs
generate_line_sentences(content_model_train_set, content_model_train_set_path)
generate_line_sentences(tags_model_train_set, tags_model_train_set_path)
generate_line_sentences(content_tags_model_train_set, content_tags_model_train_set_path)

# only pick some docs
def generate_small_corpus(probability, inpt_path, outpt_path):
    with open(inpt_path, 'r') as fp:
        try:
            # remove because later append to file _. avoid appending to old file content
            remove(outpt_path)
        except:
            pass
        for line in fp:
            if random() < probability:
                with open(outpt_path, 'a+') as op:
                    op.write(line)
                    op.write('\n')


probability_line_selection = 0.05
generate_small_corpus(probability_line_selection, content_model_train_set_path, join(output_corpus, 'content_model_train_set_SMALL.line_sentence'))
generate_small_corpus(probability_line_selection, tags_model_train_set_path, join(output_corpus, 'tags_model_train_set_SMALL.line_sentence'))
generate_small_corpus(probability_line_selection, content_tags_model_train_set_path, join(output_corpus, 'content_tags_model_train_set_SMALL.line_sentence'))

probability_line_selection = 0.005
generate_small_corpus(probability_line_selection, content_model_train_set_path, join(output_corpus, 'content_model_train_set_VERY_SMALL.line_sentence'))
generate_small_corpus(probability_line_selection, tags_model_train_set_path, join(output_corpus, 'tags_model_train_set_VERY_SMALL.line_sentence'))
generate_small_corpus(probability_line_selection, content_tags_model_train_set_path, join(output_corpus, 'content_tags_model_train_set_VERY_SMALL.line_sentence'))