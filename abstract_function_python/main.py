import json
import pickle

# new imports
import utils
from transformers import AutoModelForSequenceClassification, AutoTokenizer
import torch

hf_path_to_model = 'lgk03/NDD-claroline_test-content'  # this should be dynamically set - currently the best performing model in terms of f1 score

tokenizer = AutoTokenizer.from_pretrained(hf_path_to_model)
model = AutoModelForSequenceClassification.from_pretrained(hf_path_to_model)
model.eval()  # set model into evaluation mode

row = {'TODO'} # TODO one row of the dataset to be used for inference
state1 = row['trimmed_state1']
state2 = row['trimmed_state2']
# map the human classification to the labels that the model will predict (1 = clone/near duplicate & 0 = distinct)
actual_class = 1 if row['HUMAN_CLASSIFICATION'] == 1 or row['HUMAN_CLASSIFICATION'] == 0 else 0

inputs = utils.preprocess_for_inference(state1, state2, tokenizer)
predicted_class_id = utils.infer(model, inputs)

is_correct = predicted_class_id == actual_class

print(f"Model prediction correct? {is_correct}")

print("Exiting")
exit()


# existing code =========

import gensim
import numpy as np
from bs4 import BeautifulSoup
from bs4.element import NavigableString, Comment
from flask import Flask, request
from sklearn.metrics.pairwise import cosine_similarity

app = Flask(__name__)

# call to route /equals executes equalRoute function
# use URL, DOM String, Dom content and DOM syntax tree as params
@app.route('/', methods=('GET', 'POST'))
def equal_route():
    content_type = request.headers.get('Content-Type')
    if content_type == 'application/json' or content_type == 'application/json; utf-8':
        data = json.loads(request.data)
    else:
        return 'Content-Type not supported!'

    # get params sent by java
    parametersJava = data

    obj1 = parametersJava['dom1']
    obj2 = parametersJava['dom2']

    # compute equality of DOM objects
    result = doc2vec_equals(obj1, obj2)

    result = "true" if result[0] == 0 else "false"

    # return true if the two objects are the clones/near-duplicates
    return result


# abstraction function that computes similarity
def doc2vec_equals(obj1, obj2):
    FEATURE = None
    if "doc2vec-distance-all" in CLASSIFIER_USED:
        FEATURE = "doc2vec-distance-all"
    elif "doc2vec-distance-content" in CLASSIFIER_USED:
        FEATURE = "doc2vec-distance-content"
    elif "doc2vec-distance-tags" in CLASSIFIER_USED:
        FEATURE = "doc2vec-distance-tags"
    elif "doc2vec-distance-content-tags" in CLASSIFIER_USED:
        FEATURE = "doc2vec-distance-content-tags"
    assert FEATURE is not None

    dist = get_distance_from_embeddings(obj1, obj2, feature=FEATURE)
    dist = dist.reshape(1, -1)

    word2vec = model.predict(dist)
    return word2vec


def get_distance_from_embeddings(dom1, dom2, feature):
    # 1. extract tag, content, or tags + content
    # 2. generate line sentence? necessary?
    # 3. create embedding
    # 4. cosine similarity

    model_tags = doc2vec_model_tags
    model_content = doc2vec_model_content
    model_content_tags = doc2vec_model_content_tags

    corpus1 = process_html(dom1)
    corpus2 = process_html(dom2)

    emb_dom1 = None
    emb_dom2 = None

    if feature == 'doc2vec-distance-tags':
        data1 = corpus1[1]
        data2 = corpus2[1]
        emb_dom1 = model_tags.infer_vector(data1).reshape(1, -1)
        emb_dom2 = model_tags.infer_vector(data2).reshape(1, -1)
    elif feature == 'doc2vec-distance-content':
        data1 = corpus1[0]
        data2 = corpus2[0]
        emb_dom1 = model_content.infer_vector(data1).reshape(1, -1)
        emb_dom2 = model_content.infer_vector(data2).reshape(1, -1)
    elif feature == 'doc2vec-distance-content_tags':
        data1 = corpus1[2]
        data2 = corpus2[2]
        emb_dom1 = model_content_tags.infer_vector(data1).reshape(1, -1)
        emb_dom2 = model_content_tags.infer_vector(data2).reshape(1, -1)
    elif feature == 'doc2vec-distance-all':
        data1 = corpus1[1]
        data2 = corpus2[1]
        emb_tags1 = model_tags.infer_vector(data1).reshape(1, -1)
        emb_tags2 = model_tags.infer_vector(data2).reshape(1, -1)
        data1 = corpus1[0]
        data2 = corpus2[0]
        emb_content1 = model_content.infer_vector(data1).reshape(1, -1)
        emb_content2 = model_content.infer_vector(data2).reshape(1, -1)
        data1 = corpus1[2]
        data2 = corpus2[2]
        emb_content_tags1 = model_content_tags.infer_vector(data1).reshape(1, -1)
        emb_content_tags2 = model_content_tags.infer_vector(data2).reshape(1, -1)

        emb_dom1 = np.hstack((emb_tags1, emb_content1, emb_content_tags1))
        emb_dom2 = np.hstack((emb_tags2, emb_content2, emb_content_tags2))

    sim = cosine_similarity(emb_dom1, emb_dom2)
    final_sim = np.array([sim[0, 0]])
    return final_sim


def process_html(html):
    soup = BeautifulSoup(html, 'html.parser')
    corpus = ([], [], [])
    retrieve_abstraction_from_html(soup, corpus)
    return corpus


def retrieve_abstraction_from_html(bs, corpus):
    try:
        if type(bs) == NavigableString:
            tokens = gensim.utils.simple_preprocess(bs.string)
            if len(tokens) > 0:
                corpus[0].extend(tokens)
                corpus[2].extend(tokens)
            return

        bs_has_name = bs.name != None
        bs_is_single_tag = str(bs)[-2:] == '/>'

        if bs_has_name and not bs_is_single_tag:
            corpus[1].append(f'<{bs.name}>')
            corpus[2].append(f'<{bs.name}>')
        elif bs_has_name and bs_is_single_tag:
            corpus[1].append(f'<{bs.name}/>')
            corpus[2].append(f'<{bs.name}/>')
        try:
            for c in bs.children:
                if type(c) == Comment:
                    continue
                retrieve_abstraction_from_html(c, corpus)
        except Exception:
            pass
        if bs_has_name and not bs_is_single_tag:
            corpus[1].append(f'</{bs.name}>')
            corpus[2].append(f'</{bs.name}>')
    except Exception as e:
        print('html structure content error', e)
        pass


if __name__ == "__main__":
    app.run(debug=False)
