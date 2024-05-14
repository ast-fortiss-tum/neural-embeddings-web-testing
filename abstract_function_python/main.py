import json
import pickle

# new imports
import utils
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from datasets import load_dataset
import torch
import pandas as pd
from flask import Flask, request

feature = 'content'
hf_model_name = f'lgk03/NDD-claroline_test-{feature}'  # this should be dynamically set - currently the best performing model in terms of f1 score
tokenizer = AutoTokenizer.from_pretrained(hf_model_name)
model = AutoModelForSequenceClassification.from_pretrained(hf_model_name)
model.eval()  # set model into evaluation mode


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
    result = utils.bert_equals(obj1, obj2, model, tokenizer, feature)

    result = "true" if result == 1 else "false"

    # return true if the two objects are the clones/near-duplicates
    return result


if __name__ == "__main__":
    app.run(debug=False)
