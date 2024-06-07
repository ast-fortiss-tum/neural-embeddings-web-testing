from transformers import AutoModelForSequenceClassification, AutoTokenizer
import torch
from bs4 import BeautifulSoup, Comment
from bs4.element import NavigableString
import gensim
import json
import re

# File to define utils for abstract function python

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

# Preprocesses the input states for inference by tokenizing them using the provided tokenizer.
def preprocess_for_inference(state1, state2, tokenizer):
    tokenized_inputs = tokenizer(state1, state2,
                                 padding='max_length',
                                 truncation='longest_first',
                                 max_length=512,
                                 return_tensors='pt')
    return tokenized_inputs

# Runs inference on the provided inputs using the provided model.
def get_prediction(model, inputs):
    with torch.no_grad():  # disable gradient computation
        outputs = model(**inputs)
    # extract logits and apply softmax
    probabilities = torch.softmax(outputs.logits, dim=-1)
    # predict the class with the highest probability
    predicted_class_id = probabilities.argmax(dim=-1).item()
    return predicted_class_id

def bert_equals(dom1, dom2, model, tokenizer, feature='content_tags'):
    """
    1. Extract tag, content, or tags + content
    2. Tokenize the extracted data
    3. Predict the similarity between the two states using a fine-tuned BERT model
    :return: The predicted class (0 for distinct, 1 for clone/near-duplicate).
    """
    corpus1 = process_html(dom1)
    corpus2 = process_html(dom2)
    if feature.endswith('content_tags'):
        data1 = corpus1[2]
        data2 = corpus2[2]
    elif feature.endswith('tags'):
        data1 = corpus1[1]
        data2 = corpus2[1]
    elif feature.endswith('content'):
        data1 = corpus1[0]
        data2 = corpus2[0]
    else:
        raise ValueError(f'Invalid feature type: {feature}') #TODO: handling of distance-all

    processed_inputs = preprocess_for_inference(data1, data2, tokenizer)
    predicted = get_prediction(model, processed_inputs)
    return predicted

# Function to fix JSON strings incoming from the crawler
def fix_json(json_string):
    fixed_json = []
    in_string = False
    ix = 0
    char = ''

    for ix, char in enumerate(json_string):
        if char == '"' and in_string:
            if json_string[ix+1] == ',' or json_string[ix+1] == '}' or json_string[ix+1:].replace(" ", "").startswith('}') or json_string[ix+1:].replace(" ", "").startswith('\n'):
                in_string = False
            else:
                fixed_json.append("'")
                continue
        elif not in_string and char == '"':
            if len(fixed_json) > 2 and (fixed_json[-1] == ':' or fixed_json[-2] == ':'):
                in_string = True
        if char == '\\':
            fixed_json.append('\\')
        fixed_json.append(char)
        if not in_string and char == '}':
            break

    fixed_json_string = ''.join(fixed_json)
    fixed_json_string = fixed_json_string.replace('\n', '')
    fixed_json_string = sanitize_json_string(fixed_json_string)

    try:
        json.loads(fixed_json_string)
        return fixed_json_string
    except json.JSONDecodeError as e:
        print(f"Error decoding JSON: {e},\n fixed JSON: {fixed_json_string}")
        return "Error decoding JSON"

# Function to sanitize JSON strings by removing non-printable ASCII characters, TODO: check if this destroys the further processing somehow
def sanitize_json_string(json_str):
    # Define a regex pattern for valid printable ASCII characters (excluding control characters)
    printable_ascii_pattern = re.compile(r'[\x20-\x7E]')

    # Remove all characters that do not match the printable ASCII pattern
    sanitized_str = ''.join(char if printable_ascii_pattern.match(char) else '' for char in json_str)

    return sanitized_str
