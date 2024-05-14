from transformers import AutoModelForSequenceClassification, AutoTokenizer
import torch
from bs4 import BeautifulSoup, Comment
from bs4.element import NavigableString
import gensim

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
