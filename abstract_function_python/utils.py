from transformers import AutoModelForSequenceClassification, AutoTokenizer
import torch

# File to define utils for abstract function python

def preprocess_for_inference(state1, state2, tokenizer):
    """
    Preprocesses the input states for inference by tokenizing them using the provided tokenizer.
    :param state1: respective html representation of the first state (i.e. state12.html.content_tags)
    :param state2: respective html representation of the second state (i.e. state12.html.content_tags)
    :param tokenizer: The tokenizer to use for tokenization.
    :return: A dictionary containing the tokenized representations of `state1` and `state2`.
      This dictionary includes the following keys: 'input_ids', 'token_type_ids', 'attention_mask',
      where each key maps to a tensor representing the respective tokenized data.
    """
    tokenized_inputs = tokenizer(state1, state2,
                                 padding='max_length',
                                 truncation='longest_first',
                                 max_length=512,
                                 return_tensors='pt')
    return tokenized_inputs


def infer(model, inputs):
    """
    Runs inference on the provided inputs using the provided model.
    :param model: The model to use for inference.
    :param inputs: The inputs to run inference on.
    :return: The predicted class id.
    """
    with torch.no_grad():  # disable gradient computation
        outputs = model(**inputs)
    # extract logits and apply softmax
    probabilities = torch.softmax(outputs.logits, dim=-1)
    # predict the class with the highest probability
    predicted_class_id = probabilities.argmax(dim=-1).item()
    return predicted_class_id
