# %% [markdown]
# # State-Pair classification

# %%
from huggingface_hub import HfFolder

# Retrieve the access token from Colab's userdata
hf_access_token = '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'

if hf_access_token is not None:
    # Set the token in the Hugging Face Folder (this authenticates you)
    HfFolder.save_token(hf_access_token)
    print("Hugging Face access token set.")
else:
    print("No Hugging Face access token available.")


# %% [markdown]
# ## Initalize

# %%
import os
import csv
import json
import pandas as pd

app_num = 6

# file_ending = '.html.content_tags'
file_ending = '.html.content'
# file_ending = '.html.tags'

apps = ['addressbook', 'claroline', 'ppma', 'mrbs', 'mantisbt', 'dimeshift', 'pagekit', 'phoenix', 'petclinic'] # order from WebEmbed Paper Table 5 (Results)
cur_test_app = apps[app_num]


# Dictionary containing the number of occurrences for each class (0 and 1) in each app's dataset, with [n0, n1], n0: distinct/n1: clone/Near-duplicate
absolute_labels = {
    'addressbook': [6142, 2373],
    'claroline': [14988, 2778],
    'ppma': [4320, 531],
    'mrbs': [7254, 4071],
    'mantisbt': [10206, 1119],
    'dimeshift': [10683, 945],
    'pagekit': [5782, 3948],
    'phoenix': [6569, 4606],
    'petclinic': [9411, 1615]
}


# %% [markdown]
# ## Load and Prepare Dataset

# %%
import pandas as pd
from datasets import Dataset, DatasetDict, Features, ClassLabel, Value
from sklearn.model_selection import train_test_split

# Load the CSV file into a pandas DataFrame
file_path = f"{cur_test_app}.csv"
df = pd.read_csv(file_path)
print(f"Dataset loaded from {cur_test_app}.csv")

# Split the DataFrame into training and test sets
train_df, test_df = train_test_split(df, test_size=0.2, random_state=42)

# Convert the DataFrames into Hugging Face datasets
train_dataset = Dataset.from_pandas(train_df)
test_dataset = Dataset.from_pandas(test_df)

# Create a DatasetDict
dataset = DatasetDict({
    'train': train_dataset,
    'test': test_dataset
})

# Define the feature types
features = Features({
    'HUMAN_CLASSIFICATION': ClassLabel(names=['class_0', 'class_1']),
    'trimmed_state1': Value('string'),
    'trimmed_state2': Value('string'),
})

# Transforming into binary classification
def map_labels(example):
    if example['HUMAN_CLASSIFICATION'] == 2:
        example['HUMAN_CLASSIFICATION'] = 0
    else:
        example['HUMAN_CLASSIFICATION'] = 1
    return example

# Apply the transformation
dataset = dataset.map(map_labels)

print(f"Dataset loaded and split into training and test sets with 80/20 split for {cur_test_app}{file_ending}")

# %% [markdown]
# ## Preprocess

# %% [markdown]
# load a DistilBERT tokenizer

# %%
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("allenai/longformer-base-4096")
# tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")
print("Tokenizer loaded.")

# %% [markdown]
# Preprocess:

# %%
import torch
def preprocess(examples):
    tokenized_inputs = tokenizer(examples['state1_content'], examples['state2_content'],
                                 padding='max_length',
                                 truncation='longest_first',
                                 max_length=4096, # LongFormer max length: 4096
                                 return_tensors='pt') # Return PyTorch tensors

    return {'input_ids': tokenized_inputs['input_ids'],
            'attention_mask': tokenized_inputs['attention_mask'],
            'labels': torch.tensor(examples['HUMAN_CLASSIFICATION'])
            }

# %%
print(f"Tokenizing Data for {cur_test_app} with html representation: {file_ending.split('.')[-1]}")
tokenized_data = {split: ds.map(preprocess, batched=True, cache_file_name=None, batch_size=32) for split, ds in dataset.items()}

# %%
tokenized_data

# %% [markdown]
# Create a batch of examples using [DataCollatorWithPadding](https://huggingface.co/docs/transformers/main/en/main_classes/data_collator#transformers.DataCollatorWithPadding). It's more efficient to *dynamically pad* the samples to the longest length in a batch during collation, instead of padding the whole dataset to the maximum length.

# %%
from transformers import DataCollatorWithPadding

data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

# %% [markdown]
# ## Evaluate

# %% [markdown]
# Load F_1-Score as metric

# %%
import evaluate

f1 = evaluate.load("f1")

# %%
import numpy as np
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)

    accuracy = accuracy_score(labels, predictions)

    precision, recall, f1, _ = precision_recall_fscore_support(labels, predictions, average='weighted')

    return {
        'accuracy': accuracy,
        'f1': f1,
        'precision': precision,
        'recall': recall
    }

# %% [markdown]
# ## Train

# %%
from transformers import AutoModelForSequenceClassification, TrainingArguments, Trainer

# num labels 2 => binary classification
model = AutoModelForSequenceClassification.from_pretrained(
    "allenai/longformer-base-4096", num_labels=2) # distilbert-base-uncased

# %%
import torch
import torch.nn.functional as F
from transformers import Trainer

# arr: [n0, n1], (refer to dictionary above)
def calculate_class_weights(arr):
    n0 = arr[0]
    n1 = arr[1]

    N = n0 + n1
    w0 = N / n0
    w1 = N / n1
    w_min = min(w0, w1)

    return torch.tensor([w0 / w_min, w1 / w_min])

class FocalLossTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False):
        labels = inputs.pop("labels")
        outputs = model(**inputs)
        logits = outputs.logits

        class_weights = calculate_class_weights(absolute_labels[cur_test_app]).to(logits.device)  # Adjusted based on class distribution

        gamma = 2.0
        ce_loss = F.cross_entropy(logits, labels, weight=class_weights, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = ((1 - pt) ** gamma * ce_loss).mean()

        return (focal_loss, outputs) if return_outputs else focal_loss


model_name = f"WITHINAPPS_NDD-{cur_test_app}_test-{file_ending.split('.')[-1]}-LongFormer"
print(f"Training model: {model_name}")

training_args = TrainingArguments(
    output_dir=model_name,
    learning_rate=2e-5,
    per_device_train_batch_size=16, # we may have to reduce this if it doesn't fit in memory
    per_device_eval_batch_size=16, # we may have to reduce this if it doesn't fit in memory
    gradient_accumulation_steps=4,
    num_train_epochs=3, # 3 epochs
    weight_decay=0.01,
    eval_strategy="epoch",
    save_strategy="epoch",
    load_best_model_at_end=True,
    push_to_hub=True,
)

trainer = FocalLossTrainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_data["train"],
    eval_dataset=tokenized_data["test"],
    tokenizer=tokenizer,
    data_collator=data_collator,
    compute_metrics=compute_metrics,
)

trainer.train()

# %%
trainer.push_to_hub()
