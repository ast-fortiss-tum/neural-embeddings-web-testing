# run in neural-embeddings-web-testing\huggingface : '.\venv\Scripts\Activate'

from transformers import AutoModel, AutoTokenizer
import torch
from sklearn.metrics.pairwise import cosine_similarity


def process_html(model, tokenizer, html_content):
    # Tokenize the HTML code
    tokens = tokenizer(html_content, return_tensors="pt", truncation=True, max_length=512)

    # Run the model on the tokens
    outputs = model(**tokens)

    # Get the embeddings (adjust the aggregation method based on your needs)
    embeddings = outputs.last_hidden_state.mean(dim=1)

    return embeddings


# Load the CodeBERT model and tokenizer
model_name = "microsoft/codebert-base"
model = AutoModel.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)

# Example HTML content
mdn_webdocs = open("data/raw/MDN_webdocs.html", "r").read()
html1 = mdn_webdocs

mdn_blog = open("data/raw/MDN_Blog.html", "r").read()
html2 = mdn_blog

# Split HTML content into chunks
chunk_size = 512  # Adjust as needed

exit(0)
# do not run this
# chunks1 = [html1[i:i + chunk_size] for i in range(0, len(html1), chunk_size)]
# chunks2 = [html2[i:i + chunk_size] for i in range(0, len(html2), chunk_size)]
#
# # Process each chunk and aggregate the embeddings
# embeddings1 = torch.cat([process_html(model, tokenizer, chunk) for chunk in chunks1], dim=0)
# embeddings2 = torch.cat([process_html(model, tokenizer, chunk) for chunk in chunks2], dim=0)
#
# # Calculate cosine similarity
# similarity_score = cosine_similarity(embeddings1.detach().numpy().reshape(1, -1),
#                                      embeddings2.detach().numpy().reshape(1, -1))
# print(f"Cosine Similarity: {similarity_score[0][0]}")








# finetune => todo: look into fine-tuning
# from transformers import Trainer, TrainingArguments

# # Set up Trainer and TrainingArguments
# trainer = Trainer(
#     model=model,
#     args=TrainingArguments(
#         output_dir="./output",
#         per_device_train_batch_size=4,
#         num_train_epochs=3,
#     ),
#     train_dataset=train_dataset,  # Your training dataset
# )

# # Fine-tune the model
# trainer.train()
