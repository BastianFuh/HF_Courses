"""Based on the Huggingface Course for NLP Processing. See, https://huggingface.co/learn/nlp-course/chapter5/6."""

import pandas as pd
from datasets import load_from_disk, Dataset
from transformers import AutoTokenizer, AutoModel
from other.util import device


def concatenate_text(examples):
    return {
        "text": examples["title"]
        + " \n "
        + examples["body"]
        + " \n "
        + examples["comments"]
    }


def cls_pooling(model_output):
    return model_output.last_hidden_state[:, 0]


def get_embeddings(text_list):
    encoded_input = tokenizer(
        text_list, padding=True, truncation=True, return_tensors="pt"
    )
    encoded_input = {k: v.to(device) for k, v in encoded_input.items()}
    model_output = model(**encoded_input)
    return cls_pooling(model_output)


# Load dataset from previous script
issues_dataset = load_from_disk("datasets-issues-with_comments")
print(issues_dataset)

# Filter all out all non issues and once which have no comments
issues_dataset = issues_dataset.filter(
    lambda x: not x["is_pull_request"] and len(x["comments"]) > 0
)
print(issues_dataset)

# Remove Uneeded column
columns = set(issues_dataset.column_names)

columns_to_keep = set(["title", "body", "html_url", "comments"])
columns_to_remove = columns - columns_to_keep

issues_dataset = issues_dataset.remove_columns(columns_to_remove)
print(issues_dataset)

# Filter out empty bodies
issues_dataset = issues_dataset.filter(lambda x: x["body"] is not None)

# Create one row per comment
issues_dataset.set_format("pandas")
df: pd.DataFrame = issues_dataset[:]

comments_df = df.explode("comments", ignore_index=True)
print(comments_df.head(4))

comments_dataset = Dataset.from_pandas(comments_df)
print(comments_dataset)

# Get number of words per comment
comments_dataset = comments_dataset.map(
    lambda x: {"comment_length": len(x["comments"].split())}
)
print(comments_dataset)

# Remove short comments
comments_dataset = comments_dataset.filter(lambda x: x["comment_length"] > 15)
print(comments_dataset)


# Combine text
comments_dataset = comments_dataset.map(concatenate_text)
print(comments_dataset)

model_ckpt = "sentence-transformers/multi-qa-mpnet-base-dot-v1"
tokenizer = AutoTokenizer.from_pretrained(model_ckpt)
model = AutoModel.from_pretrained(model_ckpt).to(device)

# Create embeddings
embeddings_dataset = comments_dataset.map(
    lambda x: {"embeddings": get_embeddings(x["text"]).detach().cpu().numpy()[0]}
)

# Add a FAISS (Facebook AI Similarity Serarch) index
embeddings_dataset.add_faiss_index(column="embeddings")


# Create the embedding for the question
question = "How can I load a dataset offline?"
question_embedding = get_embeddings([question]).cpu().detach().numpy()
print(question_embedding.shape)

# Score the question against the other embeddings
scores, samples = embeddings_dataset.get_nearest_examples(
    "embeddings", question_embedding, k=5
)

# Use pandas for easy sorting
samples_df = pd.DataFrame.from_dict(samples)
samples_df["scores"] = scores
samples_df.sort_values("scores", ascending=False, inplace=True)

for _, row in samples_df.iterrows():
    print(f"COMMENT: {row.comments}")
    print(f"SCORE: {row.scores}")
    print(f"TITLE: {row.title}")
    print(f"URL: {row.html_url}")
    print("=" * 50)
    print()
