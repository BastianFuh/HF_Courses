"""Based on the Huggingface Course for NLP Processing. See, https://huggingface.co/learn/nlp-course/chapter7/3."""

import torch
import collections
from tqdm.auto import tqdm
import math

from transformers import (
    AutoModelForMaskedLM,
    AutoTokenizer,
    PreTrainedTokenizer,
    DataCollatorForLanguageModeling,
    default_data_collator,
    get_scheduler,
    pipeline,
)
from datasets import load_dataset
import numpy as np
from torch.utils.data import DataLoader
from torch.optim import AdamW
from accelerate import Accelerator
from huggingface_hub import get_full_repo_name, create_repo, Repository

#
# Config
#


training = True

model_checkpoint = "distilbert-base-uncased"

model = AutoModelForMaskedLM.from_pretrained(model_checkpoint)
tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)

data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer, mlm_probability=0.15
)

wwm_probability = 0.2

text = "This is a great [MASK]."


#
# Functions
#


def tokenize_function(examples, tokenizer: PreTrainedTokenizer = tokenizer):
    result = tokenizer(examples["text"])
    if tokenizer.is_fast:
        result["word_ids"] = [
            result.word_ids(i) for i in range(len(result["input_ids"]))
        ]
    return result


def group_texts(examples: dict):
    # Concatenate all texts
    concatenated_examples = {k: sum(examples[k], []) for k in examples.keys()}
    # Compute length of concatenated texts
    total_length = len(concatenated_examples[list(examples.keys())[0]])
    # We drop the last chunk if it's smaller than chunk_size
    total_length = (total_length // chunk_size) * chunk_size
    # Split by chunks of max_len
    result = {
        k: [t[i : i + chunk_size] for i in range(0, total_length, chunk_size)]
        for k, t in concatenated_examples.items()
    }
    # Create a new labels column
    result["labels"] = result["input_ids"].copy()
    return result


def insert_random_mask(batch):
    features = [dict(zip(batch, t)) for t in zip(*batch.values())]
    masked_inputs = whole_word_masking_data_collator(features)
    # Create a new "masked" column for each column in the dataset
    return {"masked_" + k: v.numpy() for k, v in masked_inputs.items()}


def whole_word_masking_data_collator(features):
    for feature in features:
        word_ids = feature.pop("word_ids")

        # Create a map between words and corresponding token indices
        mapping = collections.defaultdict(list)
        current_word_index = -1
        current_word = None
        for idx, word_id in enumerate(word_ids):
            if word_id is not None:
                if word_id != current_word:
                    current_word = word_id
                    current_word_index += 1
                mapping[current_word_index].append(idx)

        # Randomly mask words
        mask = np.random.binomial(1, wwm_probability, (len(mapping),))
        input_ids = feature["input_ids"]
        labels = feature["labels"]
        new_labels = [-100] * len(labels)
        for word_id in np.where(mask)[0]:
            word_id = word_id.item()
            for idx in mapping[word_id]:
                new_labels[idx] = labels[idx]
                input_ids[idx] = tokenizer.mask_token_id
        feature["labels"] = new_labels

    return default_data_collator(features)


#
# Code
#

# Test the model for a small example

inputs = tokenizer(text, return_tensors="pt")
token_logits = model(**inputs).logits

# Find the location of [MASK] and extract its logits
mask_token_index = torch.where(inputs["input_ids"] == tokenizer.mask_token_id)[1]
mask_token_logits = token_logits[0, mask_token_index, :]

# Pick the [MASK] candidates with the highest logits
top_5_tokens = torch.topk(mask_token_logits, 5, dim=1).indices[0].tolist()

for token in top_5_tokens:
    print(f"'>>> {text.replace(tokenizer.mask_token, tokenizer.decode([token]))}'")


# Get the dataset
imdb_dataset = load_dataset("imdb")

# Get some sample to see the data
see_example_data = False
if see_example_data:
    sample = imdb_dataset["train"].shuffle(seed=42).select(range(3))

    print("\n Train samples \n")
    for row in sample:
        print(f"\n'>>> Review: {row['text']}'")
        print(f"'>>> Label: {row['label']}'")

    sample = imdb_dataset["test"].shuffle(seed=42).select(range(3))

    print("\n Test samples \n")
    for row in sample:
        print(f"\n'>>> Review: {row['text']}'")
        print(f"'>>> Label: {row['label']}'")

    sample = imdb_dataset["unsupervised"].shuffle(seed=42).select(range(3))

    print("\n Unsupervised samples \n")
    for row in sample:
        print(f"\n'>>> Review: {row['text']}'")
        print(f"'>>> Label: {row['label']}'")


#
# Preprocess data
#


# First tokenize data
# Use batched=True to activate fast multithreading!
tokenized_datasets = imdb_dataset.map(
    tokenize_function,
    batched=True,
    batch_size=5000,
    remove_columns=["text", "label"],
)

chunk_size = 128


# Slicing produces a list of lists for each feature
tokenized_samples = tokenized_datasets["train"][:3]

# Chunk data
lm_datasets = tokenized_datasets.map(group_texts, batched=True)
print(lm_datasets)


#
# Training
#

# Reduce size of dataset
train_size = 20_000
test_size = int(0.1 * train_size)

downsampled_dataset = lm_datasets["train"].train_test_split(
    train_size=train_size, test_size=test_size, seed=42
)

# Setup a collator which randomly masks tokens during training
# downsampled_dataset = downsampled_dataset.remove_columns(["word_ids"])
eval_dataset = downsampled_dataset["test"].map(
    insert_random_mask,
    batched=True,
    remove_columns=downsampled_dataset["test"].column_names,
)
eval_dataset = eval_dataset.rename_columns(
    {
        "masked_input_ids": "input_ids",
        "masked_attention_mask": "attention_mask",
        "masked_labels": "labels",
    }
)


batch_size = 64
train_dataloader = DataLoader(
    downsampled_dataset["train"],
    shuffle=True,
    batch_size=batch_size,
    collate_fn=whole_word_masking_data_collator,
)
eval_dataloader = DataLoader(
    eval_dataset, batch_size=batch_size, collate_fn=default_data_collator
)


model = AutoModelForMaskedLM.from_pretrained(model_checkpoint)
optimizer = AdamW(model.parameters(), lr=5e-5)

accelerator = Accelerator()
model, optimizer, train_dataloader, eval_dataloader = accelerator.prepare(
    model, optimizer, train_dataloader, eval_dataloader
)


num_train_epochs = 3
num_update_steps_per_epoch = len(train_dataloader)
num_training_steps = num_train_epochs * num_update_steps_per_epoch

lr_scheduler = get_scheduler(
    "linear",
    optimizer=optimizer,
    num_warmup_steps=0,
    num_training_steps=num_training_steps,
)

model_name = "distilbert-base-uncased-finetuned-imdb-accelerate"
repo_name = get_full_repo_name(model_name)
print(repo_name)

create_repo(repo_name, exist_ok=True)

output_dir = model_name
repo = Repository(output_dir, clone_from=repo_name)

progress_bar = tqdm(range(num_training_steps))

if training:
    for epoch in range(num_train_epochs):
        # Training
        model.train()
        for batch in train_dataloader:
            outputs = model(**batch)
            loss = outputs.loss
            accelerator.backward(loss)

            optimizer.step()
            lr_scheduler.step()
            optimizer.zero_grad()
            progress_bar.update(1)

        # Evaluation
        model.eval()
        losses = []
        for step, batch in enumerate(eval_dataloader):
            with torch.no_grad():
                outputs = model(**batch)

            loss = outputs.loss
            losses.append(accelerator.gather(loss.repeat(batch_size)))

        losses = torch.cat(losses)
        losses = losses[: len(eval_dataset)]
        try:
            perplexity = math.exp(torch.mean(losses))
        except OverflowError:
            perplexity = float("inf")

        print(f">>> Epoch {epoch}: Perplexity: {perplexity}")

        # Save and upload
        accelerator.wait_for_everyone()
        unwrapped_model = accelerator.unwrap_model(model)
        unwrapped_model.save_pretrained(output_dir, save_function=accelerator.save)
        if accelerator.is_main_process:
            tokenizer.save_pretrained(output_dir)
            repo.push_to_hub(
                commit_message=f"Training in progress epoch {epoch}", blocking=False
            )


mask_filler = pipeline("fill-mask", model=repo_name)

preds = mask_filler(text)

for pred in preds:
    print(f">>> {pred['sequence']}")
