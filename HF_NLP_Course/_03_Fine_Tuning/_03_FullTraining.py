"""Based on the Huggingface Course for NLP Processing. See, https://huggingface.co/learn/nlp-course/chapter3/4"""

import torch
import numpy as np
import evaluate
from torch.utils.data import DataLoader
from accelerate import Accelerator
from transformers import (
    AutoTokenizer,
    get_scheduler,
    AdamW,
    AutoModelForSequenceClassification,
    DataCollatorWithPadding,
)
from tqdm.auto import tqdm
from datasets import load_dataset
from other.util import device


def tokenize_function(input):
    """
    Function handling the tokenization of the dataset. Padding is not set
    because this would apply the same for across the entire dataset.
    So it is only applied to the batch to prevent unnessary overhead.
    """
    cleaned_input = [v for k, v in input.items() if k not in ["idx", "label"]]
    return tokenizer(*cleaned_input, truncation=True)


def compute_metrics(eval_preds):
    """This functions computes the metrics for a training run."""
    metric = evaluate.load("glue", "mrpc")
    logits, labels = eval_preds
    predictions = np.argmax(logits, axis=-1)
    return metric.compute(predictions=predictions, references=labels)


# Basic setup
raw_datasets = load_dataset("glue", "mrpc")
checkpoint = "bert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(checkpoint)

# Preprocess data
tokenized_datasets = raw_datasets.map(tokenize_function, batched=True)
data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

# Remove unessary columns.
tokenized_datasets = tokenized_datasets.remove_columns(
    ["sentence1", "sentence2", "idx"]
)

# Rename columns to fit corrent naming scheme
tokenized_datasets = tokenized_datasets.rename_column("label", "labels")
# Set correct format for tensors
tokenized_datasets.set_format("torch")

print(tokenized_datasets["train"].column_names)
print(tokenized_datasets.column_names)

# Define the dataloaders
train_dataloader = DataLoader(
    tokenized_datasets["train"], shuffle=True, batch_size=8, collate_fn=data_collator
)
eval_dataloader = DataLoader(
    tokenized_datasets["validation"], batch_size=8, collate_fn=data_collator
)

# See if we have the correct arrangement
for batch in train_dataloader:
    break
print({k: v.shape for k, v in batch.items()})

# Accelerator can be used to handle cases in which the training is done
# on an distributed system.
accelerator = Accelerator()

model = AutoModelForSequenceClassification.from_pretrained(checkpoint, num_labels=2)

# Optimizer using the Adam algorithm using weight decay
optimizer = AdamW(model.parameters(), lr=5e-5)

# Test if everything works
outputs = model(**batch)
print(outputs.loss, outputs.logits.shape)

# Create Accelerator wrappers for everything
train_dataloader, eval_dataloader, model, optimizer = accelerator.prepare(
    train_dataloader, eval_dataloader, model, optimizer
)

# Setup training hyperparameters
num_epochs = 3
num_training_steps = num_epochs * len(train_dataloader)
lr_scheduler = get_scheduler(
    "linear",
    optimizer=optimizer,
    num_warmup_steps=0,
    num_training_steps=num_training_steps,
)

print(num_training_steps)

# Setup dynamic progress bar
progress_bar = tqdm(range(num_training_steps))


metric = evaluate.load("glue", "mrpc")

# Training loop
model.train()
step = 0
for epoch in range(num_epochs):
    for batch in train_dataloader:
        step += 1
        outputs = model(**batch)
        loss = outputs.loss
        accelerator.backward(loss)

        optimizer.step()
        lr_scheduler.step()
        optimizer.zero_grad()
        progress_bar.update(1)

        if step % 500 == 499:
            model.eval()
            for batch in eval_dataloader:
                batch = {k: v.to(device) for k, v in batch.items()}
                with torch.no_grad():
                    outputs = model(**batch)

                logits = outputs.logits
                predictions = torch.argmax(logits, dim=-1)
                metric.add_batch(predictions=predictions, references=batch["labels"])

            print(metric.compute())

# Evaluation loop
model.eval()
for batch in eval_dataloader:
    batch = {k: v.to(device) for k, v in batch.items()}
    with torch.no_grad():
        outputs = model(**batch)

    logits = outputs.logits
    predictions = torch.argmax(logits, dim=-1)
    metric.add_batch(predictions=predictions, references=batch["labels"])

print(metric.compute())
