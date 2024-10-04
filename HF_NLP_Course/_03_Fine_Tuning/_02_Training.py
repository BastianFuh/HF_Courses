"""Based on the Huggingface Course for NLP Processing. See, https://huggingface.co/learn/nlp-course/chapter3/3"""

import numpy as np
import evaluate
from transformers import (
    Trainer,
    AutoTokenizer,
    TrainingArguments,
    AutoModelForSequenceClassification,
    DataCollatorWithPadding,
)
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
    # Using the metrics package compute the metrics
    return metric.compute(predictions=predictions, references=labels)


# Basic setup
raw_datasets = load_dataset("glue", "mrpc")
checkpoint = "bert-base-uncased"

# Load the tokenizer
tokenizer = AutoTokenizer.from_pretrained(checkpoint)

# Prepare the data
tokenized_datasets = raw_datasets.map(tokenize_function, batched=True)
data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

# Training args. Given here are the output directory and the evaluation strategy.
training_args = TrainingArguments("test-trainer", eval_strategy="epoch")

# Model with is initialized with the given checkpoint. The head is replaced with
# a new one because it did not contain a suitable own for this task.
model = AutoModelForSequenceClassification.from_pretrained(checkpoint, num_labels=2).to(
    device
)

# Initiate the trainer
trainer = Trainer(
    model,
    training_args,
    train_dataset=tokenized_datasets["train"],
    eval_dataset=tokenized_datasets["validation"],
    data_collator=data_collator,
    tokenizer=tokenizer,
    compute_metrics=compute_metrics,
)

# Run an automated training loop
trainer.train()
