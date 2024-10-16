"""Based on the Huggingface Course for NLP Processing. See, https://huggingface.co/learn/nlp-course/chapter7/5."""


#
#
# This model does not use the recommended dataset mention in the guide because it is not
# available anymore.
#
#

import evaluate
import nltk
import numpy as np
import pandas as pd
from accelerate import Accelerator
from datasets import load_dataset, concatenate_datasets, DatasetDict
from huggingface_hub import create_repo, get_full_repo_name, Repository
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
import torch
from transformers import (
    MT5ForConditionalGeneration,
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
    DataCollatorForSeq2Seq,
    pipeline,
    get_scheduler,
    AdamW,
)
from nltk.tokenize import sent_tokenize

#
# Config
#

run_training = False

batch_size = 8

# model_checkpoint = "facebook/mbart-large-50"
model_checkpoint = "google/mt5-small"
max_length = 128

lang_from = "en"
lang_to = "de"

max_input_length = 512
max_target_length = 30

num_train_epochs = 10

# Objects
tokenizer = AutoTokenizer.from_pretrained(model_checkpoint, return_tensors="pt")
model = AutoModelForSeq2SeqLM.from_pretrained(model_checkpoint).to("cuda")
data_collator = DataCollatorForSeq2Seq(tokenizer, model=model)
rouge_score = evaluate.load("rouge")
optimizer = AdamW(model.parameters(), lr=2e-5)


# model = MT5ForConditionalGeneration.from_pretrained("google/mt5-small").to("cuda")
# tokenizer = AutoTokenizer.from_pretrained("google/mt5-small")
# article = "UN Offizier sagt, dass weiter verhandelt werden muss in Syrien."
# summary = "Weiter Verhandlung in Syrien."
# inputs = tokenizer(article, text_target=summary, return_tensors="pt")

# for k in inputs.keys():
#    inputs[k] = inputs[k].to("cuda")

# print(inputs)

# outputs = model(**inputs)
# loss = outputs.loss

# print(outputs)

# exit()

#
# Functions
#


def prepare_dataset(examples):
    text = list()
    summaries = list()
    for ex_index in range(len(examples["url"])):
        url_label = (
            examples["url"][ex_index]
            .replace("https://de.wikihow.com/", "")
            .replace("https://www.wikihow.com/", "")
            .replace("-", " ")
        )
        #
        for i in range(len(examples["article"][ex_index]["section_name"])):
            summaries.append(
                url_label + ". " + examples["article"][ex_index]["section_name"][i]
            )
            text.append(examples["article"][ex_index]["document"][i])
    #
    return {"summary": summaries, "text": text}


def preprocess_function(examples):
    model_inputs = tokenizer(
        examples["text"],
        max_length=max_input_length,
        truncation=True,
    )
    labels = tokenizer(
        examples["summary"],
        max_length=max_target_length,
        truncation=True,
    )
    model_inputs["labels"] = labels["input_ids"]
    return model_inputs


def three_sentence_summary(text):
    return "\n".join(sent_tokenize(text)[:3])


def evaluate_baseline(dataset, metric):
    summaries = [three_sentence_summary(text) for text in dataset["text"]]
    return metric.compute(predictions=summaries, references=dataset["summary"])


def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    # Decode generated summaries into text
    decoded_preds = tokenizer.batch_decode(predictions, skip_special_tokens=True)
    # Replace -100 in the labels as we can't decode them
    labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
    # Decode reference summaries into text
    decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)
    # ROUGE expects a newline after each sentence
    decoded_preds = ["\n".join(sent_tokenize(pred.strip())) for pred in decoded_preds]
    decoded_labels = [
        "\n".join(sent_tokenize(label.strip())) for label in decoded_labels
    ]
    # Compute ROUGE scores
    result = rouge_score.compute(
        predictions=decoded_preds, references=decoded_labels, use_stemmer=True
    )
    # Extract the median scores
    result = {key: value.mid.fmeasure * 100 for key, value in result.items()}
    return {k: round(v, 4) for k, v in result.items()}


def postprocess_text(preds, labels):
    preds = [pred.strip() for pred in preds]
    labels = [label.strip() for label in labels]

    # ROUGE expects a newline after each sentence
    preds = ["\n".join(nltk.sent_tokenize(pred)) for pred in preds]
    labels = ["\n".join(nltk.sent_tokenize(label)) for label in labels]

    return preds, labels


#
# Data Processing
#

german_dataset = load_dataset("esdurmus/wiki_lingua", "german", trust_remote_code=True)
english_dataset = load_dataset(
    "esdurmus/wiki_lingua", "english", trust_remote_code=True
)

# Change the format of the data to be suitable to the task
german_dataset = german_dataset.map(
    prepare_dataset, batched=True, remove_columns=english_dataset["train"].column_names
)

english_dataset = english_dataset.map(
    prepare_dataset, batched=True, remove_columns=english_dataset["train"].column_names
)

# Remove data points with excessively long references text
german_dataset = german_dataset.filter(lambda x: len(x["text"]) <= 512)
english_dataset = english_dataset.filter(lambda x: len(x["text"]) <= 512)

# Adjust the length of the dataset, so that they are the same
english_dataset["train"] = (
    english_dataset["train"].shuffle().select(range(len(german_dataset["train"])))
)

# Create split
german_dataset = german_dataset["train"].train_test_split(train_size=0.9, shuffle=True)
english_dataset = english_dataset["train"].train_test_split(
    train_size=0.9, shuffle=True
)

# Save split for validation
validate_german_dataset = german_dataset.pop("test")
validate_english_dataset = english_dataset.pop("test")

# Create split for testing
german_dataset = german_dataset["train"].train_test_split(train_size=0.9, shuffle=True)
english_dataset = english_dataset["train"].train_test_split(
    train_size=0.9, shuffle=True
)

# Readd validation split
german_dataset["validation"] = validate_german_dataset
english_dataset["validation"] = validate_english_dataset


# Combine both datasets
concat_dataset = DatasetDict()

for split in english_dataset.keys():
    concat_dataset[split] = concatenate_datasets(
        [german_dataset[split], english_dataset[split]]
    ).shuffle()

# Tokenize dataset
tokenized_datasets = concat_dataset.map(
    preprocess_function,
    batched=True,
    remove_columns=concat_dataset["train"].column_names,
)


# Create baseline
score = evaluate_baseline(concat_dataset["validation"], rouge_score)
print(score)
rouge_names = ["rouge1", "rouge2", "rougeL", "rougeLsum"]
rouge_dict = dict((rn, round(score[rn] * 100, 2)) for rn in rouge_names)
print(rouge_dict)

# Create loaders
tokenized_datasets.set_format("torch")

train_dataloader = DataLoader(
    tokenized_datasets["train"],
    shuffle=True,
    collate_fn=data_collator,
    batch_size=batch_size,
    # num_workers=3,
)
eval_dataloader = DataLoader(
    tokenized_datasets["validation"],
    collate_fn=data_collator,
    batch_size=batch_size,
    # num_workers=3,
)


#
# Code
#

accelerator = Accelerator()
model, optimizer, train_dataloader, eval_dataloader = accelerator.prepare(
    model, optimizer, train_dataloader, eval_dataloader
)

num_update_steps_per_epoch = len(train_dataloader)
num_training_steps = num_train_epochs * num_update_steps_per_epoch

lr_scheduler = get_scheduler(
    "linear",
    optimizer=optimizer,
    num_warmup_steps=0,
    num_training_steps=num_training_steps,
)


# Create and setup repo
model_name = "mt5-finetuned-wiki_lingua-en-de"
repo_name = get_full_repo_name(model_name)
create_repo(repo_name, exist_ok=True)

output_dir = model_name
repo = Repository(output_dir, clone_from=repo_name)


if run_training:
    progress_bar = tqdm(range(num_training_steps))

    for epoch in range(num_train_epochs):
        # Training
        if True:
            model.train()
            for step, batch in enumerate(train_dataloader):
                outputs = model(**batch)
                loss = outputs.loss
                accelerator.backward(loss)

                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()
                progress_bar.update(1)

        # Evaluation
        print("Start eval")
        model.eval()
        for step, batch in enumerate(eval_dataloader):
            with torch.no_grad():
                generated_tokens = accelerator.unwrap_model(model).generate(
                    batch["input_ids"],
                    attention_mask=batch["attention_mask"],
                )

                generated_tokens = accelerator.pad_across_processes(
                    generated_tokens, dim=1, pad_index=tokenizer.pad_token_id
                )
                labels = batch["labels"]

                # If we did not pad to max length, we need to pad the labels too
                labels = accelerator.pad_across_processes(
                    batch["labels"], dim=1, pad_index=tokenizer.pad_token_id
                )

                generated_tokens = accelerator.gather(generated_tokens).cpu().numpy()
                labels = accelerator.gather(labels).cpu().numpy()

                # Replace -100 in the labels as we can't decode them
                labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
                if isinstance(generated_tokens, tuple):
                    generated_tokens = generated_tokens[0]
                decoded_preds = tokenizer.batch_decode(
                    generated_tokens, skip_special_tokens=True
                )
                decoded_labels = tokenizer.batch_decode(
                    labels, skip_special_tokens=True
                )

                decoded_preds, decoded_labels = postprocess_text(
                    decoded_preds, decoded_labels
                )

                rouge_score.add_batch(
                    predictions=decoded_preds, references=decoded_labels
                )

        print("Stop eval")
        # Compute metrics
        result = rouge_score.compute()
        print(result)
        # Extract the median ROUGE scores
        result = {key: value * 100 for key, value in result.items()}
        result = {k: round(v, 4) for k, v in result.items()}
        print(f"Epoch {epoch}:", result)

        if epoch % 5 == 4:
            # Save and upload
            accelerator.wait_for_everyone()
            unwrapped_model = accelerator.unwrap_model(model)
            unwrapped_model.save_pretrained(output_dir, save_function=accelerator.save)
            if accelerator.is_main_process:
                tokenizer.save_pretrained(output_dir)
                repo.push_to_hub(
                    commit_message=f"Training in progress epoch {epoch}", blocking=False
                )


summarizer = pipeline("summarization", model=repo_name)


def print_summary(idx):
    text = concat_dataset["test"][idx]["text"]
    title = concat_dataset["test"][idx]["summary"]
    summary = summarizer(concat_dataset["test"][idx]["text"])

    print()
    print(f"'Text: {text}'")
    print(f"'Title: {title}'")
    print(f"'Summary: {summary}'")


for i in range(10):
    print_summary(i)
