"""Based on the Huggingface Course for NLP Processing. See, https://huggingface.co/learn/nlp-course/chapter7/4."""

import evaluate
import numpy as np
from accelerate import Accelerator
from datasets import load_dataset
from huggingface_hub import create_repo, get_full_repo_name, Repository
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
import torch
from transformers import (
    AdamW,
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
    DataCollatorForSeq2Seq,
    pipeline,
    get_scheduler,
)

#
# Config
#

run_training = True

batch_size = 16

model_checkpoint = "Helsinki-NLP/opus-mt-en-de"
max_length = 128

lang_from = "en"
lang_to = "de"


# Objects
tokenizer = AutoTokenizer.from_pretrained(model_checkpoint, return_tensors="pt")
model = AutoModelForSeq2SeqLM.from_pretrained(model_checkpoint)
data_collator = DataCollatorForSeq2Seq(tokenizer, model=model)
metric = evaluate.load("sacrebleu")
optimizer = AdamW(model.parameters(), lr=2e-5)


#
# Functions
#
def preprocess_function(examples):
    inputs = [ex[lang_from] for ex in examples["translation"]]
    targets = [ex[lang_to] for ex in examples["translation"]]
    model_inputs = tokenizer(
        inputs, text_target=targets, max_length=max_length, truncation=True
    )
    return model_inputs


def compute_metrics(eval_preds):
    preds, labels = eval_preds
    # In case the model returns more than the prediction logits
    if isinstance(preds, tuple):
        preds = preds[0]

    decoded_preds = tokenizer.batch_decode(preds, skip_special_tokens=True)

    # Replace -100s in the labels as we can't decode them
    labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
    decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

    # Some simple post-processing
    decoded_preds = [pred.strip() for pred in decoded_preds]
    decoded_labels = [[label.strip()] for label in decoded_labels]

    result = metric.compute(predictions=decoded_preds, references=decoded_labels)
    return {"bleu": result["score"]}


def postprocess(predictions, labels):
    predictions = predictions.cpu().numpy()
    labels = labels.cpu().numpy()

    decoded_preds = tokenizer.batch_decode(predictions, skip_special_tokens=True)

    # Replace -100 in the labels as we can't decode them.
    labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
    decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

    # Some simple post-processing
    decoded_preds = [pred.strip() for pred in decoded_preds]
    decoded_labels = [[label.strip()] for label in decoded_labels]
    return decoded_preds, decoded_labels


#
# Data Processing
#

raw_datasets = load_dataset(
    "kde4",
    # This needs to be the first lexicographically wise
    lang1=lang_from if lang_from < lang_to else lang_to,
    # This needs to be the second lexicographically wise
    lang2=lang_to if lang_from < lang_to else lang_from,
    trust_remote_code=True,
)

# Create split
split_datasets = raw_datasets["train"].train_test_split(train_size=0.9, seed=20)
split_datasets["validation"] = split_datasets.pop("test")

tokenized_datasets = split_datasets.map(
    preprocess_function,
    batched=True,
    remove_columns=split_datasets["train"].column_names,
)


#
# Code
#
tokenized_datasets.set_format("torch")
train_dataloader = DataLoader(
    tokenized_datasets["train"],
    shuffle=True,
    collate_fn=data_collator,
    batch_size=batch_size,
)
eval_dataloader = DataLoader(
    tokenized_datasets["validation"], collate_fn=data_collator, batch_size=batch_size
)


accelerator = Accelerator(mixed_precision="fp16")
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


# Create and setup repo
model_name = "marian-finetuned-kde4-en-to-de-accelerate"
repo_name = get_full_repo_name(model_name)
create_repo(repo_name, exist_ok=True)

output_dir = "marian-finetuned-kde4-en-to-de-accelerate"
repo = Repository(output_dir, clone_from=repo_name)


if run_training:
    progress_bar = tqdm(range(num_training_steps))

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
        for batch in tqdm(eval_dataloader):
            with torch.no_grad():
                generated_tokens = accelerator.unwrap_model(model).generate(
                    batch["input_ids"],
                    attention_mask=batch["attention_mask"],
                    max_length=128,
                )
            labels = batch["labels"]

            # Necessary to pad predictions and labels for being gathered
            generated_tokens = accelerator.pad_across_processes(
                generated_tokens, dim=1, pad_index=tokenizer.pad_token_id
            )
            labels = accelerator.pad_across_processes(labels, dim=1, pad_index=-100)

            predictions_gathered = accelerator.gather(generated_tokens)
            labels_gathered = accelerator.gather(labels)

            decoded_preds, decoded_labels = postprocess(
                predictions_gathered, labels_gathered
            )
            metric.add_batch(predictions=decoded_preds, references=decoded_labels)

        results = metric.compute()
        print(f"epoch {epoch}, BLEU score: {results['score']:.2f}")

        # Save and upload
        accelerator.wait_for_everyone()
        unwrapped_model = accelerator.unwrap_model(model)
        unwrapped_model.save_pretrained(output_dir, save_function=accelerator.save)
        if accelerator.is_main_process:
            tokenizer.save_pretrained(output_dir)
            repo.push_to_hub(
                commit_message=f"Training in progress epoch {epoch}", blocking=False
            )


translator = pipeline("translation", model=model_checkpoint)
print(translator("Default to expanded threads"))

translator = pipeline("translation", model=repo_name)
print(translator("Default to expanded threads"))
