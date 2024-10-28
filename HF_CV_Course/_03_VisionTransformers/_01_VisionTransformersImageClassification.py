import PIL.JpegImagePlugin
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import PIL

import torch
import torch.nn as nn

from huggingface_hub import notebook_login, repo_exists, get_full_repo_name

from datasets import load_dataset, DatasetDict, Image

from transformers import AutoImageProcessor, ViTForImageClassification

from transformers import Trainer, TrainingArguments

import evaluate

output_name = "vit-base-oxford-iiit-pets"
repo_output_name = get_full_repo_name(output_name)

if repo_exists(repo_output_name):
    model_name = repo_output_name
else:
    model_name = "google/vit-base-patch16-224"

processor = AutoImageProcessor.from_pretrained(model_name)

accuracy = evaluate.load("accuracy")


def transforms(batch):
    batch["image"] = [x.convert("RGB") for x in batch["image"]]
    inputs = processor(batch["image"], return_tensors="pt")
    inputs["labels"] = [label2id[y] for y in batch["label"]]
    return inputs


def collate_fn(batch):
    return {
        "pixel_values": torch.stack([x["pixel_values"] for x in batch]),
        "labels": torch.tensor([x["labels"] for x in batch]),
    }


def compute_metrics(eval_preds):
    logits, labels = eval_preds
    predictions = np.argmax(logits, axis=1)
    score = accuracy.compute(predictions=predictions, references=labels)
    return score


def show_predictions(rows, cols):
    samples = our_dataset["test"].shuffle().select(np.arange(rows * cols))
    processed_samples = samples.with_transform(transforms)
    predictions = trainer.predict(processed_samples).predictions.argmax(
        axis=1
    )  # predicted labels from logits
    fig = plt.figure(figsize=(cols * 4, rows * 4))
    for i in range(rows * cols):
        img = samples[i]["image"]
        prediction = predictions[i]
        label = f"label: {samples[i]['label']}\npredicted: {id2label[prediction]}"
        fig.add_subplot(rows, cols, i + 1)
        plt.imshow(img)
        plt.title(label)
        plt.axis("off")


# Load dataset
dataset = load_dataset("pcuenq/oxford-pets")

print(dataset)

from PIL import Image
import io


print(Image.open(io.BytesIO(dataset["train"][0]["image"]["bytes"])))


def convert_to_PIL(example):
    example["temp"] = Image.open(io.BytesIO(example["image"]["bytes"]))

    return example


PIL_dataset = dataset.map(convert_to_PIL, remove_columns="image")

PIL_dataset = PIL_dataset.rename_column("temp", "image")

print(PIL_dataset["train"][0])

labels = PIL_dataset["train"].unique("label")
print(len(labels), labels)

split_dataset = PIL_dataset["train"].train_test_split(
    test_size=0.2
)  # 80% train, 20% evaluation
eval_dataset = split_dataset["test"].train_test_split(
    test_size=0.5
)  # 50% validation, 50% test

# recombining the splits using a DatasetDict

our_dataset = DatasetDict(
    {
        "train": split_dataset["train"],
        "validation": eval_dataset["train"],
        "test": eval_dataset["test"],
    }
)

label2id = {c: idx for idx, c in enumerate(labels)}
id2label = {idx: c for idx, c in enumerate(labels)}

processed_dataset = our_dataset.with_transform(transforms)


model = ViTForImageClassification.from_pretrained(
    model_name,
    num_labels=len(labels),
    id2label=id2label,
    label2id=label2id,
    ignore_mismatched_sizes=True,
)

print(model)

for name, p in model.named_parameters():
    if not name.startswith("classifier"):
        p.requires_grad = False


training_args = TrainingArguments(
    output_dir="./vit-base-oxford-iiit-pets",
    per_device_train_batch_size=8,
    evaluation_strategy="epoch",
    save_strategy="epoch",
    logging_steps=100,
    num_train_epochs=5,
    learning_rate=3e-4,
    save_total_limit=2,
    remove_unused_columns=False,
    push_to_hub=True,
    report_to="tensorboard",
    load_best_model_at_end=True,
)


trainer = Trainer(
    model=model,
    args=training_args,
    data_collator=collate_fn,
    compute_metrics=compute_metrics,
    train_dataset=processed_dataset["train"],
    eval_dataset=processed_dataset["validation"],
    tokenizer=processor,
)

trainer.train()

print(trainer.evaluate(processed_dataset["test"]))

show_predictions(rows=5, cols=5)

kwargs = {
    "finetuned_from": model.config._name_or_path,
    "dataset": "pcuenq/oxford-pets",
    "tasks": "image-classification",
    "tags": ["image-classification"],
}

trainer.save_model()
trainer.push_to_hub("End of Training", **kwargs)
