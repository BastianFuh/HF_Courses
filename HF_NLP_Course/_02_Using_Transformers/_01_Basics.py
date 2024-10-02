"""Based on the Huggingface Course for NLP Processing. See, https://huggingface.co/learn/nlp-course/chapter2/2"""

import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from other.util import device


checkpoint = "distilbert-base-uncased-finetuned-sst-2-english"

# Get the tokenizer for a given model
tokenizer = AutoTokenizer.from_pretrained(checkpoint)

raw_inputs = [
    "I've been waiting for a HuggingFace course my whole life.",
    "I hate this so much!",
    "Today is gonna be a horrible day.",
    "You know, it was not actually that bad. \s",
]

# Use the tokenizer to create the input values
inputs = tokenizer(raw_inputs, padding=True, truncation=True, return_tensors="pt").to(
    device
)
print(f"inputs {inputs}")

# Create the model from the checkpoint
model = AutoModelForSequenceClassification.from_pretrained(checkpoint).to(device)

# Inference of the input for the model.
# The output will be in logits, which are the raw and unnormalized scores.
# The outputs are not directly converted into probabilites because the normalization and loss
# functions are combined during training.
outputs = model(**inputs)
print(f"outputs.logits.shape {outputs.logits.shape}")
print(f"outputs.logits {outputs.logits}")

# To get the prediciton of the the model the logits need to be normalized and converted to probabilities
prediction = torch.nn.functional.softmax(outputs.logits, dim=-1)
print(f"prediction {prediction}")
print(f"Prediction labels {model.config.id2label}")

for i, e in enumerate(prediction):
    max_result = torch.max(e, 0)
    print(
        f"Prediction: {raw_inputs[i]} : {model.config.id2label[max_result.indices.item()] } ({max_result.values.item()})"
    )
