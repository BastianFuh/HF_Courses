"""Based on the Huggingface Course for NLP Processing. See, https://huggingface.co/learn/nlp-course/chapter2/3"""

import torch
from transformers import BertConfig, BertModel
from other.util import device

# Model defintion
config = BertConfig()
print(config)
model = BertModel.from_pretrained("bert-base-cased").to(device)

# Save a model
model.save_pretrained("./")

# Using models for inference

# Converting the input sequence to input ids
sequences = ["Hello!", "Cool.", "Nice!"]
encoded_sequences = [
    [101, 7592, 999, 102],
    [101, 4658, 1012, 102],
    [101, 3835, 999, 102],
]

model_inputs = torch.tensor(encoded_sequences).to(device)
print(model(model_inputs))
