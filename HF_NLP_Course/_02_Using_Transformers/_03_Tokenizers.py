"""Based on the Huggingface Course for NLP Processing. See, https://huggingface.co/learn/nlp-course/chapter2/4"""

import torch
from transformers import BertTokenizer, AutoTokenizer
from other.util import device


model = "bert-base-cased"

# Get the specific tokenizer
tokenizer = BertTokenizer.from_pretrained(model)

# Or get it automatic
tokenizer = AutoTokenizer.from_pretrained(model)

text = "I have been waiting for a HuggingFace course my whole life."

# Processing data directly.
print(tokenizer(text))


# First step: Input sequence is broken into tokens(can be i.e. words, subwords or characters)
tokens = tokenizer.tokenize(text)
print(tokens)

# Second step: Convert Tokens to IDs
ids = tokenizer.convert_tokens_to_ids(tokens)
print(ids)

# Decoding
decoded_string = tokenizer.decode(ids)
print(decoded_string)
