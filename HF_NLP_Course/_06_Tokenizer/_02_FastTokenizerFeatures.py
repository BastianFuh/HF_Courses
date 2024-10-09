"""Based on the Huggingface Course for NLP Processing. See, https://huggingface.co/learn/nlp-course/chapter6/3."""

import torch
import numpy as np
from transformers import AutoTokenizer, AutoModelForTokenClassification, pipeline
from other.util import pretty_print

tokenizer = AutoTokenizer.from_pretrained("bert-base-cased")
example = "My name is Sylvain and I work at Hugging Face in Brooklyn."
encoding = tokenizer(example)
print(type(encoding))

print(tokenizer.is_fast)
print(encoding.is_fast)

# View tokens without converting the ids
print(encoding.tokens())

# View index of word the token came from
print(encoding.word_ids())


# Difference between different tokenizers, 81s can be seen as one word or two depending
# on the tokenize strategy
print(AutoTokenizer.from_pretrained("bert-base-cased")("”81s”").tokens())
print(AutoTokenizer.from_pretrained("roberta-base")("”81s”").tokens())
print(AutoTokenizer.from_pretrained("bert-base-cased")("”81s”").word_ids())
print(AutoTokenizer.from_pretrained("roberta-base")("”81s”").word_ids())

# Get the chars belong to a word
start, end = encoding.word_to_chars(3)
print(example[start:end])


####
#
# Token classification deep look
#
####

# baseline
token_classifier = pipeline("token-classification")
pretty_print(
    token_classifier("My name is Sylvain and I work at Hugging Face in Brooklyn.")
)

# group together the tokens which correspond to the same entity
token_classifier = pipeline("token-classification", aggregation_strategy="simple")
pretty_print(
    token_classifier("My name is Sylvain and I work at Hugging Face in Brooklyn.")
)

# Get the result without pipeline
model_checkpoint = "dbmdz/bert-large-cased-finetuned-conll03-english"
tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)
model = AutoModelForTokenClassification.from_pretrained(model_checkpoint)

example = "My name is Sylvain and I work at Hugging Face in Brooklyn."
inputs = tokenizer(example, return_tensors="pt")
outputs = model(**inputs)

print(inputs["input_ids"].shape)
print(outputs.logits.shape)

probabilities = torch.nn.functional.softmax(outputs.logits, dim=-1)[0].tolist()
predictions = outputs.logits.argmax(dim=-1)[0].tolist()
print(predictions)

print(model.config.id2label)

# Create almost identical output
results = []
tokens = inputs.tokens()

for idx, pred in enumerate(predictions):
    label = model.config.id2label[pred]
    if label != "O":
        results.append(
            {"entity": label, "score": probabilities[idx][pred], "word": tokens[idx]}
        )

pretty_print(results)

inputs_with_offsets = tokenizer(example, return_offsets_mapping=True)
print(inputs_with_offsets["offset_mapping"])


# Improved version which also gets the text a token corresponds to
results = []
inputs_with_offsets = tokenizer(example, return_offsets_mapping=True)
tokens = inputs_with_offsets.tokens()
offsets = inputs_with_offsets["offset_mapping"]

for idx, pred in enumerate(predictions):
    label = model.config.id2label[pred]
    if label != "O":
        start, end = offsets[idx]
        results.append(
            {
                "entity": label,
                "score": probabilities[idx][pred],
                "word": tokens[idx],
                "start": start,
                "end": end,
            }
        )

pretty_print(results)


# Version which also groups the tokens which belong together

results = []
inputs_with_offsets = tokenizer(example, return_offsets_mapping=True)
tokens = inputs_with_offsets.tokens()
offsets = inputs_with_offsets["offset_mapping"]

idx = 0
while idx < len(predictions):
    pred = predictions[idx]
    label = model.config.id2label[pred]
    if label != "O":
        # Remove the B- or I-
        label = label[2:]
        start, _ = offsets[idx]

        # Grab all the tokens labeled with I-label
        all_scores = []
        while (
            idx < len(predictions)
            and model.config.id2label[predictions[idx]] == f"I-{label}"
        ):
            all_scores.append(probabilities[idx][pred])
            _, end = offsets[idx]
            idx += 1

        # The score is the mean of all the scores of the tokens in that grouped entity
        score = np.mean(all_scores).item()
        word = example[start:end]
        results.append(
            {
                "entity_group": label,
                "score": score,
                "word": word,
                "start": start,
                "end": end,
            }
        )
    idx += 1

pretty_print(results)
