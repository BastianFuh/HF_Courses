"""Based on the Huggingface Course for NLP Processing. See, https://huggingface.co/learn/nlp-course/chapter2/5"""

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from other.util import device

# Define model and tokenizer
checkpoint = "distilbert-base-uncased-finetuned-sst-2-english"
tokenizer = AutoTokenizer.from_pretrained(checkpoint)
model = AutoModelForSequenceClassification.from_pretrained(checkpoint).to(device)

# Process sequence into input for the model
sequence = "I've been waiting for a HuggingFace course my whole life."

tokens = tokenizer.tokenize(sequence)
print(tokens)
ids = tokenizer.convert_tokens_to_ids(tokens)

# Convert ids to tensor for input, extra dimension is added through []
input_ids = torch.tensor([ids]).to(device)
print("Input IDs:", input_ids)

output = model(input_ids)
print("Logits:", output.logits)


batched_ids = [ids, ids]
batched_inputs_ids = torch.tensor(batched_ids).to(device)

output = model(batched_inputs_ids)
print("Batched Logits:", output.logits)


# If the inputs are different lengths they need to be padded to the same length
sequence1_ids = [[200, 200, 200]]
sequence2_ids = [[200, 200]]
batched_ids = [
    [200, 200, 200],
    [200, 200, tokenizer.pad_token_id],
]

# This will not produce the same output, because the pad_token is also attentioned for
print(model(torch.tensor(sequence1_ids).to(device)).logits)
print(model(torch.tensor(sequence2_ids).to(device)).logits)
print(model(torch.tensor(batched_ids).to(device)).logits)

# This is fixed with an attention_mask
attention_mask = [
    [1, 1, 1],
    [1, 1, 0],
]
outputs = model(
    torch.tensor(batched_ids).to(device),
    attention_mask=torch.tensor(attention_mask).to(device),
)
print(outputs.logits)


######
# Example
######


sequence_1 = "I've been waiting for a HuggingFace course my whole life."
sequence_2 = "I hate this so much!"

sequence_1_ids = tokenizer.convert_tokens_to_ids(tokenizer.tokenize(sequence_1))
sequence_2_ids = tokenizer.convert_tokens_to_ids(tokenizer.tokenize(sequence_2))

attention_mask_sequence_1 = torch.ones(len(sequence_1_ids))
attention_mask_sequence_2 = torch.ones(len(sequence_2_ids))


print("\n" * 2)
print(model(torch.tensor([sequence_1_ids]).to(device)))
print(model(torch.tensor([sequence_2_ids]).to(device)))

while len(sequence_1_ids) != len(sequence_2_ids):
    if len(sequence_1_ids) > len(sequence_2_ids):
        sequence_2_ids.append(tokenizer.pad_token_id)
        attention_mask_sequence_2 = torch.cat(
            (attention_mask_sequence_2, torch.zeros(1))
        )
    else:
        sequence_1_ids.append(tokenizer.pad_token_id)
        attention_mask_sequence_1 = torch.cat(
            (attention_mask_sequence_1, torch.zeros(1))
        )

ids = torch.tensor([sequence_1_ids, sequence_2_ids]).to(device)
attention_mask = torch.stack((attention_mask_sequence_1, attention_mask_sequence_2)).to(
    device
)


print(
    model(
        torch.tensor(ids).to(device),
        attention_mask=torch.tensor(attention_mask).to(device),
    )
)
