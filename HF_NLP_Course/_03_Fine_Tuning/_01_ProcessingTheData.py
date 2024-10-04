"""Based on the Huggingface Course for NLP Processing. See, https://huggingface.co/learn/nlp-course/chapter3/2"""

from transformers import (
    AutoTokenizer,
    DataCollatorWithPadding,
)
from datasets import load_dataset
from other.util import device, pretty_print


def tokenize_function(input, tokenizer):
    """
    Function handling the tokenization of the dataset. Padding is not set
    because this would apply the same for across the entire dataset.
    So it is only applied to the batch to prevent unnessary overhead.
    """
    cleaned_input = [v for k, v in input.items() if k not in ["idx", "label"]]
    return tokenizer(*cleaned_input, truncation=True)


raw_datasets = load_dataset("glue", "mrpc")
print(raw_datasets)

# A Dataset consits out of multiple sets for differente purposes
raw_train_dataset = raw_datasets["train"]
print(raw_train_dataset[0])
print(raw_train_dataset.features)

# Inside these sets the data can be accessed like a normal array
pretty_print(raw_datasets["train"][15])
pretty_print(raw_datasets["validation"][87])

checkpoint = "bert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(checkpoint)

# It also needs to be preprocessed to be used
tokenized_dataset = tokenizer(
    raw_datasets["train"]["sentence1"],
    raw_datasets["train"]["sentence2"],
    padding=True,
    truncation=True,
)

# The map function can be used to easily define the preprocessing steps and allows for batching
# It adds new rows to the sets as defined by the mapping function
tokenized_datasets = raw_datasets.map(tokenize_function, batched=True)


# This handles the Dynammic padding during the batch creation.
# These type of functions are called ´collate functions´
data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

# Take some samples and look at then
samples = tokenized_datasets["train"][:8]
samples = {
    k: v for k, v in samples.items() if k not in ["idx", "sentence1", "sentence2"]
}
print([len(x) for x in samples["input_ids"]])

batch = data_collator(samples)
pretty_print({k: v.shape for k, v in batch.items()})
