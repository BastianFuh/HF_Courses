"""Based on the Huggingface Course for NLP Processing. See, https://huggingface.co/learn/nlp-course/chapter6/2."""

from datasets import load_dataset
from transformers import AutoTokenizer


def get_training_corpus(dataset):
    """Return the training corpus"""
    dataset = raw_datasets["train"]
    for start_idx in range(0, len(dataset), 1000):
        samples = dataset[start_idx : start_idx + 1000]
        yield samples["whole_func_string"]


# This can take a few minutes to load, so grab a coffee or tea while you wait!
raw_datasets = load_dataset("code_search_net", "python", trust_remote_code=True)

print(raw_datasets["train"])

print(raw_datasets["train"][123456]["whole_func_string"])

# Creating a python generator
training_corpus = (
    raw_datasets["train"][i : i + 1000]["whole_func_string"]
    for i in range(0, len(raw_datasets["train"]), 1000)
)

# Because the manual created generator can only be used once, use a function to create it.
training_corpus = get_training_corpus(raw_datasets)


# To train new tokenizer, first load the old one from the model we wan't to use
old_tokenizer = AutoTokenizer.from_pretrained("gpt2")

# See how it currently works
example = '''def add_numbers(a, b):
    """Add the two numbers `a` and `b`."""
    return a + b'''

tokens = old_tokenizer.tokenize(example)
print(tokens)

tokenizer = old_tokenizer.train_new_from_iterator(training_corpus, 52000)

tokens = tokenizer.tokenize(example)
print(tokens)


example = """class LinearLayer():
    def __init__(self, input_size, output_size):
        self.weight = torch.randn(input_size, output_size)
        self.bias = torch.zeros(output_size)

    def __call__(self, x):
        return x @ self.weights + self.bias
    """
print(tokenizer.tokenize(example))

tokenizer.save_pretrained("code-search-net-tokenizer")
