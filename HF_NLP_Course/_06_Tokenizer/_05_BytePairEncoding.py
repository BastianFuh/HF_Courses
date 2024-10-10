"""Based on the Huggingface Course for NLP Processing. See, https://huggingface.co/learn/nlp-course/chapter6/5."""

from transformers import AutoTokenizer
from collections import defaultdict


def compute_pair_freqs(splits, word_freqs):
    pair_freqs = defaultdict(int)
    for word, freq in word_freqs.items():
        # Get the current split for the word
        split = splits[word]

        # If their is only one element the split only consists of the words and has reached its
        # maximun length
        if len(split) == 1:
            continue

        # Accumulate the frequency of the pairs contained in the current word
        for i in range(len(split) - 1):
            pair = (split[i], split[i + 1])
            pair_freqs[pair] += freq

    return pair_freqs


def merge_pair(a, b, splits, word_freqs):
    # Iterate over the words
    for word in word_freqs.keys():
        # Get the current splits
        split = splits[word]
        if len(split) == 1:
            continue

        # Merge based on a rule defined by a, b
        i = 0
        while i < len(split) - 1:
            if split[i] == a and split[i + 1] == b:
                split = split[:i] + [a + b] + split[i + 2 :]
            else:
                i += 1
        splits[word] = split
    return splits


def tokenize(text, tokenizer, merges):
    pre_tokenize_result = tokenizer._tokenizer.pre_tokenizer.pre_tokenize_str(text)
    pre_tokenized_text = [word for word, offset in pre_tokenize_result]
    splits = [[l for l in word] for word in pre_tokenized_text]
    for pair, merge in merges.items():
        for idx, split in enumerate(splits):
            i = 0
            while i < len(split) - 1:
                if split[i] == pair[0] and split[i + 1] == pair[1]:
                    split = split[:i] + [merge] + split[i + 2 :]
                else:
                    i += 1
            splits[idx] = split

    return sum(splits, [])


# This corpus is used to determine our initial vocabulary and merging rules
corpus = [
    "This is the Hugging Face Course.",
    "This chapter is about tokenization.",
    "This section shows several tokenizer algorithms.",
    "Hopefully, you will be able to understand how they are trained and generate tokens.",
]

# Get pretokenizer
tokenizer = AutoTokenizer.from_pretrained("gpt2")

# Compute the frequency of every word while the corpus is split into words.
word_freqs = defaultdict(int)

for text in corpus:
    # Pre Tokenize returns a list off (word, offset)
    words_with_offsets = tokenizer.backend_tokenizer.pre_tokenizer.pre_tokenize_str(
        text
    )
    # create list only containing the words
    new_words = [word for word, offset in words_with_offsets]

    # count frequency of words
    for word in new_words:
        word_freqs[word] += 1

print(word_freqs)


# Calculate alphabet
alphabet = set()

for word in word_freqs.keys():
    for letter in word:
        alphabet.add(letter)
alphabet = sorted(alphabet)

print(alphabet)

# add special tokens
vocab = ["<|endoftext|>"] + list(alphabet)

splits = {word: [c for c in word] for word in word_freqs.keys()}

pair_freqs = compute_pair_freqs(splits, word_freqs)

for i, key in enumerate(pair_freqs.keys()):
    print(f"{key}: {pair_freqs[key]}")
    if i >= 5:
        break

# Find the most common pair
best_pair = ""
max_freq = None

for pair, freq in pair_freqs.items():
    if max_freq is None or max_freq < freq:
        best_pair = pair
        max_freq = freq

print(best_pair, max_freq)

merges = {("Ġ", "t"): "Ġt"}
vocab.append("Ġt")

splits = merge_pair(*best_pair, splits, word_freqs)
print(splits)

vocab_length = 50

while len(vocab) < vocab_length:
    pair_freqs = compute_pair_freqs(splits, word_freqs)

    if len(pair_freqs) == 0:
        print(f"No new pairs found. Vocab max_length: {len(vocab)}")
        break

    best_pair = ""
    max_freq = None

    for pair, freq in pair_freqs.items():
        if max_freq is None or max_freq < freq:
            best_pair = pair
            max_freq = freq

    merges.update({best_pair: best_pair[0] + best_pair[1]})
    vocab.append(best_pair[0] + best_pair[1])

    splits = merge_pair(*best_pair, splits, word_freqs)

print(merges)
print(vocab)

print(tokenize("This is not a token.", tokenizer, merges))
