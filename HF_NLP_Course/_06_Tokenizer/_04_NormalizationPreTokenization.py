"""Based on the Huggingface Course for NLP Processing. See, https://huggingface.co/learn/nlp-course/chapter6/4."""

from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
cased_tokenizer = AutoTokenizer.from_pretrained("bert-base-cased")
print(type(tokenizer.backend_tokenizer))

# The first step of a tokenizer is to normalize the input
print(tokenizer.backend_tokenizer.normalizer.normalize_str("Héllò hôw are ü?"))
print(cased_tokenizer.backend_tokenizer.normalizer.normalize_str("Héllò hôw are ü?"))

# The next step is the pre-tokenization in which is splits the sentences into subwords
print(
    tokenizer.backend_tokenizer.pre_tokenizer.pre_tokenize_str("Hello, how are  you?")
)
print(
    cased_tokenizer.backend_tokenizer.pre_tokenizer.pre_tokenize_str(
        "Hello, how are  you?"
    )
)

# Different Tokenizers have different rules
tokenizer = AutoTokenizer.from_pretrained("gpt2")
print(
    tokenizer.backend_tokenizer.pre_tokenizer.pre_tokenize_str("Hello, how are  you?")
)

# SentencePiece based one
tokenizer = AutoTokenizer.from_pretrained("t5-small")
print(
    tokenizer.backend_tokenizer.pre_tokenizer.pre_tokenize_str("Hello, how are  you?")
)
