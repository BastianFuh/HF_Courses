"""Based on the Huggingface Course for NLP Processing. See, https://huggingface.co/learn/nlp-course/chapter6/8."""

from tokenizers import Regex
from datasets import load_dataset
from transformers import PreTrainedTokenizerFast
from tokenizers import (
    decoders,
    models,
    normalizers,
    pre_tokenizers,
    processors,
    trainers,
    Tokenizer,
)

dataset = load_dataset("wikitext", name="wikitext-2-raw-v1", split="train")


def get_training_corpus():
    for i in range(0, len(dataset), 1000):
        yield dataset[i : i + 1000]["text"]


with open("wikitext-2.txt", "w", encoding="utf-8") as f:
    for i in range(len(dataset)):
        f.write(dataset[i]["text"] + "\n")


####
# WordPiece
####
if False:
    # Initilize tokenizer
    tokenizer = Tokenizer(models.WordPiece(unk_token="[UNK]"))

    # For the text normalization a sequence similar to BertNormalizer
    tokenizer.normalizer = normalizers.Sequence(
        [normalizers.NFD(), normalizers.Lowercase(), normalizers.StripAccents()]
    )

    # Test the normalizer
    print(tokenizer.normalizer.normalize_str("Héllò hôw are ü?"))

    # Specify pre tokenization, also similar to the BertPreTokenizer
    # This tokenizer spilt not only on Whitespaced but also on all chracter which
    # are not letters, digits or underscores
    tokenizer.pre_tokenizer = pre_tokenizers.Whitespace()

    # Test the pre tokenizer
    print(tokenizer.pre_tokenizer.pre_tokenize_str("Let's test my pre-tokenizer."))

    # Alternativly to only split on the whitespaces
    pre_tokenizer = pre_tokenizers.WhitespaceSplit()
    print(pre_tokenizer.pre_tokenize_str("Let's test my pre-tokenizer."))

    # Or as a the first version as a sequence
    pre_tokenizer = pre_tokenizers.Sequence(
        [pre_tokenizers.WhitespaceSplit(), pre_tokenizers.Punctuation()]
    )
    print(pre_tokenizer.pre_tokenize_str("Let's test my pre-tokenizer."))

    # Use a trainer to determine the vocabulary. The special tokens need to be specified seperatly
    # because they are not in the corpus.
    special_tokens = ["[UNK]", "[PAD]", "[CLS]", "[SEP]", "[MASK]"]
    trainer = trainers.WordPieceTrainer(vocab_size=25000, special_tokens=special_tokens)

    # Train from our corpus
    tokenizer.train_from_iterator(get_training_corpus(), trainer=trainer)

    encoding = tokenizer.encode("Let's test this tokenizer.")
    print(encoding.tokens)

    # Or train by using a text file
    tokenizer.model = models.WordPiece(unk_token="[UNK]")
    tokenizer.train(["wikitext-2.txt"], trainer=trainer)

    encoding = tokenizer.encode("Let's test this tokenizer.")
    print(encoding.tokens)

    # Specify post-processing
    cls_token_id = tokenizer.token_to_id("[CLS]")
    sep_token_id = tokenizer.token_to_id("[SEP]")
    print(cls_token_id, sep_token_id)

    # Follows the BERT rules
    tokenizer.post_processor = processors.TemplateProcessing(
        single=f"[CLS]:0 $A:0 [SEP]:0",
        pair=f"[CLS]:0 $A:0 [SEP]:0 $B:1 [SEP]:1",
        special_tokens=[("[CLS]", cls_token_id), ("[SEP]", sep_token_id)],
    )

    encoding = tokenizer.encode("Let's test this tokenizer.")
    print(encoding.tokens)

    encoding = tokenizer.encode(
        "Let's test this tokenizer...",
        "on a pair of sentences.",
    )
    print(encoding.tokens)
    print(encoding.type_ids)

    # Include decoder
    tokenizer.decoder = decoders.WordPiece(prefix="##")

    print(tokenizer.decode(encoding.ids))

    # Saving and loading
    tokenizer.save("tokenizer.json")
    new_tokenizer = Tokenizer.from_file("tokenizer.json")

    # Make the tokenizer into a fast one
    wrapped_tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=tokenizer,
        # tokenizer_file="tokenizer.json", # You can load from the tokenizer file, alternatively
        unk_token="[UNK]",
        pad_token="[PAD]",
        cls_token="[CLS]",
        sep_token="[SEP]",
        mask_token="[MASK]",
    )

####
# BPE
####
if False:
    tokenizer = Tokenizer(models.BPE())

    # BPE does not use a normalizer

    tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    print(tokenizer.pre_tokenizer.pre_tokenize_str("Let's test pre-tokenization!"))

    trainer = trainers.BpeTrainer(vocab_size=25000, special_tokens=["<|endoftext|>"])
    tokenizer.train_from_iterator(get_training_corpus(), trainer=trainer)

    encoding = tokenizer.encode("Let's test this tokenizer.")
    print(encoding.tokens)

    # Trimoffset regulates if whitespace should be include in the offset of a word
    tokenizer.post_processor = processors.ByteLevel(trim_offsets=False)

    sentence = "Let's test this tokenizer."
    encoding = tokenizer.encode(sentence)
    start, end = encoding.offsets[4]
    print(sentence[start:end])

    tokenizer.decoder = decoders.ByteLevel()

    print(tokenizer.decode(encoding.ids))

    wrapped_tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=tokenizer,
        bos_token="<|endoftext|>",
        eos_token="<|endoftext|>",
    )


#####
# Unigram
#####
if True:
    tokenizer = Tokenizer(models.Unigram())

    tokenizer.normalizer = normalizers.Sequence(
        [
            normalizers.Replace("``", '"'),
            normalizers.Replace("''", '"'),
            normalizers.NFKD(),
            normalizers.StripAccents(),
            normalizers.Replace(Regex(" {2,}"), " "),
        ]
    )

    tokenizer.pre_tokenizer = pre_tokenizers.Metaspace()

    print(tokenizer.pre_tokenizer.pre_tokenize_str("Let's test the pre-tokenizer!"))

    special_tokens = ["<cls>", "<sep>", "<unk>", "<pad>", "<mask>", "<s>", "</s>"]
    trainer = trainers.UnigramTrainer(
        vocab_size=25000, special_tokens=special_tokens, unk_token="<unk>"
    )
    tokenizer.train_from_iterator(get_training_corpus(), trainer=trainer)

    encoding = tokenizer.encode("Let's test this tokenizer.")
    print(encoding.tokens)

    cls_token_id = tokenizer.token_to_id("<cls>")
    sep_token_id = tokenizer.token_to_id("<sep>")
    print(cls_token_id, sep_token_id)

    tokenizer.post_processor = processors.TemplateProcessing(
        single="$A:0 <sep>:0 <cls>:2",
        pair="$A:0 <sep>:0 $B:1 <sep>:1 <cls>:2",
        special_tokens=[("<sep>", sep_token_id), ("<cls>", cls_token_id)],
    )

    encoding = tokenizer.encode(
        "Let's test this tokenizer...", "on a pair of sentences!"
    )
    print(encoding.tokens)
    print(encoding.type_ids)

    tokenizer.decoder = decoders.Metaspace()

    wrapped_tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=tokenizer,
        bos_token="<s>",
        eos_token="</s>",
        unk_token="<unk>",
        pad_token="<pad>",
        cls_token="<cls>",
        sep_token="<sep>",
        mask_token="<mask>",
        padding_side="left",
    )
