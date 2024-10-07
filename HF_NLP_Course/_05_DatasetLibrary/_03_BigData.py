"""Based on the Huggingface Course for NLP Processing. See, https://huggingface.co/learn/nlp-course/chapter5/4."""

from transformers import AutoTokenizer
import psutil
import timeit
from datasets import load_dataset, DownloadConfig, interleave_datasets
from itertools import islice


data_files = "https://huggingface.co/datasets/casinca/PUBMED_title_abstracts_2019_baseline/resolve/main/PUBMED_title_abstracts_2019_baseline.jsonl.zst"
pubmed_dataset = load_dataset(
    "json",
    data_files=data_files,
    split="train",
    download_config=DownloadConfig(delete_extracted=True),
)


# Load a large dataset
if False:
    print(pubmed_dataset)

    print(pubmed_dataset[0])

    # Process.memory_info is expressed in bytes, so convert to megabytes
    print(f"RAM used: {psutil.Process().memory_info().rss / (1024 * 1024):.2f} MB")

    print(f"Number of files in dataset : {pubmed_dataset.dataset_size}")
    size_gb = pubmed_dataset.dataset_size / (1024**3)
    print(f"Dataset size (cache file) : {size_gb:.2f} GB")


if False:
    code_snippet = """batch_size = 1000

    for idx in range(0, len(pubmed_dataset), batch_size):
        _ = pubmed_dataset[idx:idx + batch_size]
    """
    time = timeit.timeit(stmt=code_snippet, number=1, globals=globals())
    print(
        f"Iterated over {len(pubmed_dataset)} examples (about {size_gb:.1f} GB) in "
        f"{time:.1f}s, i.e. {size_gb/time:.3f} GB/s"
    )

# Streaming the dataset
if True:
    pubmed_dataset_streamed = load_dataset(
        "json", data_files=data_files, split="train", streaming=True
    )
    print(next(iter(pubmed_dataset_streamed)))

    # Streaming data can be processed via the similar methods
    tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")
    tokenized_dataset = pubmed_dataset_streamed.map(lambda x: tokenizer(x["text"]))

    # They are then also processed elements wise
    # although batched=True and batch_size can be used to change this behaviour
    print(next(iter(tokenized_dataset)))

    # Shuffling is also supported but only on a set buffer_size
    shuffled_dataset = pubmed_dataset_streamed.shuffle(buffer_size=10_000, seed=42)
    print(next(iter(shuffled_dataset)))

    #####################
    # Additional commands:
    #
    #   IterableDataset.take(), Take a number of elements, similar to Dataset.select()
    #   IterableDataset.skip(), Skipps a set number of elements, usuful for creating a training and validations set
    #
    #####################
    dataset_head = pubmed_dataset_streamed.take(5)
    print(list(dataset_head))

    # Skip the first 1,000 examples and include the rest in the training set
    train_dataset = shuffled_dataset.skip(1000)
    # Take the first 1,000 examples for the validation set
    validation_dataset = shuffled_dataset.take(1000)

    # Does not work, because the-eye stopped hosting the pile dataset
    # Interleaving Datasets/ Combining them
    law_dataset_streamed = load_dataset(
        "json",
        data_files="https://mystic.the-eye.eu/public/AI/pile_preliminary_components/FreeLaw_Opinions.jsonl.zst",
        split="train",
        streaming=True,
    )
    next(iter(law_dataset_streamed))

    combined_dataset = interleave_datasets(
        [pubmed_dataset_streamed, law_dataset_streamed]
    )
    list(islice(combined_dataset, 2))
