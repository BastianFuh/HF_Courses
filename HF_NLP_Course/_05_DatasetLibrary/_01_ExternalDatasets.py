"""Based on the Huggingface Course for NLP Processing. See, https://huggingface.co/learn/nlp-course/chapter5/2
This script requires the dataset from https://github.com/crux82/squad-it/ in a folder called _05_DatasetLibrary/dataset.
"""

from datasets import load_dataset

# Loading local Datasets
data_files = {
    "train": "HF_NLP_Course/_05_DatasetLibrary/dataset/SQuAD_it-train.json",
    "test": "HF_NLP_Course/_05_DatasetLibrary/dataset/SQuAD_it-test.json",
}

squad_it_dataset = load_dataset(
    "json",
    data_files=data_files,
    field="data",
)

print(squad_it_dataset)
# print(squad_it_dataset["train"][0])

# Load remote Datasets
url = "https://github.com/crux82/squad-it/raw/master/"
data_files = {
    "train": url + "SQuAD_it-train.json.gz",
    "test": url + "SQuAD_it-test.json.gz",
}
squad_it_dataset = load_dataset("json", data_files=data_files, field="data")
print(squad_it_dataset)
