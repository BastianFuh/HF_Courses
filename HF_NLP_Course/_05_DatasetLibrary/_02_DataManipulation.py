"""Based on the Huggingface Course for NLP Processing. See, https://huggingface.co/learn/nlp-course/chapter5/3."""

from transformers import AutoTokenizer
from datasets import load_dataset, Dataset
from pandas import DataFrame
import html

DATA_PATH = "HF_NLP_Course/_05_DatasetLibrary/dataset/"

tokenizer = AutoTokenizer.from_pretrained("bert-base-cased")


def tokenize_and_split(examples):
    result = tokenizer(
        examples["review"],
        truncation=True,
        max_length=128,
        return_overflowing_tokens=True,
    )
    # Extract mapping between new and old indices
    sample_map = result.pop("overflow_to_sample_mapping")

    for key, values in examples.items():
        result[key] = [values[i] for i in sample_map]
    return result


data_files = {
    "train": DATA_PATH + "drugsComTrain_raw.tsv",
    "test": DATA_PATH + "drugsComTest_raw.tsv",
}
drug_dataset = load_dataset("csv", data_files=data_files, delimiter="\t")
print(drug_dataset)

drug_sample = drug_dataset["train"].shuffle(seed=42).select(range(1000))
# Peek at the first few examples
print(drug_sample[:3])


# Assumption the "Unnamed:0" column is for a anonymized ID
for split in drug_dataset.keys():
    assert len(drug_dataset[split]) == len(drug_dataset[split].unique("Unnamed: 0"))


drug_dataset = drug_dataset.rename_column(
    original_column_name="Unnamed: 0", new_column_name="patient_id"
)
print(drug_dataset)

for key in drug_dataset.keys():
    amount_unique_drugs = len(drug_dataset[key].unique("drugName"))
    amount_unique_conditions = len(drug_dataset[key].unique("condition"))
    print(f"Unique drugs in {key} {amount_unique_drugs}")
    print(f"Unique conditions in {key} {amount_unique_conditions}")

# Filter entries without a condition
drug_dataset = drug_dataset.filter(lambda x: x["condition"] is not None)

# convert all conditions to lowercase
drug_dataset = drug_dataset.map(
    lambda example: {"condition": example["condition"].lower()}
)

# create new column for review length
drug_dataset = drug_dataset.map(
    lambda example: {"review_length": len(example["review"].split())}
)

print(drug_dataset["train"][0])
print(drug_dataset["train"].sort("review_length", reverse=True)[:3])

# Filter out reviews under length of 30
drug_dataset = drug_dataset.filter(lambda x: x["review_length"] > 30)
print(drug_dataset.num_rows)

# Handle html characters
drug_dataset = drug_dataset.map(
    lambda example: {"review": [html.unescape(o) for o in example["review"]]},
    batched=True,
)

# result = tokenize_and_split(drug_dataset["train"][0])
# print([len(inp) for inp in result["input_ids"]])

tokenized_dataset = drug_dataset.map(
    tokenize_and_split, batched=True, remove_columns=drug_dataset["train"].column_names
)
print(str(len(tokenized_dataset["train"])) + " " + str(len(drug_dataset["train"])))
print(tokenized_dataset)

drug_dataset.set_format("pandas")
print(drug_dataset["train"][:3])

# Create Pandas dataframe. To get the entire dataset as a different format this is
# necessary because the  set_format function only changes the __getitem__() function
train_df: DataFrame = drug_dataset["train"][:]

frequencies = (
    train_df["condition"]
    .value_counts()
    .to_frame()
    .reset_index()
    .rename(columns={"index": "condition", "condition": "frequency"})
)
print(frequencies.head())

freq_dataset = Dataset.from_pandas(frequencies)
print(freq_dataset)

#####
# Get the avg score for a drug
#####

# Sum the score per drug
test = (
    train_df.groupby("drugName", as_index=False)["rating"].sum().sort_values("drugName")
)

# Count the occurance of every drug
test["count"] = (
    train_df["drugName"]
    .value_counts()
    .to_frame()
    .sort_values("drugName")
    .reset_index()["count"]
)

# Calculate the average
test["avg_rating"] = test["rating"] / test["count"]
print(test)

drug_dataset.reset_format()

# Create a seperate test and validation dataset
drug_dataset_clean = drug_dataset["train"].train_test_split(train_size=0.8, seed=42)
# Rename the default "test" split to "validation"
drug_dataset_clean["validation"] = drug_dataset_clean.pop("test")
# Add the "test" set to our `DatasetDict`
drug_dataset_clean["test"] = drug_dataset["test"]

print(drug_dataset_clean)
