"""Based on the Huggingface Course for NLP Processing. See, https://huggingface.co/learn/nlp-course/chapter5/5."""

import math
import os
import time
from pathlib import Path

import pandas as pd
import requests
from datasets import load_dataset, Dataset
from tqdm import tqdm


def fetch_issues(
    owner="huggingface",
    repo="datasets",
    num_issues=10_000,
    rate_limit=5_000,
    issues_path=Path("."),
):
    GITHUB_TOKEN = os.getenv("GITHUB_TOKEN")
    headers = {"Authorization": f"token {GITHUB_TOKEN}"}

    if Path(f"{issues_path}/{repo}-issues.jsonl").exists():
        print("Skip download")
        return

    if not issues_path.is_dir():
        issues_path.mkdir(exist_ok=True)

    batch = []
    all_issues = []
    per_page = 2000  # Number of issues to return per page
    num_pages = math.ceil(num_issues / per_page)
    base_url = "https://api.github.com/repos"

    for page in tqdm(range(num_pages)):
        # Query with state=all to get both open and closed issues
        query = f"issues?page={page}&per_page={per_page}&state=all"
        issues = requests.get(f"{base_url}/{owner}/{repo}/{query}", headers=headers)
        batch.extend(issues.json())

        if len(batch) > rate_limit and len(all_issues) < num_issues:
            all_issues.extend(batch)
            batch = []  # Flush batch for next time period
            print(f"Reached GitHub rate limit. Sleeping for one hour ...")
            time.sleep(60 * 60 + 1)

    all_issues.extend(batch)
    df = pd.DataFrame.from_records(all_issues)
    df.to_json(f"{issues_path}/{repo}-issues.jsonl", orient="records", lines=True)
    print(
        f"Downloaded all the issues for {repo}! Dataset stored at {issues_path}/{repo}-issues.jsonl"
    )


def get_comments(issue_number):
    GITHUB_TOKEN = os.getenv("GITHUB_TOKEN")
    headers = {"Authorization": f"token {GITHUB_TOKEN}"}

    url = f"https://api.github.com/repos/huggingface/datasets/issues/{issue_number}/comments"
    response = requests.get(url, headers=headers)
    return [r["body"] for r in response.json()]


# url = "https://api.github.com/repos/huggingface/datasets/issues?page=1&per_page=1"
# response = requests.get(url)

# print(response.status_code)

# print(response.json())


fetch_issues()
# The data can't be loaded directly via load_dataset because some created_at timestampt don't exist
# so it is first loaded as a DataFrame via pandas
tmp = pd.read_json("datasets-issues.jsonl", lines=True)

issues_dataset = Dataset.from_pandas(tmp)

sample = issues_dataset.shuffle(seed=666).select(range(3))

# Print out the URL and pull request entries
for url, pr in zip(sample["html_url"], sample["pull_request"]):
    print(f">> URL: {url}")
    print(f">> Pull request: {pr}\n")

issues_dataset = issues_dataset.map(
    lambda x: {"is_pull_request": False if x["pull_request"] is None else True}
)

filtered_dataset = issues_dataset.filter(lambda x: not x["is_pull_request"])
filtered_dataset.set_format("pandas")

# Calculate the average time to close an issue
df: pd.DataFrame = filtered_dataset[:]
print((df["closed_at"] - df["created_at"]).mean())


def mapping_function(x):
    try:
        return {"comments": get_comments(x["number"])}
    except TypeError:
        print(x)
        return {"comments": None}


issues_with_comments_dataset = issues_dataset.map(mapping_function)

issues_with_comments_dataset.save_to_disk("datasets-issues-with_comments.jsonl")
