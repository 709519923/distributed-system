"""Create simple fixed-token-length prompt CSV files from ShareGPT.

Run this file directly from the directory where you want the CSV files:

    python prepare_sharegpt_dataset.py

It downloads Aeala/ShareGPT_Vicuna_unfiltered and writes:

    input10.csv
    input100.csv
    input1000.csv

Each file contains 512 rows and one column named "prompt". For this data
preparation script, token length is measured by simple whitespace tokenization
so the script does not need to download or load any model tokenizer.
"""

import csv
import os

from datasets import load_dataset


DATASET_NAME = "Aeala/ShareGPT_Vicuna_unfiltered"
TARGET_LENGTHS = (10, 100, 1000)
ROWS_PER_FILE = 512
HF_MIRROR_ENDPOINT = "https://hf-mirror.com"


def first_user_prompt(row):
    conversations = row.get("conversations", [])
    if isinstance(conversations, list):
        for message in conversations:
            if not isinstance(message, dict):
                continue
            role = str(message.get("from", "")).lower()
            text = str(message.get("value", "")).strip()
            if role in ("human", "user") and text:
                return text
    return None


def text_tokens(text):
    """Return simple text tokens without depending on a model tokenizer."""
    return text.replace("\n", " ").split()


def token_length(text):
    return len(text_tokens(text))


def trim_to_token_length(text, target_length):
    tokens = text_tokens(text)
    if len(tokens) < target_length:
        return None
    return " ".join(tokens[:target_length]).strip()


def collect_prompts(dataset):
    results = {target_length: [] for target_length in TARGET_LENGTHS}

    for row in dataset:
        if all(len(prompts) >= ROWS_PER_FILE for prompts in results.values()):
            break

        prompt = first_user_prompt(row)
        if not prompt:
            continue

        for target_length in TARGET_LENGTHS:
            if len(results[target_length]) >= ROWS_PER_FILE:
                continue

            trimmed = trim_to_token_length(prompt, target_length)
            if not trimmed:
                continue

            if token_length(trimmed) == target_length:
                results[target_length].append(trimmed)

    for target_length, prompts in results.items():
        if len(prompts) < ROWS_PER_FILE:
            raise RuntimeError(
                f"Only collected {len(prompts)} rows for input{target_length}.csv"
            )

    return results


def write_csv(filename, prompts):
    with open(filename, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["prompt"])
        writer.writeheader()
        for prompt in prompts:
            writer.writerow({"prompt": prompt})


def main():
    os.environ.setdefault("HF_ENDPOINT", HF_MIRROR_ENDPOINT)
    hf_token = os.environ.get("HF_TOKEN")

    dataset = load_dataset(DATASET_NAME, split="train", token=hf_token)
    results = collect_prompts(dataset)

    for target_length, prompts in results.items():
        filename = f"input{target_length}.csv"
        write_csv(filename, prompts)
        print(f"Wrote {len(prompts)} rows to {filename}")


if __name__ == "__main__":
    main()
