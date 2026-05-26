"""Create simple fixed-token-length prompt CSV files from ShareGPT.

Run this file directly from the directory where you want the CSV files:

    python prepare_sharegpt_dataset.py

It downloads Aeala/ShareGPT_Vicuna_unfiltered and writes:

    input10.csv
    input100.csv
    input1000.csv

Each file contains 512 rows and one column named "prompt". Token length is
measured with TinyLlama's tokenizer.
"""

import csv

from datasets import load_dataset
from transformers import AutoTokenizer


DATASET_NAME = "Aeala/ShareGPT_Vicuna_unfiltered"
TOKENIZER_NAME = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
TARGET_LENGTHS = (10, 100, 1000)
ROWS_PER_FILE = 512


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


def token_length(tokenizer, text):
    return len(tokenizer(text, add_special_tokens=False)["input_ids"])


def trim_to_token_length(tokenizer, text, target_length):
    token_ids = tokenizer(text, add_special_tokens=False)["input_ids"]
    if len(token_ids) < target_length:
        return None
    return tokenizer.decode(token_ids[:target_length], skip_special_tokens=True).strip()


def collect_prompts(dataset, tokenizer):
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

            trimmed = trim_to_token_length(tokenizer, prompt, target_length)
            if not trimmed:
                continue

            # Do not over-engineer exact normalization. Keep prompts close to
            # the requested token length, which is sufficient for benchmark data.
            if token_length(tokenizer, trimmed) <= target_length:
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
    tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_NAME)
    dataset = load_dataset(DATASET_NAME, split="train")
    results = collect_prompts(dataset, tokenizer)

    for target_length, prompts in results.items():
        filename = f"input{target_length}.csv"
        write_csv(filename, prompts)
        print(f"Wrote {len(prompts)} rows to {filename}")


if __name__ == "__main__":
    main()
