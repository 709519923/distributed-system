import csv
import os


def load_prompts(file_path):
    # Read prompts from .txt (one prompt per line) or .csv (column: prompt).
    ext = os.path.splitext(file_path)[1].lower()

    if ext == ".csv":
        prompts = []
        with open(file_path, "r", encoding="utf-8", newline="") as f:
            reader = csv.DictReader(f)
            if "prompt" not in (reader.fieldnames or []):
                raise ValueError("CSV must contain a 'prompt' column.")
            for row in reader:
                value = (row.get("prompt") or "").strip()
                if value:
                    prompts.append(value)
        return prompts

    with open(file_path, "r", encoding="utf-8") as f:
        lines = [line.strip() for line in f.readlines()]
    return [line for line in lines if line]


def build_dataset_prompts(base_prompts, target_batch_size):
    # Repeat prompt pool to create exactly `target_batch_size` samples.
    if len(base_prompts) == 0:
        raise ValueError("No base prompts found to build synthetic dataset.")
    prompts = []
    while len(prompts) < target_batch_size:
        prompts.extend(base_prompts)
    return prompts[:target_batch_size]


def build_tokenized_inputs(tokenizer, prompts, sequence_length):
    # Build padded batch tensors and truncate each sample to `sequence_length`.
    return tokenizer(
        prompts,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=sequence_length,
    )


def chunk_prompts(prompts, batch_size):
    # Split full dataset into mini-batches.
    if batch_size <= 0:
        raise ValueError("batch_size must be > 0.")
    for i in range(0, len(prompts), batch_size):
        yield prompts[i : i + batch_size]
