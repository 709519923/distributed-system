import os
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

import random
import csv
from datasets import load_dataset
from transformers import AutoTokenizer
from tqdm import tqdm

# =========================
# Config
# =========================
MODEL_PATH = "/home/dingcong/models/google/gemma-2-2b-it"
TARGET_LENGTHS = TARGET_LENGTHS = [64, 128, 256, 512, 1024]
NUM_SAMPLES = 1000
TOLERANCE = 0.1

OUTPUT_PREFIX = "input_"

# =========================
# Load tokenizer (LOCAL)
# =========================
print("Loading Gemma tokenizer from local path...")
tokenizer = AutoTokenizer.from_pretrained(
    MODEL_PATH,
    local_files_only=True   # ✅关键：不走网络
)

# =========================
# Chat format (Gemma必须)
# =========================
def format_chat(text):
    messages = [
        {"role": "user", "content": text}
    ]
    return tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )

def get_len(text):
    return len(tokenizer(text).input_ids)

# =========================
# Load dataset
# =========================
print("Loading ShareGPT dataset...")
dataset = load_dataset(
    "Aeala/ShareGPT_Vicuna_unfiltered",
    split="train"
)

# =========================
# Extract prompt
# =========================
def extract_prompt(sample, max_turns=2):
    conversations = sample["conversations"]
    text = ""
    for turn in conversations[:max_turns]:
        text += turn["value"].strip() + " "
    return text.strip()

# =========================
# Collect base prompts
# =========================
print("Extracting base prompts...")
base_prompts = []

for sample in tqdm(dataset):
    try:
        raw = extract_prompt(sample)
        if raw:
            prompt = format_chat(raw)
            base_prompts.append(prompt)
    except:
        continue

print(f"Collected {len(base_prompts)} base prompts")

# =========================
# Expand function
# =========================
def expand_to_length(text, target_len):
    while get_len(text) < target_len:
        extra = random.choice(base_prompts)
        text += " " + extra
    return text

# =========================
# Build dataset
# =========================
def build_dataset(target_len):
    lower = int(target_len * (1 - TOLERANCE))
    upper = int(target_len * (1 + TOLERANCE))

    selected = []

    print(f"\nBuilding dataset for ~{target_len}")

    # Step 1: filter
    for text in tqdm(base_prompts):
        l = get_len(text)
        if lower <= l <= upper:
            selected.append(text)
        if len(selected) >= NUM_SAMPLES:
            break

    print(f"Filtered: {len(selected)}")

    # Step 2: expand
    while len(selected) < NUM_SAMPLES:
        base = random.choice(base_prompts)
        new_text = expand_to_length(base, target_len)

        l = get_len(new_text)
        if lower <= l <= upper:
            selected.append(new_text)

    print(f"Final: {len(selected)}")

    return selected

# =========================
# Save CSV
# =========================
def save_csv(filename, data):
    with open(filename, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["prompt"])
        for line in data:
            writer.writerow([line.replace("\n", " ")])

# =========================
# Main
# =========================
if __name__ == "__main__":
    for target_len in TARGET_LENGTHS:
        data = build_dataset(target_len)
        filename = f"{OUTPUT_PREFIX}{target_len}.csv"
        save_csv(filename, data)
        print(f"Saved: {filename}")