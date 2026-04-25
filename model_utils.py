import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer

MODEL_NAME = "gpt2"  # 12 layers, hidden=768

def load_head(device="cuda"):
    """Embedding + first 2 transformer layers (for 3060)"""
    tokenizer = GPT2Tokenizer.from_pretrained(MODEL_NAME)
    model = GPT2LMHeadModel.from_pretrained(MODEL_NAME)
    model.eval()

    head = {
        "wte": model.transformer.wte.to(device),
        "wpe": model.transformer.wpe.to(device),
        "drop": model.transformer.drop.to(device),
        "layers": model.transformer.h[:2].to(device),
    }
    return tokenizer, head

def load_worker(device="cuda"):
    """Remaining layers + ln_f + lm_head (for A800)"""
    model = GPT2LMHeadModel.from_pretrained(MODEL_NAME)
    model.eval()

    worker = {
        "layers": model.transformer.h[2:].to(device),
        "ln_f": model.transformer.ln_f.to(device),
        "lm_head": model.lm_head.to(device),
    }
    return worker