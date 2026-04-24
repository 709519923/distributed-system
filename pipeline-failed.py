import os
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

import torch
import torch.distributed as dist
import deepspeed
from transformers import AutoTokenizer, AutoModelForCausalLM
from deepspeed.pipe import PipelineModule, LayerSpec
deepspeed.init_distributed()

torch.cuda.set_device(0)   # 所有 rank 强制用同一张卡
print(f"Rank={dist.get_rank()}, Local_rank={local_rank}, Device={torch.cuda.current_device()}")
# =========================
# 1. Load model config only (no weights yet)
# =========================
model_name = "Qwen/Qwen2-7B-Instruct"

tokenizer = AutoTokenizer.from_pretrained(model_name)

hf_model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.float16
)

# =========================
# 2. Split model into 3 pipeline stages
# =========================

# NOTE:
# Qwen/LLaMA-like structure:
# model.transformer.h (list of blocks)

layers = hf_model.model.layers
num_layers = len(layers)

# simple equal split
split_size = num_layers // 3

stage0_layers = layers[:split_size]
stage1_layers = layers[split_size:2 * split_size]
stage2_layers = layers[2 * split_size:]

# =========================
# 3. Wrap each stage
# =========================

class Stage0(torch.nn.Module):
    def __init__(self, embed, layers):
        super().__init__()
        self.embed = embed
        self.layers = torch.nn.ModuleList(layers)

    def forward(self, input_ids):
        x = self.embed(input_ids)
        for layer in self.layers:
            x = layer(x)[0]
        return x


class Stage1(torch.nn.Module):
    def __init__(self, layers):
        super().__init__()
        self.layers = torch.nn.ModuleList(layers)

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)[0]
        return x


class Stage2(torch.nn.Module):
    def __init__(self, layers, norm, lm_head):
        super().__init__()
        self.layers = torch.nn.ModuleList(layers)
        self.norm = norm
        self.lm_head = lm_head

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)[0]
        x = self.norm(x)
        logits = self.lm_head(x)
        return logits


# =========================
# 4. Build PipelineModule
# =========================

pipe_model = PipelineModule(
    layers=[
        LayerSpec(Stage0, hf_model.model.embed_tokens, stage0_layers),
        LayerSpec(Stage1, stage1_layers),
        LayerSpec(Stage2, stage2_layers, hf_model.model.norm, hf_model.lm_head),
    ],
    num_stages=3,
    loss_fn=None  # inference only
)

# =========================
# 5. DeepSpeed init
# =========================

engine, _, _, _ = deepspeed.initialize(
    model=pipe_model,
    model_parameters=[p for p in pipe_model.parameters()],
    config={
        "train_batch_size": 1,
        "fp16": {
            "enabled": False
        },
        "bf16": {
            "enabled": True
        },
        "pipeline": {
            "seed_layers": True
        }
    }
)

# =========================
# 6. Inference
# =========================

prompt = "Explain KV cache in Transformer."

inputs = tokenizer(prompt, return_tensors="pt")
input_ids = inputs["input_ids"].to(engine.device)

# Pipeline requires batch dimension
outputs = engine(input_ids)

# take last token
logits = outputs[:, -1, :]
next_token = torch.argmax(logits, dim=-1)

print("Next token:", tokenizer.decode(next_token))