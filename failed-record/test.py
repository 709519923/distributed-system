# 3 stage cannot run 4.25.2026

import os
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
import torch
import time
from transformers import AutoModelForCausalLM, AutoTokenizer


# =========================
# Config
# =========================
class Config:
    model_name = "Qwen/Qwen2-7B-Instruct"
    device = "cuda:0"
    dtype = torch.bfloat16

    stage0_layers = 4
    stage1_layers = 12

    simulate_network = True
    bandwidth_MBps = 1000


# =========================
# Network Simulator
# =========================
class NetworkSimulator:
    def __init__(self, enabled=True, bandwidth_MBps=1000):
        self.enabled = enabled
        self.bandwidth = bandwidth_MBps

    def transfer(self, x):
        if not self.enabled:
            return x

        size_mb = x.numel() * 2 / 1024 / 1024
        delay = size_mb / self.bandwidth
        time.sleep(delay)
        return x


# =========================
# Model Loader & Split
# =========================
class ModelPartition:
    def __init__(self, config: Config):
        self.config = config

        print("Loading model...")
        self.tokenizer = AutoTokenizer.from_pretrained(config.model_name)

        self.model = AutoModelForCausalLM.from_pretrained(
            "Qwen/Qwen2-7B-Instruct",
            local_files_only=True,
            torch_dtype=config.dtype,
            device_map=config.device
        )
        self.model.eval()
        print("Loaded model")
        self._split_model()

    def _split_model(self):
        layers = self.model.model.layers
        n = len(layers)

        s0 = self.config.stage0_layers
        s1 = self.config.stage1_layers

        self.stage0 = layers[:s0]
        self.stage1 = layers[s0:s0 + s1]
        self.stage2 = layers[s0 + s1:]

        self.embed = self.model.model.embed_tokens
        self.norm = self.model.model.norm
        self.lm_head = self.model.lm_head
        
        # 保存旋转位置编码层
        # self.rotary_emb = self.model.model.rotary_emb

        print(f"Total layers: {n}")
        print(f"Stage0: {len(self.stage0)}")
        print(f"Stage1: {len(self.stage1)}")
        print(f"Stage2: {len(self.stage2)}")


# =========================
# Pipeline Engine
# =========================
class PipelineEngine:
    def __init__(self, partition: ModelPartition, network: NetworkSimulator):
        self.p = partition
        self.net = network

    def forward(self, input_ids):
        timing = {}

        # ----- Embedding -----
        t0 = time.time()
        x = self.p.embed(input_ids)
        seq_len = x.shape[1]
        # 计算 position_ids，传给每一层让其内部自行计算 position_embeddings
        position_ids = torch.arange(seq_len, dtype=torch.long, device=x.device).unsqueeze(0)

        # ----- Stage 0 -----
        for layer in self.p.stage0:
            x = layer(x, position_ids=position_ids)[0]

        t1 = time.time()
        timing["stage0"] = t1 - t0
        x = self.net.transfer(x)

        # ----- Stage 1 -----
        t2 = time.time()
        for layer in self.p.stage1:
            x = layer(x, position_ids=position_ids)[0]

        t3 = time.time()
        timing["stage1"] = t3 - t2
        x = self.net.transfer(x)

        # ----- Stage 2 -----
        t4 = time.time()
        for layer in self.p.stage2:
            x = layer(x, position_ids=position_ids)[0]

        x = self.p.norm(x)
        logits = self.p.lm_head(x)

        t5 = time.time()
        timing["stage2"] = t5 - t4

        return logits, timing

# =========================
# Generator
# =========================
class Generator:
    def __init__(self, engine: PipelineEngine, tokenizer, device):
        self.engine = engine
        self.tokenizer = tokenizer
        self.device = device

    @torch.no_grad()
    def generate(self, prompt, max_new_tokens=20):
        input_ids = self.tokenizer(prompt, return_tensors="pt").input_ids.to(self.device)

        print("\n=== Generation Start ===")
        print("Prompt:", prompt)

        for step in range(max_new_tokens):
            logits, timing = self.engine.forward(input_ids)

            next_token = torch.argmax(logits[:, -1, :], dim=-1, keepdim=True)
            input_ids = torch.cat([input_ids, next_token], dim=1)

            print(f"[Step {step}] "
                  f"S0={timing['stage0']:.4f}s "
                  f"S1={timing['stage1']:.4f}s "
                  f"S2={timing['stage2']:.4f}s")

        print("=== Generation Done ===\n")

        return self.tokenizer.decode(input_ids[0], skip_special_tokens=True)


# =========================
# Main
# =========================
def main():
    config = Config()

    partition = ModelPartition(config)
    network = NetworkSimulator(config.simulate_network, config.bandwidth_MBps)
    engine = PipelineEngine(partition, network)

    generator = Generator(engine, partition.tokenizer, config.device)

    output = generator.generate(
        "Explain pipeline parallelism simply.",
        max_new_tokens=20
    )

    print("=== Output ===")
    print(output)


if __name__ == "__main__":
    main()