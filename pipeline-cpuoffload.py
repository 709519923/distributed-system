import os
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

import torch
import deepspeed
from transformers import AutoTokenizer, AutoModelForCausalLM
from deepspeed.pipe import PipelineModule, LayerSpec

# 初始化DeepSpeed分布式环境
deepspeed.init_distributed()

# 设置为单卡运行
torch.cuda.set_device(0)

# 加载模型
model_name = "Qwen/Qwen2-7B-Instruct"
tokenizer = AutoTokenizer.from_pretrained(model_name)
hf_model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.float16
)

# 将模型分成3个stage
layers = hf_model.model.layers
num_layers = len(layers)
split_size = num_layers // 3

stage0_layers = layers[:split_size]
stage1_layers = layers[split_size:2 * split_size]
stage2_layers = layers[2 * split_size:]

# 定义三个stage的模块
class Stage0(torch.nn.Module):
    def __init__(self, embed, layers):
        super().__init__()
        self.embed = embed
        self.layers = torch.nn.ModuleList(layers)

    def forward(self, input_ids):
        x = self.embed(input_ids)
        for layer in self.layers:
            x = layer(x)
        return x

class Stage1(torch.nn.Module):
    def __init__(self, layers):
        super().__init__()
        self.layers = torch.nn.ModuleList(layers)

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x

class Stage2(torch.nn.Module):
    def __init__(self, layers, norm, lm_head):
        super().__init__()
        self.layers = torch.nn.ModuleList(layers)
        self.norm = norm
        self.lm_head = lm_head

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        x = self.norm(x)
        logits = self.lm_head(x)
        return logits

# 构建PipelineModule
pipe_model = PipelineModule(
    layers=[
        LayerSpec(Stage0, hf_model.model.embed_tokens, stage0_layers),
        LayerSpec(Stage1, stage1_layers),
        LayerSpec(Stage2, stage2_layers, hf_model.model.norm, hf_model.lm_head),
    ],
    num_stages=3,
    loss_fn=None
)

# 初始化DeepSpeed引擎，关键配置：
# - stage: 3 (使用ZeRO-3)
# - offload_optimizer: 将优化器状态卸载到CPU
# - offload_param: 将参数卸载到CPU
engine, _, _, _ = deepspeed.initialize(
    model=pipe_model,
    model_parameters=[p for p in pipe_model.parameters()],
    config={
        "train_batch_size": 1,
        "fp16": {"enabled": False},
        "bf16": {"enabled": True},
        "zero_optimization": {
            "stage": 3,
            "offload_optimizer": {"device": "cpu", "pin_memory": True},
            "offload_param": {"device": "cpu", "pin_memory": True},
            "contiguous_gradients": True
        },
        "pipeline": {"seed_layers": True}
    }
)

# 进行推理
prompt = "Explain KV cache in Transformer."
inputs = tokenizer(prompt, return_tensors="pt")
input_ids = inputs["input_ids"].to(engine.device)

outputs = engine(input_ids)
logits = outputs[:, -1, :]
next_token = torch.argmax(logits, dim=-1)

print("Next token:", tokenizer.decode(next_token))