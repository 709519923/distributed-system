import os
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

model_name = "Qwen/Qwen2-7B-Instruct"

tokenizer = AutoTokenizer.from_pretrained(model_name)

model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.bfloat16
).cuda()

model.eval()

# ====== 切分 layer ======
layers = model.model.layers
num_layers = len(layers)

# 3段划分
split1 = num_layers // 3
split2 = 2 * num_layers // 3

edge_layers = layers[:split1]
mid_layers = layers[split1:split2]
cloud_layers = layers[split2:]

# ====== 定义三个stage ======
class Stage(torch.nn.Module):
    def __init__(self, embed, layers, norm=None, lm_head=None):
        super().__init__()
        self.embed = embed
        self.layers = torch.nn.ModuleList(layers)
        self.norm = norm
        self.lm_head = lm_head

    def forward(self, x, attention_mask=None, position_ids=None):
        hidden_states = x

        for layer in self.layers:
            hidden_states = layer(
                hidden_states,
                attention_mask=attention_mask,
                position_ids=position_ids
            )[0]

        if self.norm is not None:
            hidden_states = self.norm(hidden_states)

        if self.lm_head is not None:
            hidden_states = self.lm_head(hidden_states)

        return hidden_states


# ====== 构建三个节点 ======
edge = Stage(
    embed=model.model.embed_tokens,
    layers=edge_layers
).cuda()

mid = Stage(
    embed=None,
    layers=mid_layers
).cuda()

cloud = Stage(
    embed=None,
    layers=cloud_layers,
    norm=model.model.norm,
    lm_head=model.lm_head
).cuda()


# ====== 推理（单步 forward pipeline）======
prompt = "Explain KV cache in Transformer."
inputs = tokenizer(prompt, return_tensors="pt").to("cuda")

input_ids = inputs["input_ids"]

# embedding（只在 edge 做）
hidden_states = edge.embed(input_ids)

# edge
hidden_states = edge(hidden_states)

# mid
hidden_states = mid(hidden_states)

# cloud
logits = cloud(hidden_states)

# 取最后一个 token
next_token = torch.argmax(logits[:, -1, :], dim=-1)

print("Next token:", tokenizer.decode(next_token))