import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

# ===================== 1. 基础配置 =====================
model_name = "Qwen/Qwen2-7B-Instruct"
# 三节点设备定义（单A800环境）
DEVICE_EDGE = torch.device("cpu")       # 最弱节点：仅嵌入层，无计算
DEVICE_EDGE_NODE = torch.device("cuda:0")# 边缘节点：1份算力
DEVICE_CLOUD = torch.device("cuda:0")   # 云节点：4份算力

# ===================== 2. 加载Tokenizer =====================
tokenizer = AutoTokenizer.from_pretrained(model_name)

# ===================== 3. 核心：新版层切分Device Map =====================
# Edge：仅embed层 | 边缘：6层(1) | 云：26层(4)
device_map = {
    # ========== Edge节点 (CPU)：仅词嵌入层（最弱，无任何计算层）==========
    "embed_tokens": DEVICE_EDGE_NODE,

    # ========== 边缘节点 (GPU)：1份 → 6层 Decoder ==========
    "layers.0": DEVICE_EDGE_NODE,
    "layers.1": DEVICE_EDGE_NODE,
    "layers.2": DEVICE_EDGE_NODE,
    "layers.3": DEVICE_EDGE_NODE,
    "layers.4": DEVICE_EDGE_NODE,
    "layers.5": DEVICE_EDGE_NODE,

    # ========== 云节点 (GPU)：4份 → 26层 Decoder + 输出层 ==========
    "layers.6": DEVICE_CLOUD,
    "layers.7": DEVICE_CLOUD,
    "layers.8": DEVICE_CLOUD,
    "layers.9": DEVICE_CLOUD,
    "layers.10": DEVICE_CLOUD,
    "layers.11": DEVICE_CLOUD,
    "layers.12": DEVICE_CLOUD,
    "layers.13": DEVICE_CLOUD,
    "layers.14": DEVICE_CLOUD,
    "layers.15": DEVICE_CLOUD,
    "layers.16": DEVICE_CLOUD,
    "layers.17": DEVICE_CLOUD,
    "layers.18": DEVICE_CLOUD,
    "layers.19": DEVICE_CLOUD,
    "layers.20": DEVICE_CLOUD,
    "layers.21": DEVICE_CLOUD,
    "layers.22": DEVICE_CLOUD,
    "layers.23": DEVICE_CLOUD,
    "layers.24": DEVICE_CLOUD,
    "layers.25": DEVICE_CLOUD,
    "layers.26": DEVICE_CLOUD,
    "layers.27": DEVICE_CLOUD,
    "layers.28": DEVICE_CLOUD,
    "layers.29": DEVICE_CLOUD,
    "layers.30": DEVICE_CLOUD,
    "layers.31": DEVICE_CLOUD,
    "norm": DEVICE_CLOUD,
    "lm_head": DEVICE_CLOUD
}

# ===================== 4. 加载切分模型 =====================
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    device_map=device_map,
    torch_dtype=torch.bfloat16,
    trust_remote_code=True
)

# ===================== 5. 三节点推理流程（入口=最弱Edge）=====================
def pipeline_inference(prompt: str, max_new_tokens=256):
    print("="*60)
    print(f"【Edge节点（CPU，最弱）】请求入口，仅做嵌入处理：{prompt}")
    
    # 输入编码（在Edge节点CPU上完成，无计算压力）
    inputs = tokenizer([prompt], return_tensors="pt").to(DEVICE_EDGE)
    
    # 数据流：Edge(embed) → 边缘节点(6层) → 云节点(26层+生成)
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=True, top_p=0.8, temperature=0.7
        )
    
    # 云节点解码输出
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    print(f"【云节点】生成完成")
    print("回复：", response)
    print("="*60)
    return response

# ===================== 测试 =====================
if __name__ == "__main__":
    pipeline_inference("你好，请简单介绍人工智能")