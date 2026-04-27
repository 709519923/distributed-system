import os
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"  # 国内镜像

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import os

# ================= 配置区域 =================
# 选项 A: 如果是本地绝对路径
LOCAL_MODEL_PATH = "/home/dingcong/models/google/gemma-2-2b-it"

# 选项 B: 如果是 Hugging Face 模型 ID (如果本地路径不存在则回退到此)
HF_MODEL_ID = "google/gemma-2-2b-it"

# 检查本地路径是否存在，决定使用哪个标识符
if os.path.exists(LOCAL_MODEL_PATH):
    model_id_or_path = LOCAL_MODEL_PATH
    print(f"✅ 使用本地路径: {LOCAL_MODEL_PATH}")
else:
    model_id_or_path = HF_MODEL_ID
    print(f"⚠️ 本地路径不存在，尝试从 HuggingFace 下载/加载: {HF_MODEL_ID}")

# ================= 核心逻辑 =================

def run_gemma_inference():
    # 1. 加载 Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_id_or_path)
    
    # Gemma 需要设置 pad_token
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # 2. 定义手动 Device Map
    # Gemma-2-2b 通常有 26 层 (model.layers.0 ~ 25)
    # 要求：CPU 只分配一个 layer (这里选第 0 层)，其余尽量放 GPU
    device_map = {
        "model.embed_tokens": "cuda:0",       # 嵌入层放 GPU (加速输入处理)
        "model.layers.0": "cpu",              # 【关键】强制第 0 层在 CPU
        **{f"model.layers.{i}": "cuda:0" for i in range(1, 26)}, # 其余 1-25 层放 GPU
        "model.norm": "cuda:0",               # 最终 Norm 放 GPU
        "lm_head": "cuda:0"                   # 输出头放 GPU
    }

    # 如果没有 GPU，将所有 cuda:0 改为 "cpu" 以允许运行，但会失去“跨设备”演示意义
    if not torch.cuda.is_available():
        print("❌ 未检测到 GPU，将所有设备映射改为 CPU (无法演示跨设备张量切换)")
        device_map = {k: "cpu" for k in device_map.keys()}
    else:
        print("✅ 检测到 GPU，将执行跨设备 (CPU <-> GPU) 张量切换演示")

    print("\n🗺️ 预设的设备映射:")
    for k, v in sorted(device_map.items()):
        print(f"  {k:<30} -> {v}")

    # 3. 加载模型并应用 Device Map
    # from_pretrained 内部会自动调用 accelerate 的 dispatch 逻辑
    print("\n⏳ 正在加载模型权重并分发到指定设备...")
    try:
        model = AutoModelForCausalLM.from_pretrained(
            model_id_or_path,
            device_map=device_map,
            torch_dtype=torch.float16,  # Gemma 推荐 fp16/bf16
            trust_remote_code=True      # 某些自定义模型可能需要
        )
    except Exception as e:
        print(f"❌ 加载失败: {e}")
        return

    # 4. 验证实际生效的映射
    print("\n🔍 实际生效的设备映射 (hf_device_map):")
    for name, device in model.hf_device_map.items():
        print(f"  {name:<30} -> {device}")

    # 5. 推理测试
    prompt = "The capital of France is"
    inputs = tokenizer(prompt, return_tensors="pt")
    
    # ⚠️ 重要：输入必须放在 device_map 中第一个执行模块的设备上
    # 在我们的 map 中，embed_tokens 在 cuda:0，所以输入要去 cuda:0
    first_layer_device = device_map["model.embed_tokens"]
    inputs = {k: v.to(first_layer_device) for k, v in inputs.items()}
    
    print(f"\n🚀 开始推理 (输入位于: {first_layer_device})...")
    print(f"   提示词: '{prompt}'")
    
    with torch.no_grad():
        outputs = model.generate(
            **inputs, 
            max_new_tokens=20, 
            do_sample=False  # 确定性输出以便复现
        )
        
    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    print(f"\n📝 生成结果:\n{generated_text}")
    
    # 6. 验证张量切换是否发生 (可选调试)
    # 如果 model.layers.0 在 CPU，而 embed_tokens 在 GPU，
    # Accelerate 会在 forward 过程中自动插入 .to('cpu') 和 .to('cuda:0')
    print("\n✅ 完成！如果未报错，说明跨设备张量路由正常工作。")

if __name__ == "__main__":
    run_gemma_inference()