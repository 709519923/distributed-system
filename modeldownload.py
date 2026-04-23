import os
import torch
os.environ["HUGGING_FACE_HUB_TOKEN"] = "to set"
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"  # 国内镜像

from transformers import AutoModelForCausalLM, AutoTokenizer

local_path = "/home/dingcong/models/google/gemma-2-2b-it"

# 设置本地缓存目录，下载后会自动保存到这里
os.environ["HF_HOME"] = local_path  # 可选：指定缓存根目录

tokenizer = AutoTokenizer.from_pretrained("google/gemma-2-2b-it")
model = AutoModelForCausalLM.from_pretrained(
    "google/gemma-2-2b-it",
    torch_dtype=torch.float16,
    device_map="auto"
)

# 显式保存到指定目录
model.save_pretrained(local_path)
tokenizer.save_pretrained(local_path)