import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import deepspeed

MODEL_PATH = "/home/dingcong/models/google/gemma-2-2b-it"
DEVICE = 0  # GPU 0

# Load tokenizer and model
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)

model = AutoModelForCausalLM.from_pretrained(
    MODEL_PATH,
    torch_dtype=torch.float16,
)

# Use DeepSpeed lightweight inference (no MPI needed)
ds_engine = deepspeed.init_inference(
    model,
    mp_size=1,          # no model parallelism
    dtype=torch.float16,
    replace_method='auto',
    replace_with_kernel_inject=True
)

# Sample inference
prompt = "Once upon a time in a futuristic city"
inputs = tokenizer(prompt, return_tensors="pt").to(DEVICE)

with torch.no_grad():
    outputs = ds_engine.generate(
        **inputs,
        max_new_tokens=50
    )

generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
print("\nGenerated Text:\n", generated_text)