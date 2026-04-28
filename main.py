import os
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

import torch
import time
from transformers import AutoModelForCausalLM, AutoTokenizer

# ======================================================================
# Configuration Section
# ======================================================================

MODEL_ID = "/home/dingcong/models/google/gemma-2-2b-it"

# Simulated Network Bandwidth (in Bytes per second)
# Example: 10 Gbps Ethernet ≈ 1.25 GB/s = 1,250,000,000 Bytes/s
# Example: Slow WAN connection ≈ 100 Mbps = 12,500,000 Bytes/s
SIMULATED_BANDWIDTH_BPS = 1_000_000_000  # 1 GB/s (Simulating a fast LAN/InfiniBand)

# Fixed overhead latency (seconds) - e.g., TCP handshake, serialization start
FIXED_LATENCY_SEC = 0.0001  # 0.1ms

DEVICE_MAP = {
    "model.embed_tokens": "cuda:0",
    "model.layers.0": "cpu",          # CPU Layer (Source of transfer)
    **{f"model.layers.{i}": "cuda:0" for i in range(1, 26)}, # GPU Layers
    "model.norm": "cuda:0",
    "lm_head": "cuda:0"
}

# ======================================================================
# Model Loading
# ======================================================================

print("Loading model...")
model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID,
    device_map=DEVICE_MAP,
    torch_dtype=torch.float16
)
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
tokenizer.pad_token = tokenizer.eos_token
print("Model loaded.")

# ======================================================================
# Data-Dependent Latency Simulation Hook
# ======================================================================

def simulate_data_dependent_latency_hook(module, input_args, output):
    """
    Simulates network transfer latency based on the size of the output tensor.
    
    Formula: Total Delay = Fixed Latency + (Tensor Size in Bytes / Bandwidth)
    """
    # 1. Calculate the size of the output tensor in Bytes
    # output can be a tuple or a single tensor. We handle both.
    if isinstance(output, tuple):
        # Sum up sizes of all tensors in the tuple
        total_bytes = sum(t.element_size() * t.nelement() for t in output if isinstance(t, torch.Tensor))
    elif isinstance(output, torch.Tensor):
        total_bytes = output.element_size() * output.nelement()
    else:
        total_bytes = 0
        
    # 2. Calculate variable delay based on bandwidth
    if SIMULATED_BANDWIDTH_BPS > 0:
        variable_delay_sec = total_bytes / SIMULATED_BANDWIDTH_BPS
    else:
        variable_delay_sec = 0
        
    # 3. Total delay
    total_delay_sec = FIXED_LATENCY_SEC + variable_delay_sec
    
    # 4. Sleep to simulate the transfer time
    if total_delay_sec > 0:
        time.sleep(total_delay_sec)
        
    # Optional: Print debug info for the first few layers to verify
    # if hasattr(module, 'layer_idx') and module.layer_idx < 3:
    #     print(f"Layer {module.layer_idx}: Output Size={total_bytes/1024:.2f} KB, Simulated Delay={total_delay_sec*1000:.2f} ms")

    return output

# Register hook only on the boundary layer (CPU -> GPU transition)
# Here, Layer 0 is on CPU, its output goes to Layer 1 on GPU.
model.model.layers[0].register_forward_hook(simulate_data_dependent_latency_hook)

print(f"Data-dependent latency simulation enabled.")
print(f"  Fixed Latency: {FIXED_LATENCY_SEC*1000:.2f} ms")
print(f"  Simulated Bandwidth: {SIMULATED_BANDWIDTH_BPS / 1_000_000:.2f} MB/s")

# ======================================================================
# Inference Execution
# ======================================================================

prompt = "The capital of France is"
inputs = tokenizer(prompt, return_tensors="pt").to("cuda:0")

print("\nStarting inference with data-dependent delays...")
start_time = time.time()
with torch.no_grad():
    outputs = model.generate(**inputs, max_new_tokens=20, do_sample=False)
end_time = time.time()

generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
elapsed_time = end_time - start_time

print(f"\nTotal Inference Time: {elapsed_time:.2f} seconds")
print(f"Generated Text:\n{generated_text}")
print("\nDone.")