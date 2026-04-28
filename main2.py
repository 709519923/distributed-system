import os
# Set Hugging Face mirror for faster downloads in China (optional)
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

import torch
import time
from transformers import AutoModelForCausalLM, AutoTokenizer

# ======================================================================
# Configuration Section
# ======================================================================

MODEL_ID = "/home/dingcong/models/google/gemma-2-2b-it"

# Simulated Network Bandwidth (Bytes per second)
# 1 GB/s = 1,000,000,000 Bytes/s (Typical high-speed LAN/InfiniBand)
SIMULATED_BANDWIDTH_BPS = 1_000_000_000 

# Fixed overhead latency (seconds) - e.g., TCP handshake, serialization start
FIXED_LATENCY_SEC = 0.0001  # 0.1ms

# Dynamic Slicing Configuration: Number of layers to place on CPU
# Layers 0 to (NUM_CPU_LAYERS - 1) will be on CPU
# Layers NUM_CPU_LAYERS to 25 will be on GPU
NUM_CPU_LAYERS = 8  # Change this value to adjust the slicing position

TOTAL_LAYERS = 26   # Gemma-2-2b has 26 layers

# Device mapping strategy generated dynamically based on NUM_CPU_LAYERS
DEVICE_MAP = {
    "model.embed_tokens": "cuda:0",
    **{f"model.layers.{i}": "cpu" for i in range(NUM_CPU_LAYERS)},           # Force first N layers to CPU
    **{f"model.layers.{i}": "cuda:0" for i in range(NUM_CPU_LAYERS, TOTAL_LAYERS)}, # Remaining layers on GPU
    "model.norm": "cuda:0",
    "lm_head": "cuda:0"
}

# ======================================================================
# Model Loading and Preparation
# ======================================================================

print("Loading model with custom device map...")
try:
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        device_map=DEVICE_MAP,
        torch_dtype=torch.float16
    )
    
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        
    print("Model loaded successfully.")
except Exception as e:
    print(f"Failed to load model: {e}")
    exit()

# ======================================================================
# Data-Dependent Latency Simulation Hook
# ======================================================================

def simulate_data_dependent_latency_hook(module, input_args, output):
    """
    Simulates network transfer latency based on the size of the output tensor.
    Prints data volume and simulated delay.
    """
    # 1. Calculate the size of the output tensor in Bytes
    total_bytes = 0
    if isinstance(output, tuple):
        # Sum up sizes of all tensors in the tuple (e.g., hidden_states, past_key_values)
        for t in output:
            if isinstance(t, torch.Tensor):
                total_bytes += t.element_size() * t.nelement()
    elif isinstance(output, torch.Tensor):
        total_bytes = output.element_size() * output.nelement()
    
    # Convert to MB for readability
    total_mb = total_bytes / (1024 * 1024)
    
    # 2. Calculate variable delay based on bandwidth
    if SIMULATED_BANDWIDTH_BPS > 0:
        variable_delay_sec = total_bytes / SIMULATED_BANDWIDTH_BPS
    else:
        variable_delay_sec = 0
        
    # 3. Total delay
    total_delay_sec = FIXED_LATENCY_SEC + variable_delay_sec
    
    # 4. Print Debug Info
    # Get module name for logging
    module_name = module.__class__.__name__
    # Try to get layer index if available (Gemma layers have 'layer_idx' attribute usually)
    layer_idx = getattr(module, 'layer_idx', 'N/A')
    
    print(f"[Transfer Sim] Layer {layer_idx} ({module_name}): "
          f"Output Size={total_mb:.2f} MB | "
          f"Simulated Delay={total_delay_sec*1000:.2f} ms "
          f"(Fixed: {FIXED_LATENCY_SEC*1000:.2f}ms + Transfer: {variable_delay_sec*1000:.2f}ms)")

    # 5. Sleep to simulate the transfer time
    if total_delay_sec > 0:
        time.sleep(total_delay_sec)
        
    return output

# ======================================================================
# Register Hook
# ======================================================================

# Register hook on the LAST CPU-bound layer.
# This simulates the delay when the last CPU layer sends its output to the first GPU layer.
# The boundary is between layer (NUM_CPU_LAYERS - 1) and layer (NUM_CPU_LAYERS).
last_cpu_layer_index = NUM_CPU_LAYERS - 1
if last_cpu_layer_index >= 0 and last_cpu_layer_index < TOTAL_LAYERS:
    target_layer = model.model.layers[last_cpu_layer_index]
    target_layer.register_forward_hook(simulate_data_dependent_latency_hook)
    print(f"\nLatency simulation enabled:")
    print(f"  Slicing Position: After Layer {last_cpu_layer_index} (CPU) -> Layer {last_cpu_layer_index + 1} (GPU)")
    print(f"  Bandwidth: {SIMULATED_BANDWIDTH_BPS / 1_000_000:.0f} MB/s")
    print(f"  Fixed Overhead: {FIXED_LATENCY_SEC * 1000:.2f} ms")
else:
    print("\nWarning: No CPU layers configured, skipping hook registration.")

print("-" * 60)

# ======================================================================
# Inference Execution and Timing
# ======================================================================

prompt = "what's the biggest country in the world? how many area does it have?"
inputs = tokenizer(prompt, return_tensors="pt")

# IMPORTANT: Input tensors must be placed on the device of the first layer
first_layer_device = DEVICE_MAP["model.embed_tokens"]
inputs = {k: v.to(first_layer_device) for k, v in inputs.items()}

print("\nStarting inference...")
start_time = time.time()
with torch.no_grad():
    outputs = model.generate(**inputs, max_new_tokens=125, do_sample=False)
end_time = time.time()

generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
elapsed_time = end_time - start_time

print("-" * 60)
print(f"\nTotal Inference Time: {elapsed_time:.2f} seconds")
print(f"Generated Text:\n{generated_text}")
print("\nDone.")