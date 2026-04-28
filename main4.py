import os
# Set Hugging Face mirror for faster downloads in China (optional)
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
import torch
import time
import re
from transformers import AutoModelForCausalLM, AutoTokenizer

# ======================================================================
# Configuration Section
# ======================================================================
MODEL_ID = "/home/dingcong/models/google/gemma-2-2b-it"

# Simulated Network Bandwidth (Bytes per second)
# 1 GB/s = 1,000,000,000 Bytes/s (Typical high-speed LAN/InfiniBand)
SIMULATED_BANDWIDTH_BPS = 1_000_000_000 

# NEW: Simulated GPU-to-GPU Bandwidth (e.g., NVLink or PCIe bottleneck simulation)
# Usually faster than CPU-GPU, but we can set it to test sensitivity
SIMULATED_GPU_TO_GPU_BANDWIDTH_BPS = 50_000_000_000 # 50 GB/s

# Fixed overhead latency (seconds) - e.g., TCP handshake, serialization start
FIXED_LATENCY_SEC = 0.0001  # 0.1ms

# Dynamic Slicing Configuration
NUM_CPU_LAYERS = 3  # Layers 0 to (NUM_CPU_LAYERS - 1) on CPU

# NEW: Define the split point between the two simulated GPUs
# Layers NUM_CPU_LAYERS to (MID_GPU_LAYER - 1) on Simulated GPU 0
# Layers MID_GPU_LAYER to (TOTAL_LAYERS - 1) on Simulated GPU 1
MID_GPU_LAYER = 12  # Split the remaining 25 layers roughly in half (13 on GPU0, 12 on GPU1)

TOTAL_LAYERS = 26   # Gemma-2-2b has 26 layers

# Device mapping strategy generated dynamically
# Note: Both 'simulated_gpu_0' and 'simulated_gpu_1' map to 'cuda:0' physically
DEVICE_MAP = {
    "model.embed_tokens": "cpu", # Embeddings usually stay with first layer device, but let's put on CPU if layer 0 is CPU
    
    # CPU Layers
    **{f"model.layers.{i}": "cpu" for i in range(NUM_CPU_LAYERS)},
    
    # Simulated GPU 0 Layers (Physically cuda:0)
    **{f"model.layers.{i}": "cuda:0" for i in range(NUM_CPU_LAYERS, MID_GPU_LAYER)},
    
    # Simulated GPU 1 Layers (Physically cuda:0)
    **{f"model.layers.{i}": "cuda:0" for i in range(MID_GPU_LAYER, TOTAL_LAYERS)},
    
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
def simulate_data_dependent_latency_hook(module, input_args, output, bandwidth_bps):
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
        
    # Convert to KB for better visibility of small decode steps
    total_kb = total_bytes / 1024

    # 2. Calculate variable delay based on provided bandwidth
    if bandwidth_bps > 0:
        variable_delay_sec = total_bytes / bandwidth_bps
    else:
        variable_delay_sec = 0
        
    # 3. Total delay
    total_delay_sec = FIXED_LATENCY_SEC + variable_delay_sec

    # 4. Extract Layer Index from Module Name if attribute is missing
    layer_idx = getattr(module, 'injected_layer_idx', None)
    if layer_idx is None:
        # Try to parse from name like "model.layers.0"
        match = re.search(r'layers\.(\d+)', str(module))
        if match:
            layer_idx = int(match.group(1))
        else:
            layer_idx = "N/A"
            
    # 5. Print Debug Info
    # print(f"[Transfer Sim] After Layer {layer_idx}: "
    #       f"Output Size={total_kb:.2f} KB ({total_bytes} B) | "
    #       f"Simulated Delay={total_delay_sec*1000:.2f} ms")

    # 6. Sleep to simulate the transfer time
    if total_delay_sec > 0:
        time.sleep(total_delay_sec)
        
    return output

# ======================================================================
# Register Hooks
# ======================================================================

# Helper to register hook with specific bandwidth
def register_hook(layer_index, bandwidth, description):
    if 0 <= layer_index < TOTAL_LAYERS:
        target_layer = model.model.layers[layer_index]
        target_layer.injected_layer_idx = layer_index
        
        # Create a closure to pass the specific bandwidth to the generic hook
        def hook_wrapper(module, input_args, output):
            return simulate_data_dependent_latency_hook(module, input_args, output, bandwidth)
            
        target_layer.register_forward_hook(hook_wrapper)
        print(f"  Registered Hook: {description} (After Layer {layer_index})")
    else:
        print(f"  Warning: Invalid layer index {layer_index} for {description}")

print("\nLatency simulation enabled: ")
print(f"  Architecture: CPU (Layers 0-{NUM_CPU_LAYERS-1}) -> GPU0 (Layers {NUM_CPU_LAYERS}-{MID_GPU_LAYER-1}) -> GPU1 (Layers {MID_GPU_LAYER}-{TOTAL_LAYERS-1})")
print(f"  Physical Device: All on cuda:0 (Simulated Distribution)")
print("-" * 60)

# Boundary 1: CPU -> Simulated GPU 0
# The last CPU layer sends data to the first GPU 0 layer
last_cpu_layer_index = NUM_CPU_LAYERS - 1
if last_cpu_layer_index >= 0:
    register_hook(last_cpu_layer_index, SIMULATED_BANDWIDTH_BPS, "CPU -> GPU0 Transfer")

# Boundary 2: Simulated GPU 0 -> Simulated GPU 1
# The last GPU 0 layer sends data to the first GPU 1 layer
last_gpu0_layer_index = MID_GPU_LAYER - 1
if last_gpu0_layer_index >= NUM_CPU_LAYERS:
    register_hook(last_gpu0_layer_index, SIMULATED_GPU_TO_GPU_BANDWIDTH_BPS, "GPU0 -> GPU1 Transfer")

print("-" * 60)

# ======================================================================
# Inference Execution and Timing
# ======================================================================
prompt = "Design a distributed large language model inference system under an edge–cloud collaborative architecture, where a user query is first processed on a resource-constrained edge device and then partially offloaded to a cloud server. Clearly define the model partitioning strategy across layers, the communication latency between edge and cloud, the computation latency on the edge device and on the cloud server, and the overall end-to-end latency."
inputs = tokenizer(prompt, return_tensors="pt")

# IMPORTANT: Input tensors must be placed on the device of the first layer
# Since Layer 0 is on CPU, inputs go to CPU. 
first_layer_device = DEVICE_MAP["model.layers.0"] if NUM_CPU_LAYERS > 0 else DEVICE_MAP["model.embed_tokens"]
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


