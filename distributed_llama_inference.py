import torch
from transformers import LlamaForCausalLM, LlamaTokenizer
from transformers import AutoTokenizer, AutoModelForCausalLM
import deepspeed
import threading
import queue
import time

# Model configuration
model_path = "/home/dingcong/models/google/gemma-2-2b-it"
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForCausalLM.from_pretrained(
    model_path,
    torch_dtype=torch.float16
)

# Initialize DeepSpeed for inference optimization
model = deepspeed.init_inference(
    model,
    dtype=torch.float16,
    replace_with_kernel_inject=True,  # Use DeepSpeed kernel injection for speed
    max_out_tokens=512,  # Adjust as needed
)

model.to("cuda")

# Split layers into 3 groups for 3 "machines"
num_layers = len(model.model.layers)
group_sizes = [num_layers // 3, num_layers // 3, num_layers - 2 * (num_layers // 3)]
groups = []
start = 0
for size in group_sizes:
    groups.append(list(range(start, start + size)))
    start += size

# Function to apply a group of layers
def apply_layers(hidden, attention_mask, layer_indices, delay=0):
    for i in layer_indices:
        layer = model.model.layers[i]
        hidden = layer(hidden, attention_mask=attention_mask)[0]
    time.sleep(delay)  # Simulate different compute powers
    return hidden

# Stage functions
def stage0(input_ids, attention_mask, delay=0):
    hidden = model.model.embed_tokens(input_ids)
    hidden = apply_layers(hidden, attention_mask, groups[0], delay)
    return hidden

def stage1(hidden, attention_mask, delay=0):
    hidden = apply_layers(hidden, attention_mask, groups[1], delay)
    return hidden

def stage2(hidden, attention_mask, delay=0):
    hidden = apply_layers(hidden, attention_mask, groups[2], delay)
    hidden = model.model.norm(hidden)
    logits = model.lm_head(hidden)
    return logits

# Queues for inter-thread communication
q01 = queue.Queue()
q12 = queue.Queue()
q23 = queue.Queue()
output_queue = queue.Queue()

# Worker threads for each "machine"
def worker0(delay):
    while True:
        data = q01.get()
        if data == 'STOP':
            break
        input_ids, attention_mask = data
        hidden = stage0(input_ids, attention_mask, delay)
        q12.put((hidden, attention_mask))

def worker1(delay):
    while True:
        data = q12.get()
        if data == 'STOP':
            break
        hidden, attention_mask = data
        hidden = stage1(hidden, attention_mask, delay)
        q23.put((hidden, attention_mask))

def worker2(delay):
    while True:
        data = q23.get()
        if data == 'STOP':
            break
        hidden, attention_mask = data
        logits = stage2(hidden, attention_mask, delay)
        output_queue.put(logits)

# Delays to simulate different compute powers (in seconds)
delays = [0.0, 0.05, 0.1]  # Machine 0: fastest, Machine 1: medium, Machine 2: slowest

# Start threads
t0 = threading.Thread(target=worker0, args=(delays[0],))
t1 = threading.Thread(target=worker1, args=(delays[1],))
t2 = threading.Thread(target=worker2, args=(delays[2],))
t0.start()
t1.start()
t2.start()

# Example inference with generation
prompt = "Explain KV cache in Transformer."
inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
input_ids = inputs['input_ids']
attention_mask = inputs['attention_mask']

max_new_tokens = 50
generated = input_ids.clone()

for _ in range(max_new_tokens):
    q01.put((generated, attention_mask))
    logits = output_queue.get()
    next_token = torch.argmax(logits[:, -1:], dim=-1)
    generated = torch.cat([generated, next_token], dim=1)
    attention_mask = torch.cat([attention_mask, torch.ones_like(next_token)], dim=1)

# Decode and print the result
output_text = tokenizer.decode(generated[0], skip_special_tokens=True)
print(output_text)

# Stop threads
q01.put('STOP')
q12.put('STOP')
q23.put('STOP')
t0.join()
t1.join()
t2.join()