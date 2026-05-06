import time

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


def load_model_and_tokenizer(model_id, device_map):
    print("Loading model...")
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        device_map=device_map,
        torch_dtype=torch.float16,
    )
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    print("Model loaded.")
    return model, tokenizer


def register_latency_hooks(model, args):
    def simulate_data_dependent_latency_hook(module, input_args, output, bandwidth_bps):
        total_bytes = 0

        if isinstance(output, tuple):
            for tensor in output:
                if isinstance(tensor, torch.Tensor):
                    total_bytes += tensor.element_size() * tensor.nelement()
        elif isinstance(output, torch.Tensor):
            total_bytes = output.element_size() * output.nelement()

        total_kb = total_bytes / 1024
        variable_delay = (total_bytes / bandwidth_bps) if bandwidth_bps > 0 else 0
        total_delay = args.fixed_latency_sec + variable_delay
        layer_idx = getattr(module, "injected_layer_idx", "N/A")

        try:
            batch_size_from_input = input_args[0].shape[0]
        except Exception:
            batch_size_from_input = "?"

        print(
            f"[Transfer Sim] Layer {layer_idx} | "
            f"Batch={batch_size_from_input} | "
            f"Output={total_kb:.2f} KB ({total_bytes} B) | "
            f"Delay={total_delay * 1000:.2f} ms"
        )

        if total_delay > 0:
            time.sleep(total_delay)

        return output

    def register_hook(layer_index, bandwidth):
        layer = model.model.layers[layer_index]
        layer.injected_layer_idx = layer_index

        def hook_wrapper(module, input_args, output):
            return simulate_data_dependent_latency_hook(module, input_args, output, bandwidth)

        layer.register_forward_hook(hook_wrapper)

    register_hook(args.num_cpu_layers - 1, args.simulated_bandwidth_bps)
    register_hook(args.mid_gpu_layer - 1, args.simulated_gpu_to_gpu_bandwidth_bps)
