import os
import argparse
import time

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"


def parse_args():
    parser = argparse.ArgumentParser(description="Distributed inference benchmark runner")
    parser.add_argument("--model_id", type=str, default="/home/dingcong/models/google/gemma-2-2b-it")
    parser.add_argument("--input_file", type=str, default="input.txt")
    parser.add_argument("--output_file", type=str, default="outputs.txt")
    parser.add_argument("--summary_file", type=str, default="single_run_summary.txt")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--sequence_length", type=int, default=128)
    parser.add_argument("--num_cpu_layers", type=int, default=3)
    parser.add_argument("--mid_gpu_layer", type=int, default=12)
    parser.add_argument("--total_layers", type=int, default=26)
    parser.add_argument("--max_new_tokens", type=int, default=125)
    parser.add_argument("--simulated_bandwidth_bps", type=float, default=1_000_000_000)
    parser.add_argument("--simulated_gpu_to_gpu_bandwidth_bps", type=float, default=50_000_000_000)
    parser.add_argument("--fixed_latency_sec", type=float, default=0.0001)
    parser.add_argument(
        "--build_dataset",
        action="store_true",
        help="Build synthetic prompts to match batch_size from the input prompt pool",
    )
    return parser.parse_args()


def validate_args(args):
    if args.num_cpu_layers < 1:
        raise ValueError("--num_cpu_layers must be >= 1 so hook registration is valid.")
    if args.mid_gpu_layer < 1 or args.mid_gpu_layer > args.total_layers:
        raise ValueError("--mid_gpu_layer must be in [1, total_layers].")
    if args.num_cpu_layers > args.total_layers:
        raise ValueError("--num_cpu_layers must be <= total_layers.")


def build_device_map(args):
    return {
        "model.embed_tokens": "cpu",
        **{f"model.layers.{i}": "cpu" for i in range(args.num_cpu_layers)},
        **{f"model.layers.{i}": "cuda:0" for i in range(args.num_cpu_layers, args.mid_gpu_layer)},
        **{f"model.layers.{i}": "cuda:0" for i in range(args.mid_gpu_layer, args.total_layers)},
        "model.norm": "cuda:0",
        "lm_head": "cuda:0",
    }


def load_prompts(file_path):
    with open(file_path, "r", encoding="utf-8") as f:
        lines = [line.strip() for line in f.readlines()]
    return [line for line in lines if line]


def build_dataset_prompts(base_prompts, target_batch_size):
    if len(base_prompts) == 0:
        raise ValueError("No base prompts found to build synthetic dataset.")
    prompts = []
    while len(prompts) < target_batch_size:
        prompts.extend(base_prompts)
    return prompts[:target_batch_size]


def main():
    args = parse_args()
    validate_args(args)
    device_map = build_device_map(args)

    print("Loading model...")
    model = AutoModelForCausalLM.from_pretrained(
        args.model_id,
        device_map=device_map,
        torch_dtype=torch.float16,
    )

    tokenizer = AutoTokenizer.from_pretrained(args.model_id)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    print("Model loaded.")

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

    base_prompts = load_prompts(args.input_file)
    if len(base_prompts) == 0:
        raise ValueError("Input file is empty!")

    prompts = build_dataset_prompts(base_prompts, args.batch_size) if args.build_dataset else base_prompts

    inputs = tokenizer(
        prompts,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=args.sequence_length,
    )

    batch_size = inputs["input_ids"].shape[0]
    effective_seq_len = inputs["input_ids"].shape[1]

    first_layer_device = device_map["model.layers.0"]
    inputs = {k: v.to(first_layer_device) for k, v in inputs.items()}

    print("\nStarting batch inference...")
    start_time = time.time()

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=args.max_new_tokens,
            do_sample=False,
        )

    end_time = time.time()
    total_time = end_time - start_time

    texts = tokenizer.batch_decode(outputs, skip_special_tokens=True)

    with open(args.output_file, "w", encoding="utf-8") as f:
        f.write(f"===== Batch Size: {batch_size} =====\n")
        f.write(f"Sequence Length: {effective_seq_len}\n")
        f.write(f"NUM_CPU_LAYERS: {args.num_cpu_layers}\n")
        f.write(f"MID_GPU_LAYER: {args.mid_gpu_layer}\n")
        f.write(f"Total Time: {total_time:.6f} s\n\n")
        for i, text in enumerate(texts):
            f.write(f"[Sample {i}]\n{text}\n\n")

    with open(args.summary_file, "w", encoding="utf-8") as f:
        f.write(f"batch_size={batch_size}\n")
        f.write(f"sequence_length={effective_seq_len}\n")
        f.write(f"num_cpu_layers={args.num_cpu_layers}\n")
        f.write(f"mid_gpu_layer={args.mid_gpu_layer}\n")
        f.write(f"total_time_sec={total_time:.6f}\n")

    print("\n" + "=" * 60)
    print(f"Batch Size: {batch_size}")
    print(f"Sequence Length: {effective_seq_len}")
    print(f"NUM_CPU_LAYERS: {args.num_cpu_layers}")
    print(f"MID_GPU_LAYER: {args.mid_gpu_layer}")
    print(f"Total Inference Time: {total_time:.2f} s")
    print(f"Results saved to: {args.output_file}")
    print(f"Summary saved to: {args.summary_file}")
    print("=" * 60)
    print("Done.")


if __name__ == "__main__":
    main()
