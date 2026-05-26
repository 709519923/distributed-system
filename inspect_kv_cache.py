"""Inspect TinyLlama KV-cache structure on one node.

This script does not use torch.distributed or NCCL. It only loads the local
model, runs one prefill forward pass with use_cache=True, and prints where the
returned KV-cache tensors live. Use it to adapt kv_cache_utils.py to the exact
Transformers version installed on the server.
"""

import argparse

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


def parse_args():
    parser = argparse.ArgumentParser(description="Inspect local TinyLlama KV-cache tensors.")
    parser.add_argument(
        "--model-dir",
        default="/home/dingcong/models/TinyLlama",
        help="Local TinyLlama directory. Default: /home/dingcong/models/TinyLlama",
    )
    parser.add_argument(
        "--prompt",
        default="Please introduce TinyLlama in one short paragraph.",
        help="Prompt used for the prefill pass.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=1,
        help="Repeat the prompt this many times. Default: 1",
    )
    parser.add_argument(
        "--max-input-tokens",
        type=int,
        default=128,
        help="Maximum input tokens for tokenizer truncation. Default: 128",
    )
    parser.add_argument(
        "--cuda-device",
        default="0",
        help="CUDA device index. Default: 0",
    )
    parser.add_argument(
        "--dtype",
        choices=("float16", "bfloat16", "float32"),
        default="float16",
        help="Model dtype. Default: float16",
    )
    parser.add_argument(
        "--max-depth",
        type=int,
        default=5,
        help="Maximum object recursion depth to print. Default: 5",
    )
    return parser.parse_args()


def resolve_dtype(name):
    if name == "float16":
        return torch.float16
    if name == "bfloat16":
        return torch.bfloat16
    return torch.float32


def tensor_mb(tensor):
    return tensor.numel() * tensor.element_size() / 1024.0 / 1024.0


def print_line(depth, text):
    print("  " * depth + text)


def describe_object(obj, name, depth, max_depth, seen_objects, seen_tensors):
    """Print a compact tree of objects and tensors inside past_key_values."""
    if obj is None:
        print_line(depth, f"{name}: None")
        return 0

    if torch.is_tensor(obj):
        tensor_id = id(obj)
        if tensor_id in seen_tensors:
            print_line(depth, f"{name}: tensor already seen")
            return 0
        seen_tensors.add(tensor_id)
        mb = tensor_mb(obj)
        print_line(
            depth,
            f"{name}: tensor shape={tuple(obj.shape)} dtype={obj.dtype} "
            f"device={obj.device} contiguous={obj.is_contiguous()} mb={mb:.4f}",
        )
        return obj.numel() * obj.element_size()

    object_id = id(obj)
    if object_id in seen_objects:
        print_line(depth, f"{name}: already seen object type={type(obj)}")
        return 0
    seen_objects.add(object_id)

    print_line(depth, f"{name}: type={type(obj)}")
    if depth >= max_depth:
        return 0

    total_bytes = 0

    if isinstance(obj, dict):
        print_line(depth, f"{name}: dict_keys={list(obj.keys())}")
        for key, value in obj.items():
            total_bytes += describe_object(
                value,
                str(key),
                depth + 1,
                max_depth,
                seen_objects,
                seen_tensors,
            )
        return total_bytes

    if isinstance(obj, (list, tuple)):
        print_line(depth, f"{name}: len={len(obj)}")
        for index, value in enumerate(obj[:4]):
            total_bytes += describe_object(
                value,
                f"{name}[{index}]",
                depth + 1,
                max_depth,
                seen_objects,
                seen_tensors,
            )
        if len(obj) > 4:
            print_line(depth + 1, f"... {len(obj) - 4} more items not printed")
        return total_bytes

    common_attrs = [
        "key_cache",
        "value_cache",
        "layers",
        "caches",
        "cache",
        "_cache",
        "keys",
        "values",
        "k_cache",
        "v_cache",
        "key_states",
        "value_states",
    ]
    for attr in common_attrs:
        if hasattr(obj, attr):
            try:
                value = getattr(obj, attr)
            except Exception as exc:
                print_line(depth + 1, f"{attr}: <error reading: {exc}>")
                continue
            total_bytes += describe_object(
                value,
                attr,
                depth + 1,
                max_depth,
                seen_objects,
                seen_tensors,
            )

    object_dict = getattr(obj, "__dict__", None)
    if isinstance(object_dict, dict):
        print_line(depth, f"{name}: __dict__.keys={list(object_dict.keys())}")
        for key, value in object_dict.items():
            total_bytes += describe_object(
                value,
                key,
                depth + 1,
                max_depth,
                seen_objects,
                seen_tensors,
            )

    return total_bytes


def try_legacy_cache(past_key_values):
    """Print legacy-cache tensor sizes if this Transformers Cache supports it."""
    to_legacy_cache = getattr(past_key_values, "to_legacy_cache", None)
    if not callable(to_legacy_cache):
        print("to_legacy_cache: not available")
        return

    try:
        legacy = to_legacy_cache()
    except Exception as exc:
        print(f"to_legacy_cache: failed: {exc}")
        return

    print(f"to_legacy_cache: type={type(legacy)} len={len(legacy) if hasattr(legacy, '__len__') else 'unknown'}")
    total_bytes = describe_object(
        legacy,
        "legacy_cache",
        depth=1,
        max_depth=4,
        seen_objects=set(),
        seen_tensors=set(),
    )
    print(f"legacy_cache_total_mb={total_bytes / 1024.0 / 1024.0:.4f}")


def main():
    args = parse_args()
    device = torch.device(f"cuda:{args.cuda_device}" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        torch.cuda.set_device(device)

    dtype = resolve_dtype(args.dtype)
    tokenizer = AutoTokenizer.from_pretrained(args.model_dir, local_files_only=True)
    if tokenizer.pad_token_id is None and tokenizer.eos_token_id is not None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        args.model_dir,
        torch_dtype=dtype,
        local_files_only=True,
        low_cpu_mem_usage=True,
    )
    model.eval().to(device)

    prompts = [args.prompt] * args.batch_size
    encoded = tokenizer(
        prompts,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=args.max_input_tokens,
    )
    input_ids = encoded["input_ids"].to(device)
    attention_mask = encoded["attention_mask"].to(device)

    print(f"device={device}")
    print(f"model_dtype={getattr(model, 'dtype', None)}")
    print(f"input_ids_shape={tuple(input_ids.shape)}")
    print(f"attention_mask_shape={tuple(attention_mask.shape)}")

    with torch.inference_mode():
        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            use_cache=True,
            return_dict=True,
        )

    past_key_values = outputs.past_key_values
    print(f"past_key_values_type={type(past_key_values)}")
    print(f"has_key_cache={hasattr(past_key_values, 'key_cache')}")
    print(f"has_value_cache={hasattr(past_key_values, 'value_cache')}")
    print(f"past_key_values_dict_keys={list(getattr(past_key_values, '__dict__', {}).keys())}")

    total_bytes = describe_object(
        past_key_values,
        "past_key_values",
        depth=0,
        max_depth=args.max_depth,
        seen_objects=set(),
        seen_tensors=set(),
    )
    print(f"recursive_cache_total_mb={total_bytes / 1024.0 / 1024.0:.4f}")
    try_legacy_cache(past_key_values)


if __name__ == "__main__":
    main()
