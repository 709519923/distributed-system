"""Inspect whether a local Qwen2-style model can reuse this pipeline code.

This script is intentionally single-node and read-only for the model directory.
It checks:
- config fields needed by layer splitting and tensor shape reasoning;
- Hugging Face module paths used by the current pipeline code;
- safetensors checkpoint keys used by the lazy loader;
- optional one-pass prefill KV-cache structure.

Run:
    python inspect_qwen2_structure.py --model-dir /path/to/Qwen2-7B
"""

import argparse
import contextlib
import json
import os
import threading
import time
import traceback
from pathlib import Path

import torch
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer


CONFIG_FIELDS = [
    "model_type",
    "architectures",
    "num_hidden_layers",
    "hidden_size",
    "intermediate_size",
    "num_attention_heads",
    "num_key_value_heads",
    "head_dim",
    "max_position_embeddings",
    "rope_theta",
    "torch_dtype",
    "tie_word_embeddings",
    "vocab_size",
]


def parse_args():
    parser = argparse.ArgumentParser(
        description="Inspect local Qwen2-7B structure for distributed pipeline compatibility."
    )
    parser.add_argument(
        "--model-dir",
        default=os.environ.get("MODEL_DIR", "/home/dingcong/models/Qwen2-7B"),
        help="Local Qwen2 model directory. Default: $MODEL_DIR or /home/dingcong/models/Qwen2-7B",
    )
    parser.add_argument(
        "--prompt",
        default="Please say hello in one short sentence.",
        help="Prompt used by the optional prefill pass.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=1,
        help="Repeat the prompt this many times for the optional prefill pass. Default: 1",
    )
    parser.add_argument(
        "--max-input-tokens",
        type=int,
        default=128,
        help="Tokenizer truncation length for the optional prefill pass. Default: 128",
    )
    parser.add_argument(
        "--device",
        choices=("auto", "cpu", "cuda"),
        default="auto",
        help="Device for the optional prefill pass. Default: auto",
    )
    parser.add_argument(
        "--cuda-device",
        default="0",
        help="CUDA device index when --device is auto/cuda. Default: 0",
    )
    parser.add_argument(
        "--dtype",
        choices=("auto", "float16", "bfloat16", "float32"),
        default="auto",
        help="Model dtype for the optional prefill pass. Default: auto",
    )
    parser.add_argument(
        "--skip-forward",
        action="store_true",
        help="Skip loading full weights and running the KV-cache prefill test.",
    )
    parser.add_argument(
        "--trust-remote-code",
        action="store_true",
        help="Pass trust_remote_code=True to Transformers if the local model requires it.",
    )
    return parser.parse_args()


def print_section(title):
    print("", flush=True)
    print("=" * 80, flush=True)
    print(title, flush=True)
    print("=" * 80, flush=True)


def current_rss_mb():
    """Return current Linux process RSS without adding a psutil dependency."""
    status_path = Path("/proc/self/status")
    if not status_path.exists():
        return None

    try:
        for line in status_path.read_text(encoding="utf-8").splitlines():
            if line.startswith("VmRSS:"):
                return int(line.split()[1]) / 1024.0
    except (OSError, ValueError, IndexError):
        return None
    return None


def runtime_status(start_time):
    """Build a small progress line without forcing CUDA initialization."""
    parts = [f"elapsed={time.perf_counter() - start_time:.1f}s"]
    rss_mb = current_rss_mb()
    if rss_mb is not None:
        parts.append(f"rss={rss_mb:.1f}MB")

    # Do not initialize CUDA merely for diagnostic output. Once CUDA is active,
    # its allocated memory is useful context while loading or running the model.
    if torch.cuda.is_initialized():
        try:
            device_index = torch.cuda.current_device()
            allocated_mb = torch.cuda.memory_allocated(device_index) / 1024.0 / 1024.0
            reserved_mb = torch.cuda.memory_reserved(device_index) / 1024.0 / 1024.0
            parts.append(f"cuda:{device_index}_allocated={allocated_mb:.1f}MB")
            parts.append(f"cuda:{device_index}_reserved={reserved_mb:.1f}MB")
        except RuntimeError:
            pass
    return "; ".join(parts)


@contextlib.contextmanager
def progress_stage(name, heartbeat_seconds=5.0):
    """Print start, periodic heartbeat, and finish messages for long stages."""
    start_time = time.perf_counter()
    stopped = threading.Event()

    def heartbeat():
        while not stopped.wait(heartbeat_seconds):
            print(f"[progress] {name}: running; {runtime_status(start_time)}", flush=True)

    print(f"[progress] {name}: started; {runtime_status(start_time)}", flush=True)
    worker = threading.Thread(target=heartbeat, name="inspection-progress", daemon=True)
    worker.start()
    try:
        yield
    except BaseException:
        print(f"[progress] {name}: failed; {runtime_status(start_time)}", flush=True)
        raise
    else:
        print(f"[progress] {name}: completed; {runtime_status(start_time)}", flush=True)
    finally:
        stopped.set()
        worker.join(timeout=heartbeat_seconds + 1.0)


def format_value(value):
    if isinstance(value, torch.dtype):
        return str(value)
    return repr(value)


def resolve_device(args):
    if args.device == "cpu":
        return torch.device("cpu")
    if args.device == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("--device cuda was requested, but torch.cuda.is_available() is False.")
        device = torch.device(f"cuda:{args.cuda_device}")
        torch.cuda.set_device(device)
        return device
    if torch.cuda.is_available():
        device = torch.device(f"cuda:{args.cuda_device}")
        torch.cuda.set_device(device)
        return device
    return torch.device("cpu")


def resolve_dtype(args, config):
    if args.dtype == "float16":
        return torch.float16
    if args.dtype == "bfloat16":
        return torch.bfloat16
    if args.dtype == "float32":
        return torch.float32

    config_dtype = getattr(config, "torch_dtype", None)
    if isinstance(config_dtype, torch.dtype):
        return config_dtype
    if isinstance(config_dtype, str) and hasattr(torch, config_dtype):
        return getattr(torch, config_dtype)
    return "auto"


def safe_get_path(obj, dotted_path):
    current = obj
    for part in dotted_path.split("."):
        if not hasattr(current, part):
            return None
        current = getattr(current, part)
    return current


def inspect_config(config):
    print_section("Config fields")
    for field in CONFIG_FIELDS:
        print(f"{field}={format_value(getattr(config, field, None))}")

    hidden_size = getattr(config, "hidden_size", None)
    num_heads = getattr(config, "num_attention_heads", None)
    if hidden_size and num_heads:
        print(f"computed_head_dim={hidden_size // num_heads}")


def instantiate_empty_model(config, trust_remote_code):
    try:
        from transformers.modeling_utils import no_init_weights

        init_context = no_init_weights()
    except ImportError:
        init_context = contextlib.nullcontext()

    # ``no_init_weights`` skips random initialization but some PyTorch versions
    # still allocate parameter storage. Meta tensors retain the whole module tree
    # and its signatures without allocating the 7B parameter payload in host RAM.
    with contextlib.ExitStack() as stack:
        stack.enter_context(init_context)
        stack.enter_context(torch.device("meta"))
        model = AutoModelForCausalLM.from_config(
            config,
            trust_remote_code=trust_remote_code,
        )
    model.eval()
    return model


def inspect_module_paths(config, trust_remote_code):
    print_section("Module path compatibility")
    try:
        with progress_stage("Build Qwen2 model skeleton on meta device"):
            model = instantiate_empty_model(config, trust_remote_code)
    except Exception:
        print("empty_model_init=failed")
        traceback.print_exc()
        return None

    checks = {
        "model.model": safe_get_path(model, "model"),
        "model.model.layers": safe_get_path(model, "model.layers"),
        "model.model.embed_tokens": safe_get_path(model, "model.embed_tokens"),
        "model.model.norm": safe_get_path(model, "model.norm"),
        "model.lm_head": safe_get_path(model, "lm_head"),
    }
    for path, value in checks.items():
        print(f"{path}_exists={value is not None}")
        if value is not None:
            print(f"{path}_type={type(value)}")

    layers = checks["model.model.layers"]
    if layers is not None:
        print(f"model_layers_len={len(layers)}")
        if len(layers) > 0:
            layer0 = layers[0]
            print(f"layer0_type={type(layer0)}")
            print(f"layer0_has_layer_idx={hasattr(layer0, 'layer_idx')}")
            self_attn = getattr(layer0, "self_attn", None)
            mlp = getattr(layer0, "mlp", None)
            print(f"layer0_self_attn_type={type(self_attn)}")
            print(f"layer0_mlp_type={type(mlp)}")
            try:
                import inspect

                print(f"model_forward_args={list(inspect.signature(model.model.forward).parameters.keys())}")
                print(f"layer_forward_args={list(inspect.signature(layer0.forward).parameters.keys())}")
            except Exception as exc:
                print(f"signature_inspection_failed={exc}")

    compatible = all(
        checks[path] is not None
        for path in (
            "model.model",
            "model.model.layers",
            "model.model.embed_tokens",
            "model.model.norm",
            "model.lm_head",
        )
    )
    print(f"llama_style_module_paths_compatible={compatible}")
    return model


def load_safetensors_weight_keys(model_dir):
    model_dir = Path(model_dir)
    index_path = model_dir / "model.safetensors.index.json"
    index_keys = set()
    indexed_shards = []
    if index_path.exists():
        with open(index_path, "r", encoding="utf-8") as f:
            index = json.load(f)
        weight_map = index.get("weight_map", {})
        index_keys = set(weight_map.keys())
        indexed_shards = sorted(set(weight_map.values()))

    safetensor_files = [model_dir / name for name in indexed_shards]
    if not safetensor_files:
        safetensor_files = sorted(model_dir.glob("*.safetensors"))
    if not safetensor_files:
        return [], [], "missing"

    try:
        from safetensors import safe_open
    except ImportError:
        print("safetensors_import=failed")
        return [], [path.name for path in safetensor_files], "safetensors_import_failed"

    keys = []
    with progress_stage("Read safetensors shard headers"):
        for shard_index, shard_path in enumerate(safetensor_files, start=1):
            print(
                f"[progress] safetensors header {shard_index}/{len(safetensor_files)}: "
                f"{shard_path.name}",
                flush=True,
            )
            with safe_open(str(shard_path), framework="pt", device="cpu") as shard:
                shard_keys = list(shard.keys())
            keys.extend(shard_keys)
            print(
                f"[progress] safetensors header {shard_index}/{len(safetensor_files)}: "
                f"found_keys={len(shard_keys)}",
                flush=True,
            )

    scanned_keys = set(keys)
    if index_keys:
        source = "index_and_header_scan"
        print(f"index_weight_key_count={len(index_keys)}")
        print(f"index_matches_scanned_keys={index_keys == scanned_keys}")
        missing_from_shards = sorted(index_keys - scanned_keys)
        unexpected_in_shards = sorted(scanned_keys - index_keys)
        print(f"index_keys_missing_from_shards={missing_from_shards[:20]}")
        print(f"shard_keys_missing_from_index={unexpected_in_shards[:20]}")
    else:
        source = "header_scan"
    return sorted(scanned_keys), [path.name for path in safetensor_files], source


def parse_layer_index(key):
    parts = key.split(".")
    if len(parts) < 4:
        return None
    if parts[0] != "model" or parts[1] != "layers":
        return None
    try:
        return int(parts[2])
    except ValueError:
        return None


def inspect_safetensors_keys(model_dir, config):
    print_section("Safetensors key compatibility")
    keys, shard_files, source = load_safetensors_weight_keys(model_dir)
    print(f"weight_key_source={source}")
    print(f"safetensors_shard_count={len(shard_files)}")
    print(f"weight_key_count={len(keys)}")
    if shard_files:
        print(f"first_shards={shard_files[:5]}")

    if not keys:
        print("lazy_load_key_check=skipped_no_safetensors_keys")
        return

    key_set = set(keys)
    required_keys = [
        "model.embed_tokens.weight",
        "model.norm.weight",
        "lm_head.weight",
    ]
    for key in required_keys:
        print(f"has_{key}={key in key_set}")

    layer_indices = sorted({idx for idx in (parse_layer_index(key) for key in keys) if idx is not None})
    print(f"checkpoint_layer_index_count={len(layer_indices)}")
    if layer_indices:
        print(f"checkpoint_layer_index_min={layer_indices[0]}")
        print(f"checkpoint_layer_index_max={layer_indices[-1]}")

    total_layers = getattr(config, "num_hidden_layers", None)
    if total_layers is not None:
        expected_layers = list(range(int(total_layers)))
        missing_layers = sorted(set(expected_layers) - set(layer_indices))
        print(f"checkpoint_layers_match_config={not missing_layers and len(layer_indices) == int(total_layers)}")
        print(f"missing_layer_indices={missing_layers[:20]}")

    for prefix in ("model.layers.0.", f"model.layers.{layer_indices[-1]}." if layer_indices else ""):
        if not prefix:
            continue
        samples = [key for key in keys if key.startswith(prefix)][:12]
        print(f"sample_keys_{prefix.rstrip('.')}={samples}")

    tie_word_embeddings = bool(getattr(config, "tie_word_embeddings", False))
    lazy_lm_head_ok = "lm_head.weight" in key_set or (
        tie_word_embeddings and "model.embed_tokens.weight" in key_set
    )
    lazy_load_likely_compatible = (
        "model.embed_tokens.weight" in key_set
        and "model.norm.weight" in key_set
        and lazy_lm_head_ok
        and bool(layer_indices)
    )
    print(f"lazy_load_key_layout_likely_compatible={lazy_load_likely_compatible}")


def tensor_mb(tensor):
    return tensor.numel() * tensor.element_size() / 1024.0 / 1024.0


def cache_layers(past_key_values):
    layers = getattr(past_key_values, "layers", None)
    if layers is not None:
        return layers
    if hasattr(past_key_values, "key_cache") and hasattr(past_key_values, "value_cache"):
        return list(zip(past_key_values.key_cache, past_key_values.value_cache))
    if isinstance(past_key_values, (list, tuple)):
        return list(past_key_values)
    return None


def layer_key_value(layer):
    if isinstance(layer, (list, tuple)) and len(layer) >= 2:
        return layer[0], layer[1]
    key = getattr(layer, "keys", None)
    value = getattr(layer, "values", None)
    if key is None:
        key = getattr(layer, "key_cache", None)
    if value is None:
        value = getattr(layer, "value_cache", None)
    return key, value


def estimate_cache_bytes(past_key_values):
    total = 0
    layers = cache_layers(past_key_values)
    if layers is None:
        return 0
    for layer in layers:
        key, value = layer_key_value(layer)
        if torch.is_tensor(key):
            total += key.numel() * key.element_size()
        if torch.is_tensor(value):
            total += value.numel() * value.element_size()
    return total


def inspect_kv_cache(past_key_values):
    print_section("KV-cache structure")
    print(f"past_key_values_type={type(past_key_values)}")
    print(f"past_key_values_dict_keys={list(getattr(past_key_values, '__dict__', {}).keys())}")
    print(f"has_key_cache={hasattr(past_key_values, 'key_cache')}")
    print(f"has_value_cache={hasattr(past_key_values, 'value_cache')}")
    print(f"has_layers={hasattr(past_key_values, 'layers')}")

    layers = cache_layers(past_key_values)
    if layers is None:
        print("cache_layers_detected=False")
        return

    print(f"cache_layers_detected=True")
    print(f"cache_layer_count={len(layers)}")
    total_bytes = estimate_cache_bytes(past_key_values)
    print(f"cache_total_mb={total_bytes / 1024.0 / 1024.0:.4f}")

    sample_indices = []
    if layers:
        sample_indices.extend(range(min(4, len(layers))))
        if len(layers) > 4:
            sample_indices.append(len(layers) - 1)
    seen = set()
    for index in sample_indices:
        if index in seen:
            continue
        seen.add(index)
        key, value = layer_key_value(layers[index])
        print(f"cache_layer_{index}_type={type(layers[index])}")
        if torch.is_tensor(key):
            print(
                f"cache_layer_{index}_key_shape={tuple(key.shape)} "
                f"dtype={key.dtype} device={key.device} contiguous={key.is_contiguous()} "
                f"mb={tensor_mb(key):.4f}"
            )
        else:
            print(f"cache_layer_{index}_key_missing_or_not_tensor={type(key)}")
        if torch.is_tensor(value):
            print(
                f"cache_layer_{index}_value_shape={tuple(value.shape)} "
                f"dtype={value.dtype} device={value.device} contiguous={value.is_contiguous()} "
                f"mb={tensor_mb(value):.4f}"
            )
        else:
            print(f"cache_layer_{index}_value_missing_or_not_tensor={type(value)}")


def inspect_forward(args, config):
    print_section("Single-node prefill forward")
    device = resolve_device(args)
    dtype = resolve_dtype(args, config)
    print(f"forward_device={device}")
    print(f"forward_dtype={dtype}")

    with progress_stage("Load tokenizer"):
        tokenizer = AutoTokenizer.from_pretrained(
            args.model_dir,
            local_files_only=True,
            trust_remote_code=args.trust_remote_code,
        )
    if tokenizer.pad_token_id is None and tokenizer.eos_token_id is not None:
        tokenizer.pad_token = tokenizer.eos_token
    print(f"tokenizer_type={type(tokenizer)}")
    print(f"tokenizer_pad_token_id={tokenizer.pad_token_id}")
    print(f"tokenizer_eos_token_id={tokenizer.eos_token_id}")

    with progress_stage("Load full Qwen2 checkpoint"):
        model = AutoModelForCausalLM.from_pretrained(
            args.model_dir,
            torch_dtype=dtype,
            local_files_only=True,
            low_cpu_mem_usage=True,
            trust_remote_code=args.trust_remote_code,
        )
    with progress_stage(f"Move full model to {device}"):
        model.eval().to(device)
    print(f"loaded_model_type={type(model)}")
    print(f"loaded_model_dtype={getattr(model, 'dtype', None)}")

    prompts = [args.prompt] * args.batch_size
    with progress_stage("Tokenize inspection prompt"):
        encoded = tokenizer(
            prompts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=args.max_input_tokens,
        )
    input_ids = encoded["input_ids"].to(device)
    attention_mask = encoded["attention_mask"].to(device)
    print(f"input_ids_shape={tuple(input_ids.shape)}")
    print(f"attention_mask_shape={tuple(attention_mask.shape)}")

    with progress_stage("Run full-model prefill and build KV cache"):
        with torch.inference_mode():
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                use_cache=True,
                return_dict=True,
            )
        if device.type == "cuda":
            torch.cuda.synchronize(device)

    logits = getattr(outputs, "logits", None)
    if torch.is_tensor(logits):
        print(f"logits_shape={tuple(logits.shape)} dtype={logits.dtype} device={logits.device}")
    with progress_stage("Inspect generated KV cache"):
        inspect_kv_cache(outputs.past_key_values)


def normalize_split(boundaries, total_layers):
    normalized = [max(0, min(total_layers, int(value))) for value in boundaries]
    normalized[0] = 0
    normalized[-1] = total_layers
    for index in range(1, len(normalized)):
        if normalized[index] <= normalized[index - 1]:
            normalized[index] = min(total_layers, normalized[index - 1] + 1)
    if normalized[-1] != total_layers:
        normalized[-1] = total_layers
    return normalized


def print_split_suggestions(config):
    print_section("Split suggestions")
    total_layers = int(getattr(config, "num_hidden_layers", 0) or 0)
    print(f"total_layers={total_layers}")
    if total_layers <= 2:
        print("split_suggestions=skipped_total_layers_too_small")
        return

    balanced = [0, total_layers // 3, (2 * total_layers) // 3, total_layers]
    ratio_based = [
        0,
        round(total_layers * 5 / 22),
        round(total_layers * 15 / 22),
        total_layers,
    ]
    balanced = normalize_split(balanced, total_layers)
    ratio_based = normalize_split(ratio_based, total_layers)
    print(f"balanced_3rank_boundaries={balanced}")
    print(f"balanced_3rank_split_layers={balanced[1:-1]}")
    print(f"tinyllama_ratio_3rank_boundaries={ratio_based}")
    print(f"tinyllama_ratio_3rank_split_layers={ratio_based[1:-1]}")


def main():
    args = parse_args()
    model_dir = Path(args.model_dir)
    print_section("Input")
    print(f"model_dir={model_dir}")
    print(f"model_dir_exists={model_dir.exists()}")
    print(f"skip_forward={args.skip_forward}")

    with progress_stage("Load model config"):
        config = AutoConfig.from_pretrained(
            args.model_dir,
            local_files_only=True,
            trust_remote_code=args.trust_remote_code,
        )
    inspect_config(config)
    inspect_module_paths(config, args.trust_remote_code)
    inspect_safetensors_keys(args.model_dir, config)
    print_split_suggestions(config)

    if args.skip_forward:
        print_section("Single-node prefill forward")
        print("forward_skipped=True")
    else:
        try:
            inspect_forward(args, config)
        except Exception:
            print("forward_failed=True")
            traceback.print_exc()


if __name__ == "__main__":
    main()
