"""
TinyLlama two-node pipeline inference demo.

This script is intentionally written as a small, explicit pipeline instead of
using a high-level serving framework. The goal is to show how a causal language
model can be split across two NCCL ranks:

- Rank 0 owns input/output work: CSV prompts, tokenizer, token generation loop,
  embedding, and the first decoder layers.
- Rank 1 owns the later decoder layers, final norm, and lm_head.
- Rank 0 sends intermediate hidden states to Rank 1.
- Rank 1 sends the selected next token back to Rank 0.

Important note: each rank currently loads the full Hugging Face checkpoint first,
then removes the layers it does not execute. That keeps the code simple and easy
to verify. It means startup still reads the full model on both nodes.

Use --lazy-load to switch to the newer path where each rank only reads the
safetensors weights it actually needs.
"""

import argparse
import contextlib
import csv
import inspect
import json
import os
import socket
from datetime import timedelta
from pathlib import Path

import torch
import torch.distributed as dist
import torch.nn as nn
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer


# Default distributed rendezvous address. Rank 1 connects to this address when
# torch.distributed initializes the process group. It can also be overridden by
# --init-method or the DIST_INIT_METHOD environment variable.
DEFAULT_INIT_METHOD = "tcp://10.50.0.57:29500"

# Rank 1 starts from this decoder layer index. With the default value 5:
# Rank 0 runs layers 0, 1, 2, 3, 4; Rank 1 runs layers 5 ... last.
DEFAULT_SPLIT_LAYER = 5


# -----------------------------------------------------------------------------
# Command-line configuration
# -----------------------------------------------------------------------------

def parse_args():
    """Parse runtime options shared by both ranks.

    The same script is launched on both nodes. The RANK environment variable
    decides which half of the pipeline this process runs.
    """
    parser = argparse.ArgumentParser(
        description="Run TinyLlama pipeline inference across two NCCL ranks."
    )
    parser.add_argument(
        "--model-dir",
        default="model/tinyllama",
        help="Local TinyLlama model directory. Default: model/tinyllama",
    )
    parser.add_argument(
        "--input-csv",
        default="prompts.csv",
        help="CSV file. One row is one prompt. Default: prompts.csv",
    )
    parser.add_argument(
        "--output-csv",
        default="outputs.csv",
        help="Rank 0 writes generated results here. Default: outputs.csv",
    )
    parser.add_argument(
        "--prompt-column",
        default=None,
        help=(
            "Prompt column name when --csv-has-header is set, or zero-based column "
            "index when there is no header. Default: first column."
        ),
    )
    parser.add_argument(
        "--csv-has-header",
        action="store_true",
        help="Treat the first CSV row as a header row.",
    )
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=64,
        help="Maximum generated tokens for each prompt. Default: 64",
    )
    parser.add_argument(
        "--max-input-tokens",
        type=int,
        default=1024,
        help="Truncate prompts to this many tokens. Default: 1024",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.0,
        help="0 means greedy decoding. Values > 0 enable sampling. Default: 0",
    )
    parser.add_argument(
        "--split-layer",
        type=int,
        default=DEFAULT_SPLIT_LAYER,
        help="Layer index where Rank 1 starts. Default: 5",
    )
    parser.add_argument(
        "--init-method",
        default=os.environ.get("DIST_INIT_METHOD", DEFAULT_INIT_METHOD),
        help=f"torch.distributed init method. Default: {DEFAULT_INIT_METHOD}",
    )
    parser.add_argument(
        "--timeout-seconds",
        type=int,
        default=120,
        help="Distributed initialization timeout. Default: 120",
    )
    parser.add_argument(
        "--dtype",
        choices=("auto", "float16", "bfloat16", "float32"),
        default="float16",
        help="Model dtype. Default: float16",
    )
    parser.add_argument(
        "--lazy-load",
        action="store_true",
        help=(
            "Only load the checkpoint tensors used by this rank. "
            "Requires a safetensors-format Hugging Face checkpoint."
        ),
    )
    return parser.parse_args()


# -----------------------------------------------------------------------------
# Distributed environment and model loading
# -----------------------------------------------------------------------------

def get_rank_world_size():
    """Read and validate the distributed identity of this process.

    RANK and WORLD_SIZE are set outside Python so the exact same command can be
    used on both machines except for RANK. This demo is deliberately limited to
    two ranks because the model is split into two pipeline stages.
    """
    try:
        rank = int(os.environ["RANK"])
        world_size = int(os.environ["WORLD_SIZE"])
    except KeyError as exc:
        raise RuntimeError("Please set RANK and WORLD_SIZE before running this script.") from exc

    if world_size != 2:
        raise RuntimeError("This script expects WORLD_SIZE=2.")
    if rank not in (0, 1):
        raise RuntimeError("This script only supports RANK=0 or RANK=1.")
    return rank, world_size


def resolve_dtype(dtype_name):
    """Convert the CLI dtype name to the value expected by Transformers."""
    if dtype_name == "auto":
        return "auto"
    if dtype_name == "float16":
        return torch.float16
    if dtype_name == "bfloat16":
        return torch.bfloat16
    if dtype_name == "float32":
        return torch.float32
    raise ValueError(f"Unsupported dtype: {dtype_name}")


def load_model_part_full(model_dir, rank, split_layer, dtype, device):
    """Original loading path: load the whole checkpoint, then keep this rank's layers.

    This remains useful as a stable fallback because it delegates all checkpoint
    details to Transformers. The drawback is that both ranks read the complete
    model weights before unused layers are removed.
    """
    model = AutoModelForCausalLM.from_pretrained(
        model_dir,
        torch_dtype=dtype,
        local_files_only=True,
        low_cpu_mem_usage=True,
    )
    model.eval()

    total_layers = len(model.model.layers)
    validate_split_layer(split_layer, total_layers)

    if rank == 0:
        model.model.layers = nn.ModuleList(list(model.model.layers[:split_layer]))
    else:
        model.model.layers = nn.ModuleList(list(model.model.layers[split_layer:]))

    model.to(device)
    return model, total_layers, "full"


def validate_split_layer(split_layer, total_layers):
    """Make sure the split leaves at least one decoder layer on each rank."""
    if split_layer <= 0 or split_layer >= total_layers:
        raise ValueError(
            f"--split-layer must be between 1 and {total_layers - 1}; got {split_layer}."
        )


def effective_lazy_dtype(dtype, config):
    """Choose the real tensor dtype used by the lazy loader.

    In the full from_pretrained() path, Transformers understands torch_dtype="auto".
    In the lazy path we read tensors ourselves, so we convert "auto" to the dtype
    recorded in config.json when available. If config.json does not specify one,
    float16 is a practical default for this NCCL GPU inference demo.
    """
    if dtype != "auto":
        return dtype

    config_dtype = getattr(config, "torch_dtype", None)
    if isinstance(config_dtype, torch.dtype):
        return config_dtype
    if isinstance(config_dtype, str) and hasattr(torch, config_dtype):
        return getattr(torch, config_dtype)
    return torch.float16


def prune_model_for_rank(model, rank, split_layer):
    """Remove modules this rank will never execute before moving to GPU.

    The pruning step keeps lazy loading economical. Rank 0 does not need final
    norm or lm_head; Rank 1 does not need token embeddings. Decoder layers are
    also renumbered locally after pruning, so layer 5 in the original checkpoint
    becomes model.layers.0 inside Rank 1's Python object.
    """
    if rank == 0:
        model.model.layers = nn.ModuleList(list(model.model.layers[:split_layer]))
        model.model.norm = nn.Identity()
        model.lm_head = nn.Identity()
    else:
        model.model.embed_tokens = nn.Identity()
        model.model.layers = nn.ModuleList(list(model.model.layers[split_layer:]))


def original_checkpoint_key(local_key, rank, split_layer, config):
    """Map a local state_dict key back to its original checkpoint key.

    Rank 0 keeps the original layer numbers, so most keys are unchanged. Rank 1
    renumbers later layers after pruning: local model.layers.0 corresponds to
    checkpoint model.layers.<split_layer>. This function reverses that renumbering
    when deciding which checkpoint tensor to read.
    """
    if rank == 1 and local_key.startswith("model.layers."):
        parts = local_key.split(".", 3)
        local_layer_index = int(parts[2])
        return f"model.layers.{local_layer_index + split_layer}.{parts[3]}"

    # Some tied-embedding checkpoints may not store lm_head.weight separately.
    # In that case Rank 1 can initialize lm_head.weight from embed_tokens.weight.
    if (
        rank == 1
        and local_key == "lm_head.weight"
        and getattr(config, "tie_word_embeddings", False)
    ):
        return "model.embed_tokens.weight"

    return local_key


def load_safetensors_weight_map(model_dir):
    """Return {tensor_name: shard_file} for a Hugging Face safetensors checkpoint.

    Sharded models usually have model.safetensors.index.json. Single-file models
    may only have model.safetensors. For either layout, we build the same map so
    later code can open only the shards containing tensors needed by this rank.
    """
    model_dir = Path(model_dir)
    index_path = model_dir / "model.safetensors.index.json"

    if index_path.exists():
        with open(index_path, "r", encoding="utf-8") as f:
            index = json.load(f)
        return {
            tensor_name: model_dir / shard_name
            for tensor_name, shard_name in index["weight_map"].items()
        }

    safetensor_files = sorted(model_dir.glob("*.safetensors"))
    if not safetensor_files:
        raise RuntimeError(
            "--lazy-load requires a safetensors checkpoint, but no .safetensors "
            f"files were found in {model_dir}. Use the default full loader or "
            "convert the model to safetensors."
        )

    from safetensors import safe_open

    weight_map = {}
    for shard_path in safetensor_files:
        with safe_open(str(shard_path), framework="pt", device="cpu") as shard:
            for key in shard.keys():
                weight_map[key] = shard_path
    return weight_map


def build_lazy_load_plan(model, rank, split_layer, config, weight_map):
    """Decide exactly which checkpoint tensors this rank needs to read.

    The model has already been pruned for this rank, so model.state_dict() only
    contains active parameters and buffers. We translate those local keys back to
    checkpoint keys and group them by shard file for efficient reading.
    """
    plan_by_shard = {}
    missing_parameter_keys = []
    parameter_keys = set(dict(model.named_parameters()).keys())

    for local_key in model.state_dict().keys():
        checkpoint_key = original_checkpoint_key(local_key, rank, split_layer, config)
        shard_path = weight_map.get(checkpoint_key)

        if shard_path is None:
            if local_key in parameter_keys:
                missing_parameter_keys.append((local_key, checkpoint_key))
            continue

        plan_by_shard.setdefault(shard_path, []).append((checkpoint_key, local_key))

    if missing_parameter_keys:
        details = ", ".join(
            f"{local_key}<-{checkpoint_key}"
            for local_key, checkpoint_key in missing_parameter_keys[:8]
        )
        raise RuntimeError(f"Lazy loader could not find required parameters: {details}")

    return plan_by_shard


def read_lazy_state_dict(plan_by_shard, dtype, device):
    """Read only planned tensors from safetensors shards.

    Tensors are moved to the target CUDA device while being loaded. Floating point
    tensors are also cast to the requested inference dtype, matching the old
    from_pretrained(torch_dtype=...) behavior.
    """
    from safetensors import safe_open

    state_dict = {}
    for shard_path, key_pairs in plan_by_shard.items():
        with safe_open(str(shard_path), framework="pt", device="cpu") as shard:
            for checkpoint_key, local_key in key_pairs:
                tensor = shard.get_tensor(checkpoint_key)
                if tensor.is_floating_point():
                    tensor = tensor.to(dtype=dtype)
                state_dict[local_key] = tensor.to(device=device, non_blocking=True)
    return state_dict


def assert_active_parameters_loaded(model):
    """Fail early if any active parameter is still empty or on the meta device."""
    meta_names = [name for name, param in model.named_parameters() if param.is_meta]
    if meta_names:
        raise RuntimeError(
            "Some active parameters are still on the meta device: "
            + ", ".join(meta_names[:8])
        )


def load_model_part_lazy(model_dir, rank, split_layer, dtype, device):
    """Lazy checkpoint loader: read only the tensors needed by this rank.

    Flow:
    1. Read config.json and build the TinyLlama module structure without loading
       checkpoint weights.
    2. Prune the module tree to this rank's pipeline stage.
    3. Build a safetensors loading plan for the remaining parameters.
    4. Load only those checkpoint tensors and copy them into the pruned model.

    This is the path that prevents both nodes from reading all 201 checkpoint
    tensors during startup.
    """
    config = AutoConfig.from_pretrained(model_dir, local_files_only=True)
    lazy_dtype = effective_lazy_dtype(dtype, config)

    # Build module structure from config. no_init_weights() avoids spending time
    # filling parameters with random values that will immediately be overwritten
    # by the selected checkpoint tensors.
    try:
        from transformers.modeling_utils import no_init_weights
        init_context = no_init_weights()
    except ImportError:
        init_context = contextlib.nullcontext()

    with init_context:
        model = AutoModelForCausalLM.from_config(config)
    model.eval()

    total_layers = len(model.model.layers)
    validate_split_layer(split_layer, total_layers)
    prune_model_for_rank(model, rank, split_layer)

    weight_map = load_safetensors_weight_map(model_dir)
    plan_by_shard = build_lazy_load_plan(model, rank, split_layer, config, weight_map)

    # Move only the pruned model to GPU. Unused layers and modules were already
    # removed, so they do not consume GPU memory in the lazy path.
    model.to(device=device, dtype=lazy_dtype)

    lazy_state_dict = read_lazy_state_dict(plan_by_shard, lazy_dtype, device)
    missing_keys, unexpected_keys = model.load_state_dict(lazy_state_dict, strict=False)

    parameter_keys = set(dict(model.named_parameters()).keys())
    missing_parameters = [key for key in missing_keys if key in parameter_keys]
    if missing_parameters:
        raise RuntimeError(
            "Lazy loader missed active parameters: " + ", ".join(missing_parameters[:8])
        )
    if unexpected_keys:
        raise RuntimeError(
            "Lazy loader produced unexpected keys: " + ", ".join(unexpected_keys[:8])
        )

    assert_active_parameters_loaded(model)

    tensor_count = sum(len(items) for items in plan_by_shard.values())
    shard_count = len(plan_by_shard)
    print(
        f"[Rank {rank}] Lazy-loaded {tensor_count} tensors from {shard_count} "
        "safetensors shard(s)."
    )
    return model, total_layers, "lazy"


def load_model_part(model_dir, rank, split_layer, dtype, device, lazy_load=False):
    """Load the model stage for this rank.

    The default path preserves the previously verified full-load behavior. Passing
    --lazy-load switches to selective safetensors loading so each node reads only
    its assigned stage's weights.
    """
    if lazy_load:
        return load_model_part_lazy(model_dir, rank, split_layer, dtype, device)
    return load_model_part_full(model_dir, rank, split_layer, dtype, device)


# -----------------------------------------------------------------------------
# TinyLlama forward helpers
# -----------------------------------------------------------------------------

def maybe_rotary_embeddings(model, hidden_states, position_ids):
    """Build rotary position embeddings when the installed Transformers needs them.

    Different Transformers versions expose Llama/TinyLlama decoder-layer forward
    signatures slightly differently. Newer versions may pass precomputed rotary
    embeddings through position_embeddings; older versions only use position_ids.
    Returning None is fine for the older path.
    """
    rotary = getattr(model.model, "rotary_emb", None)
    if rotary is None:
        return None
    try:
        return rotary(hidden_states, position_ids)
    except TypeError:
        return None


def make_causal_mask(batch_size, seq_len, dtype, device):
    """Create a standard causal attention mask.

    Shape is [batch, heads, query_length, key_length]. Values above the diagonal
    are set to a very negative number so a token cannot attend to future tokens.
    """
    min_value = torch.finfo(dtype).min
    mask = torch.full((seq_len, seq_len), min_value, dtype=dtype, device=device)
    mask = torch.triu(mask, diagonal=1)
    return mask.view(1, 1, seq_len, seq_len).expand(batch_size, 1, seq_len, seq_len)


def run_decoder_layers(model, hidden_states, position_ids, attention_mask):
    """Run whichever decoder layers remain in this rank's model object.

    The same helper is used by both ranks. Rank 0's model contains only early
    layers; Rank 1's model contains only later layers. The inspect.signature()
    logic makes this script tolerate small API differences across Transformers
    versions without changing the core distributed logic.
    """
    position_embeddings = maybe_rotary_embeddings(model, hidden_states, position_ids)
    cache_position = torch.arange(hidden_states.shape[1], device=hidden_states.device)

    for layer in model.model.layers:
        signature = inspect.signature(layer.forward)
        kwargs = {}
        if "attention_mask" in signature.parameters:
            kwargs["attention_mask"] = attention_mask
        if "position_ids" in signature.parameters:
            kwargs["position_ids"] = position_ids
        if "past_key_value" in signature.parameters:
            kwargs["past_key_value"] = None
        if "output_attentions" in signature.parameters:
            kwargs["output_attentions"] = False
        if "use_cache" in signature.parameters:
            kwargs["use_cache"] = False
        if "cache_position" in signature.parameters:
            kwargs["cache_position"] = cache_position
        if "position_embeddings" in signature.parameters and position_embeddings is not None:
            kwargs["position_embeddings"] = position_embeddings

        layer_outputs = layer(hidden_states, **kwargs)
        hidden_states = layer_outputs[0] if isinstance(layer_outputs, tuple) else layer_outputs

    return hidden_states


def rank0_forward(model, input_ids, device):
    """Run Rank 0's part of the model and return hidden states for Rank 1.

    Rank 0 starts from token ids, so it must apply token embedding first. It then
    runs the early decoder layers and sends the resulting hidden_states tensor to
    Rank 1 through NCCL.
    """
    batch_size, seq_len = input_ids.shape
    position_ids = torch.arange(seq_len, device=device, dtype=torch.long).unsqueeze(0)
    attention_mask = make_causal_mask(batch_size, seq_len, model.dtype, device)

    hidden_states = model.model.embed_tokens(input_ids)
    hidden_states = run_decoder_layers(model, hidden_states, position_ids, attention_mask)
    return hidden_states.contiguous()


def rank1_forward_logits(model, hidden_states, device):
    """Run Rank 1's part of the model and return logits for the last token.

    Rank 1 receives hidden states, not token ids. Therefore it skips embeddings,
    runs the later decoder layers, applies final norm and lm_head, then returns
    only the last-token logits needed to choose the next generated token.
    """
    batch_size, seq_len, _ = hidden_states.shape
    position_ids = torch.arange(seq_len, device=device, dtype=torch.long).unsqueeze(0)
    attention_mask = make_causal_mask(batch_size, seq_len, model.dtype, device)

    hidden_states = run_decoder_layers(model, hidden_states, position_ids, attention_mask)
    hidden_states = model.model.norm(hidden_states)
    logits = model.lm_head(hidden_states)
    return logits[:, -1, :]


# -----------------------------------------------------------------------------
# NCCL point-to-point message protocol
# -----------------------------------------------------------------------------

def send_stop(device):
    """Tell Rank 1 that Rank 0 has no more prompts to process.

    The first value in meta is a status code:
    - 0 means a hidden_states tensor will follow.
    - 1 means stop serving and exit the receive loop.
    """
    meta = torch.tensor([1, 0, 0, 0], dtype=torch.long, device=device)
    dist.send(meta, dst=1)


def send_hidden_to_rank1(hidden_states):
    """Send tensor metadata first, then the hidden_states tensor itself.

    dist.recv() needs the receiver to allocate a correctly shaped tensor before
    receiving payload data. The small meta tensor carries [status, batch, seq,
    hidden_size] so Rank 1 knows exactly what buffer to allocate.
    """
    batch_size, seq_len, hidden_size = hidden_states.shape
    meta = torch.tensor([0, batch_size, seq_len, hidden_size], dtype=torch.long, device=hidden_states.device)
    dist.send(meta, dst=1)
    dist.send(hidden_states, dst=1)


def recv_hidden_from_rank0(device, dtype):
    """Receive one message from Rank 0.

    Returns a CUDA hidden_states tensor, or None when Rank 0 sends the stop code.
    """
    meta = torch.empty(4, dtype=torch.long, device=device)
    dist.recv(meta, src=0)
    status, batch_size, seq_len, hidden_size = meta.tolist()
    if status == 1:
        return None
    if status != 0:
        raise RuntimeError(f"Unknown message status from Rank 0: {status}")

    hidden_states = torch.empty(
        (batch_size, seq_len, hidden_size), dtype=dtype, device=device
    )
    dist.recv(hidden_states, src=0)
    return hidden_states


def choose_next_token(logits, temperature):
    """Convert last-token logits into one token id.

    temperature=0 uses greedy decoding. A positive temperature samples from the
    softmax distribution, which makes output less deterministic.
    """
    if temperature and temperature > 0:
        probs = torch.softmax(logits / temperature, dim=-1)
        return torch.multinomial(probs, num_samples=1)
    return torch.argmax(logits, dim=-1, keepdim=True)


# -----------------------------------------------------------------------------
# CSV input and output
# -----------------------------------------------------------------------------

def read_prompts(csv_path, has_header, prompt_column):
    """Read prompts from CSV.

    Supported formats:
    - Without header: read the first column by default, or --prompt-column as a
      zero-based column index.
    - With header: read --prompt-column as a column name, or use the first header
      column if --prompt-column is omitted.
    """
    prompts = []
    with open(csv_path, "r", encoding="utf-8-sig", newline="") as f:
        if has_header:
            reader = csv.DictReader(f)
            if prompt_column is None:
                if not reader.fieldnames:
                    return []
                prompt_column = reader.fieldnames[0]
            for row in reader:
                value = (row.get(prompt_column) or "").strip()
                if value:
                    prompts.append(value)
        else:
            column_index = int(prompt_column) if prompt_column is not None else 0
            reader = csv.reader(f)
            for row in reader:
                if len(row) <= column_index:
                    continue
                value = row[column_index].strip()
                if value:
                    prompts.append(value)
    return prompts


# -----------------------------------------------------------------------------
# Rank-specific execution loops
# -----------------------------------------------------------------------------

def rank0_generate(args, model, tokenizer, device):
    """Rank 0 driver loop.

    For every prompt, Rank 0 repeatedly:
    1. Encodes or extends the current token sequence.
    2. Runs embeddings and early decoder layers.
    3. Sends hidden states to Rank 1.
    4. Receives one next-token id from Rank 1.
    5. Appends that token and continues until EOS or max_new_tokens.

    Only Rank 0 writes outputs.csv because it owns the original prompts and final
    generated token sequence.
    """
    prompts = read_prompts(args.input_csv, args.csv_has_header, args.prompt_column)
    if not prompts:
        print(f"[Rank 0] No prompts found in {args.input_csv}")
        send_stop(device)
        return

    output_path = Path(args.output_csv)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    eos_token_id = tokenizer.eos_token_id
    rows = []

    for index, prompt in enumerate(prompts, start=1):
        encoded = tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=args.max_input_tokens,
        )
        input_ids = encoded["input_ids"].to(device)
        prompt_len = input_ids.shape[1]

        print(f"[Rank 0] Prompt {index}/{len(prompts)}: {prompt}")

        with torch.inference_mode():
            for _ in range(args.max_new_tokens):
                hidden_states = rank0_forward(model, input_ids, device)
                send_hidden_to_rank1(hidden_states)

                next_token = torch.empty((1, 1), dtype=torch.long, device=device)
                dist.recv(next_token, src=1)
                input_ids = torch.cat([input_ids, next_token], dim=1)

                if eos_token_id is not None and int(next_token.item()) == eos_token_id:
                    break

        generated_ids = input_ids[0, prompt_len:]
        generated_text = tokenizer.decode(generated_ids, skip_special_tokens=True)
        full_text = tokenizer.decode(input_ids[0], skip_special_tokens=True)
        rows.append(
            {
                "prompt": prompt,
                "generated_text": generated_text,
                "full_text": full_text,
            }
        )
        print(f"[Rank 0] Output {index}: {generated_text}")

    with open(output_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["prompt", "generated_text", "full_text"])
        writer.writeheader()
        writer.writerows(rows)

    print(f"[Rank 0] Wrote {len(rows)} rows to {output_path}")
    send_stop(device)


def rank1_serve(args, model, device):
    """Rank 1 service loop.

    Rank 1 does not read prompts and does not write the output CSV. It waits for
    hidden states from Rank 0, runs the second half of the model, sends one token
    id back, and repeats until a stop message arrives.
    """
    print("[Rank 1] Waiting for hidden states from Rank 0...")
    with torch.inference_mode():
        while True:
            hidden_states = recv_hidden_from_rank0(device, model.dtype)
            if hidden_states is None:
                break
            logits = rank1_forward_logits(model, hidden_states, device)
            next_token = choose_next_token(logits, args.temperature).to(torch.long).contiguous()
            dist.send(next_token, dst=0)
    print("[Rank 1] Stop signal received.")


# -----------------------------------------------------------------------------
# Program entry point
# -----------------------------------------------------------------------------

def main():
    """Initialize NCCL, load this rank's model part, then run the right loop."""
    args = parse_args()
    rank, world_size = get_rank_world_size()

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for NCCL inference, but torch.cuda.is_available() is False.")

    device = torch.device("cuda:0")
    torch.cuda.set_device(device)

    print(f"[Rank {rank}] Starting on host {socket.gethostname()}")
    print(f"[Rank {rank}] init_method={args.init_method}")

    # NCCL is used because all tensors passed between ranks are CUDA tensors.
    # The init_method address must be reachable from both nodes.
    dist.init_process_group(
        backend="nccl",
        init_method=args.init_method,
        rank=rank,
        world_size=world_size,
        timeout=timedelta(seconds=args.timeout_seconds),
    )

    dtype = resolve_dtype(args.dtype)
    model, total_layers, load_mode = load_model_part(
        args.model_dir,
        rank,
        args.split_layer,
        dtype,
        device,
        lazy_load=args.lazy_load,
    )
    print(
        f"[Rank {rank}] Loaded TinyLlama from {args.model_dir}; "
        f"total_layers={total_layers}; split_layer={args.split_layer}; "
        f"load_mode={load_mode}"
    )

    try:
        if rank == 0:
            tokenizer = AutoTokenizer.from_pretrained(args.model_dir, local_files_only=True)
            if tokenizer.pad_token_id is None and tokenizer.eos_token_id is not None:
                tokenizer.pad_token = tokenizer.eos_token
            rank0_generate(args, model, tokenizer, device)
        else:
            rank1_serve(args, model, device)

        # Synchronize before shutdown so both ranks finish cleanly.
        dist.barrier()
        print(f"[Rank {rank}] SUCCESS")
    finally:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
