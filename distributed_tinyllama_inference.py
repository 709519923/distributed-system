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
import gc
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

from scheduler import Scheduler


# Default distributed rendezvous address. Rank 1 connects to this address when
# torch.distributed initializes the process group. It can also be overridden by
# --init-method or the DIST_INIT_METHOD environment variable.
DEFAULT_INIT_METHOD = "tcp://10.50.0.57:29500"

# Rank 1 starts from this decoder layer index in two-node mode. In three-node
# mode, this is the first split and DEFAULT_SECOND_SPLIT_LAYER is the second.
DEFAULT_SPLIT_LAYER = 5
DEFAULT_SECOND_SPLIT_LAYER = 15

# Small message protocol used between neighboring pipeline ranks.
# STATUS_HIDDEN means a hidden_states tensor follows the metadata message.
# STATUS_STOP means the whole job is complete.
# STATUS_BATCH_DONE means the current dynamic-loading batch is complete. Rank 1
# keeps its current model partition cached unless the next batch uses a different
# scheduler midpoint.
STATUS_HIDDEN = 0
STATUS_STOP = 1
STATUS_BATCH_DONE = 2


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
        help="Two-node split layer, or first split in three-node mode. Default: 5",
    )
    parser.add_argument(
        "--split-layers",
        default=None,
        help=(
            "Comma-separated split layers. Use one value for WORLD_SIZE=2 "
            "(example: 5), or two values for WORLD_SIZE=3 (example: 5,15)."
        ),
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
    parser.add_argument(
        "--dynamic-load",
        action="store_true",
        help=(
            "Process prompts in batches and reload the rank-local layer partition "
            "only when Scheduler changes the split point. Requires --lazy-load."
        ),
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=64,
        help="Number of prompts per tensor batch in both normal and dynamic-loading mode. Default: 64",
    )
    parser.add_argument(
        "--allocation-csv",
        default="allocation.csv",
        help="Scheduler allocation CSV path. Default: allocation.csv",
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

    if world_size not in (2, 3):
        raise RuntimeError("This script expects WORLD_SIZE=2 or WORLD_SIZE=3.")
    if rank < 0 or rank >= world_size:
        raise RuntimeError(f"RANK must be between 0 and {world_size - 1}.")
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


def get_total_layers_from_config(model_dir):
    """Read total decoder layer count without loading any checkpoint weights."""
    config = AutoConfig.from_pretrained(model_dir, local_files_only=True)
    total_layers = getattr(config, "num_hidden_layers", None)
    if total_layers is None:
        raise RuntimeError("config.json does not define num_hidden_layers.")
    return int(total_layers)


def parse_split_layers(value):
    """Parse --split-layers into a list of integer split points."""
    if value is None:
        return None
    splits = [item.strip() for item in value.split(",") if item.strip()]
    if not splits:
        return None
    return [int(item) for item in splits]


def default_boundaries_for_world_size(args, world_size, total_layers):
    """Build [0, split..., total_layers] for WORLD_SIZE=2 or WORLD_SIZE=3."""
    explicit_splits = parse_split_layers(args.split_layers)

    if explicit_splits is None:
        if world_size == 2:
            explicit_splits = [args.split_layer]
        else:
            second_split = DEFAULT_SECOND_SPLIT_LAYER
            if second_split <= args.split_layer or second_split >= total_layers:
                second_split = max(args.split_layer + 1, (2 * total_layers) // 3)
            explicit_splits = [args.split_layer, second_split]

    expected_count = world_size - 1
    if len(explicit_splits) != expected_count:
        raise ValueError(
            f"WORLD_SIZE={world_size} expects {expected_count} split value(s); "
            f"got {explicit_splits}"
        )

    boundaries = [0] + explicit_splits + [total_layers]
    validate_boundaries(boundaries, world_size, total_layers)
    return boundaries


def validate_boundaries(boundaries, world_size, total_layers):
    """Validate a full pipeline boundary list."""
    if len(boundaries) != world_size + 1:
        raise ValueError(f"Expected {world_size + 1} boundaries, got {boundaries}")
    if boundaries[0] != 0 or boundaries[-1] != total_layers:
        raise ValueError(f"Boundaries must start at 0 and end at {total_layers}: {boundaries}")
    for left, right in zip(boundaries, boundaries[1:]):
        if left >= right:
            raise ValueError(f"Boundaries must be strictly increasing: {boundaries}")


def stage_from_boundaries(boundaries, rank):
    """Return (layer_start, layer_end) for this rank."""
    return int(boundaries[rank]), int(boundaries[rank + 1])


def load_model_part_full(model_dir, rank, world_size, layer_start, layer_end, dtype, device):
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
    validate_layer_range(layer_start, layer_end, total_layers)

    prune_model_for_rank(model, rank, world_size, layer_start, layer_end)

    model.to(device)
    return model, total_layers, "full"


def validate_layer_range(layer_start, layer_end, total_layers):
    """Make sure one rank receives a non-empty decoder layer interval."""
    if layer_start < 0 or layer_end > total_layers or layer_start >= layer_end:
        raise ValueError(
            f"Invalid layer range [{layer_start}, {layer_end}) for total_layers={total_layers}."
        )


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


def prune_model_for_rank(model, rank, world_size, layer_start, layer_end):
    """Remove modules this rank will never execute before moving to GPU.

    Rank 0 owns token embeddings. The last rank owns final norm and lm_head.
    Middle ranks own only decoder layers. Decoder layers are renumbered locally
    after pruning, so checkpoint model.layers.<layer_start> becomes local
    model.layers.0.
    """
    model.model.layers = nn.ModuleList(list(model.model.layers[layer_start:layer_end]))

    if rank == 0:
        model.model.norm = nn.Identity()
        model.lm_head = nn.Identity()

    if rank != 0:
        model.model.embed_tokens = nn.Identity()

    if rank != world_size - 1:
        model.model.norm = nn.Identity()
        model.lm_head = nn.Identity()


def original_checkpoint_key(local_key, rank, world_size, layer_start, config):
    """Map a local state_dict key back to its original checkpoint key.

    Local layer indices are always relative to the pruned stage. For any rank,
    local model.layers.0 corresponds to checkpoint model.layers.<layer_start>.
    """
    if local_key.startswith("model.layers."):
        parts = local_key.split(".", 3)
        local_layer_index = int(parts[2])
        return f"model.layers.{local_layer_index + layer_start}.{parts[3]}"

    # Some tied-embedding checkpoints may not store lm_head.weight separately.
    # In that case the last rank can initialize lm_head.weight from embed_tokens.
    if (
        rank == world_size - 1
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


def build_lazy_load_plan(model, rank, world_size, layer_start, config, weight_map):
    """Decide exactly which checkpoint tensors this rank needs to read.

    The model has already been pruned for this rank, so model.state_dict() only
    contains active parameters and buffers. We translate those local keys back to
    checkpoint keys and group them by shard file for efficient reading.
    """
    plan_by_shard = {}
    missing_parameter_keys = []
    parameter_keys = set(dict(model.named_parameters()).keys())

    for local_key in model.state_dict().keys():
        checkpoint_key = original_checkpoint_key(local_key, rank, world_size, layer_start, config)
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


def load_model_part_lazy(model_dir, rank, world_size, layer_start, layer_end, dtype, device):
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
    validate_layer_range(layer_start, layer_end, total_layers)
    prune_model_for_rank(model, rank, world_size, layer_start, layer_end)

    weight_map = load_safetensors_weight_map(model_dir)
    plan_by_shard = build_lazy_load_plan(model, rank, world_size, layer_start, config, weight_map)

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


def load_model_part(
    model_dir,
    rank,
    world_size,
    layer_start,
    layer_end,
    dtype,
    device,
    lazy_load=False,
):
    """Load the model stage for this rank.

    The default path preserves the previously verified full-load behavior. Passing
    --lazy-load switches to selective safetensors loading so each node reads only
    its assigned stage's weights.
    """
    if lazy_load:
        return load_model_part_lazy(
            model_dir, rank, world_size, layer_start, layer_end, dtype, device
        )
    return load_model_part_full(
        model_dir, rank, world_size, layer_start, layer_end, dtype, device
    )


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


def make_causal_mask(batch_size, seq_len, dtype, device, attention_mask_2d=None):
    """Create a causal attention mask, optionally blocking padding tokens.

    Shape is [batch, heads, query_length, key_length]. Values above the diagonal
    are set to a very negative number so a token cannot attend to future tokens.
    When attention_mask_2d is passed, key positions with value 0 are also masked.
    """
    min_value = torch.finfo(dtype).min
    mask = torch.full((seq_len, seq_len), min_value, dtype=dtype, device=device)
    mask = torch.triu(mask, diagonal=1)
    mask = mask.view(1, 1, seq_len, seq_len).expand(batch_size, 1, seq_len, seq_len)

    if attention_mask_2d is not None:
        padding_mask = attention_mask_2d.to(device=device)
        padding_mask = padding_mask.view(batch_size, 1, 1, seq_len)
        mask = mask.masked_fill(padding_mask == 0, min_value)

    return mask


def make_position_ids(seq_len, device, attention_mask_2d=None):
    """Build position ids that work for both single prompts and left-padded batches."""
    if attention_mask_2d is None:
        return torch.arange(seq_len, device=device, dtype=torch.long).unsqueeze(0)

    position_ids = attention_mask_2d.to(device=device, dtype=torch.long).cumsum(dim=-1) - 1
    return position_ids.masked_fill(attention_mask_2d.to(device=device) == 0, 0)


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


def rank0_forward(model, input_ids, device, attention_mask_2d=None):
    """Run Rank 0's part of the model and return hidden states for Rank 1.

    Rank 0 starts from token ids, so it must apply token embedding first. It then
    runs the early decoder layers and sends the resulting hidden_states tensor to
    Rank 1 through NCCL.
    """
    batch_size, seq_len = input_ids.shape
    position_ids = make_position_ids(seq_len, device, attention_mask_2d)
    attention_mask = make_causal_mask(batch_size, seq_len, model.dtype, device, attention_mask_2d)

    hidden_states = model.model.embed_tokens(input_ids)
    hidden_states = run_decoder_layers(model, hidden_states, position_ids, attention_mask)
    return hidden_states.contiguous()


def rank_middle_forward(model, hidden_states, device, attention_mask_2d=None):
    """Run a middle pipeline rank: hidden_states -> local decoder layers."""
    batch_size, seq_len, _ = hidden_states.shape
    position_ids = make_position_ids(seq_len, device, attention_mask_2d)
    attention_mask = make_causal_mask(batch_size, seq_len, model.dtype, device, attention_mask_2d)

    hidden_states = run_decoder_layers(model, hidden_states, position_ids, attention_mask)
    return hidden_states.contiguous()


def rank1_forward_logits(model, hidden_states, device, attention_mask_2d=None):
    """Run the last pipeline rank and return logits for the last token.

    The last rank receives hidden states, not token ids. Therefore it skips embeddings,
    runs the later decoder layers, applies final norm and lm_head, then returns
    only the last-token logits needed to choose the next generated token.
    """
    batch_size, seq_len, _ = hidden_states.shape
    position_ids = make_position_ids(seq_len, device, attention_mask_2d)
    attention_mask = make_causal_mask(batch_size, seq_len, model.dtype, device, attention_mask_2d)

    hidden_states = run_decoder_layers(model, hidden_states, position_ids, attention_mask)
    hidden_states = model.model.norm(hidden_states)
    logits = model.lm_head(hidden_states)
    return logits[:, -1, :]


# -----------------------------------------------------------------------------
# NCCL point-to-point message protocol
# -----------------------------------------------------------------------------

def send_status(status, device, dst):
    """Send a metadata-only status message to a neighboring pipeline rank."""
    meta = torch.tensor([status, 0, 0, 0, 0], dtype=torch.long, device=device)
    dist.send(meta, dst=dst)


def send_stop(device, dst=1):
    """Tell the next pipeline rank that Rank 0 has no more prompts to process.

    The first value in meta is a status code:
    - STATUS_HIDDEN means a hidden_states tensor will follow.
    - STATUS_STOP means stop serving and exit the receive loop.
    """
    send_status(STATUS_STOP, device, dst)


def send_batch_done(device, dst=1):
    """Tell the next pipeline rank that the current dynamic-loading batch is complete."""
    send_status(STATUS_BATCH_DONE, device, dst)


def send_hidden(hidden_states, dst, attention_mask_2d=None):
    """Send tensor metadata first, then the hidden_states tensor itself.

    dist.recv() needs the receiver to allocate a correctly shaped tensor before
    receiving payload data. The small meta tensor carries
    [status, batch, seq, hidden_size, has_attention_mask] so the receiver knows
    exactly what buffers to allocate.
    """
    batch_size, seq_len, hidden_size = hidden_states.shape
    has_attention_mask = 1 if attention_mask_2d is not None else 0
    meta = torch.tensor(
        [STATUS_HIDDEN, batch_size, seq_len, hidden_size, has_attention_mask],
        dtype=torch.long,
        device=hidden_states.device,
    )
    dist.send(meta, dst=dst)
    dist.send(hidden_states, dst=dst)
    if attention_mask_2d is not None:
        dist.send(attention_mask_2d.to(device=hidden_states.device, dtype=torch.long), dst=dst)


def send_hidden_to_rank1(hidden_states):
    """Backward-compatible helper for the two-node path."""
    send_hidden(hidden_states, dst=1)


def recv_hidden(src, device, dtype):
    """Receive one hidden-state or control message from a neighboring rank.

    Returns a CUDA hidden_states tensor for normal inference data.

    Special returns:
    - None means the whole job is complete.
    - STATUS_BATCH_DONE means only the current dynamic-loading batch is complete.
    """
    meta = torch.empty(5, dtype=torch.long, device=device)
    dist.recv(meta, src=src)
    status, batch_size, seq_len, hidden_size, has_attention_mask = meta.tolist()
    if status == STATUS_STOP:
        return None
    if status == STATUS_BATCH_DONE:
        return STATUS_BATCH_DONE
    if status != STATUS_HIDDEN:
        raise RuntimeError(f"Unknown message status from Rank 0: {status}")

    hidden_states = torch.empty(
        (batch_size, seq_len, hidden_size), dtype=dtype, device=device
    )
    dist.recv(hidden_states, src=src)
    attention_mask_2d = None
    if has_attention_mask:
        attention_mask_2d = torch.empty((batch_size, seq_len), dtype=torch.long, device=device)
        dist.recv(attention_mask_2d, src=src)
    return hidden_states, attention_mask_2d


def recv_hidden_from_rank0(device, dtype):
    """Backward-compatible helper for the old Rank 1 receive path."""
    return recv_hidden(src=0, device=device, dtype=dtype)


def send_token(next_token, dst):
    """Send a generated token id to the previous pipeline rank."""
    dist.send(next_token.contiguous(), dst=dst)


def recv_token(src, device, batch_size=1):
    """Receive a generated token id from the next pipeline rank."""
    next_token = torch.empty((batch_size, 1), dtype=torch.long, device=device)
    dist.recv(next_token, src=src)
    return next_token


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

def chunk_items(items, batch_size):
    """Yield (batch_number, start_index, chunk) for dynamic prompt batching.

    batch_number is 1-based because it is written to allocation.csv and is meant
    to be read by humans. start_index is 0-based and is only used for progress
    logging.
    """
    if batch_size <= 0:
        raise ValueError("--batch-size must be greater than 0.")
    for start in range(0, len(items), batch_size):
        batch_number = start // batch_size + 1
        yield batch_number, start, items[start:start + batch_size]


def generate_rows_for_prompts(args, model, tokenizer, device, prompts, start_index=0, total_count=None):
    """Generate outputs for an in-memory list of prompts using the active model.

    This function is shared by normal mode and dynamic-loading mode. Rank 0 owns
    tokenization and receives one next-token id per prompt from Rank 1 for every
    generation step. The whole prompt list passed here is one tensor batch.
    """
    eos_token_id = tokenizer.eos_token_id
    pad_token_id = tokenizer.pad_token_id
    if pad_token_id is None:
        pad_token_id = eos_token_id
    if pad_token_id is None:
        raise RuntimeError("Tokenizer must define pad_token_id or eos_token_id for batched inference.")

    total_count = total_count if total_count is not None else len(prompts)
    old_padding_side = tokenizer.padding_side
    tokenizer.padding_side = "left"

    try:
        encoded = tokenizer(
            prompts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=args.max_input_tokens,
        )
        input_ids = encoded["input_ids"].to(device)
        attention_mask_2d = encoded["attention_mask"].to(device)
    finally:
        tokenizer.padding_side = old_padding_side

    batch_size = input_ids.shape[0]
    finished = torch.zeros(batch_size, dtype=torch.bool, device=device)
    generated_tokens = [[] for _ in range(batch_size)]

    first_prompt_number = start_index + 1
    last_prompt_number = start_index + len(prompts)
    print(
        f"[Rank 0] Prompt batch {first_prompt_number}-{last_prompt_number}/"
        f"{total_count}; batch_size={batch_size}"
    )

    with torch.inference_mode():
        for _ in range(args.max_new_tokens):
            hidden_states = rank0_forward(model, input_ids, device, attention_mask_2d)
            send_hidden(hidden_states, dst=1, attention_mask_2d=attention_mask_2d)

            next_token = recv_token(src=1, device=device, batch_size=batch_size)
            active = ~finished
            tokens_to_append = torch.where(
                active.unsqueeze(1),
                next_token,
                torch.full_like(next_token, pad_token_id),
            )
            mask_to_append = active.to(dtype=attention_mask_2d.dtype).unsqueeze(1)

            for row_index in range(batch_size):
                if active[row_index]:
                    token_value = int(next_token[row_index, 0].item())
                    generated_tokens[row_index].append(token_value)
                    if eos_token_id is not None and token_value == eos_token_id:
                        finished[row_index] = True

            input_ids = torch.cat([input_ids, tokens_to_append], dim=1)
            attention_mask_2d = torch.cat([attention_mask_2d, mask_to_append], dim=1)

            if bool(finished.all().item()):
                break

    rows = []
    for local_index, prompt in enumerate(prompts, start=1):
        global_index = start_index + local_index
        generated_ids = generated_tokens[local_index - 1]
        generated_text = tokenizer.decode(generated_ids, skip_special_tokens=True)
        full_text = prompt + generated_text
        rows.append(
            {
                "prompt": prompt,
                "generated_text": generated_text,
                "full_text": full_text,
            }
        )
        print(f"[Rank 0] Output {global_index}: {generated_text}")

    return rows


def write_output_rows(output_csv, rows):
    """Write Rank 0 generation results to CSV."""
    output_path = Path(output_csv)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["prompt", "generated_text", "full_text"])
        writer.writeheader()
        writer.writerows(rows)

    print(f"[Rank 0] Wrote {len(rows)} rows to {output_path}")


def release_model(model):
    """Release the current model partition before loading a different one."""
    if model is None:
        return
    del model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


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

    all_rows = []
    for batch_number, start_index, prompt_batch in chunk_items(prompts, args.batch_size):
        print(f"[Rank 0] Static batch {batch_number}; prompts={len(prompt_batch)}")
        rows = generate_rows_for_prompts(
            args,
            model,
            tokenizer,
            device,
            prompt_batch,
            start_index=start_index,
            total_count=len(prompts),
        )
        all_rows.extend(rows)

    write_output_rows(args.output_csv, all_rows)
    send_stop(device)


def broadcast_boundaries(boundaries, world_size, device, rank):
    """Broadcast pipeline boundaries from Rank 0 to all ranks.

    A first boundary of -1 is reserved as the dynamic scheduler stop signal.
    Normal boundaries always start with 0 and end with total_layers.
    """
    if rank == 0:
        values = [int(value) for value in boundaries]
    else:
        values = [0] * (world_size + 1)
    tensor = torch.tensor(values, dtype=torch.long, device=device)
    dist.broadcast(tensor, src=0)
    return [int(value) for value in tensor.tolist()]


def stop_boundaries(world_size):
    """Return a broadcast payload that means dynamic inference is complete."""
    return [-1] + [0] * world_size


def rank0_generate_dynamic(args, tokenizer, dtype, world_size, device):
    """Rank 0 dynamic-loading driver.

    Rank 0 reads all prompts, groups them into batches, asks Scheduler which
    layer split should be used for each batch, broadcasts that split to Rank 1,
    and keeps the current model partition cached. It only releases and reloads
    weights when the scheduler midpoint changes between batches.
    """
    prompts = read_prompts(args.input_csv, args.csv_has_header, args.prompt_column)
    if not prompts:
        print(f"[Rank 0] No prompts found in {args.input_csv}")
        broadcast_boundaries(stop_boundaries(world_size), world_size, device, rank=0)
        return

    total_layers = get_total_layers_from_config(args.model_dir)
    default_boundaries = default_boundaries_for_world_size(args, world_size, total_layers)
    scheduler = Scheduler(args.allocation_csv, total_layers, default_boundaries, world_size)
    all_rows = []
    model = None
    current_stage = None

    try:
        for batch_number, start_index, prompt_batch in chunk_items(prompts, args.batch_size):
            allocation = scheduler.get_or_create(batch_number)
            boundaries = allocation.boundaries
            layer_start, layer_end = stage_from_boundaries(boundaries, rank=0)
            next_stage = (layer_start, layer_end)
            interval_text = " ".join(
                f"rank{rank}={allocation.interval_for_rank(rank)}"
                for rank in range(world_size)
            )
            print(
                f"[Rank 0] Batch {batch_number}: {interval_text}; "
                f"prompts={len(prompt_batch)}"
            )

            broadcast_boundaries(boundaries, world_size, device, rank=0)
            if model is None or next_stage != current_stage:
                if model is not None:
                    release_model(model)
                    model = None
                    print(
                        f"[Rank 0] Batch {batch_number}: split changed "
                        f"{current_stage}->{next_stage}; old partition released."
                    )

                model, _, load_mode = load_model_part(
                    args.model_dir,
                    rank=0,
                    world_size=world_size,
                    layer_start=layer_start,
                    layer_end=layer_end,
                    dtype=dtype,
                    device=device,
                    lazy_load=True,
                )
                current_stage = next_stage
                print(
                    f"[Rank 0] Batch {batch_number} model loaded; "
                    f"stage=[{layer_start},{layer_end}); load_mode={load_mode}"
                )
            else:
                print(
                    f"[Rank 0] Batch {batch_number}: reuse cached model partition "
                    f"for stage=[{layer_start},{layer_end})."
                )

            rows = generate_rows_for_prompts(
                args,
                model,
                tokenizer,
                device,
                prompt_batch,
                start_index=start_index,
                total_count=len(prompts),
            )
            all_rows.extend(rows)

            send_batch_done(device)
            print(f"[Rank 0] Batch {batch_number} complete.")

        write_output_rows(args.output_csv, all_rows)
        broadcast_boundaries(stop_boundaries(world_size), world_size, device, rank=0)
    finally:
        release_model(model)


def pipeline_serve_static(args, model, rank, world_size, device):
    """Serve a non-master pipeline rank in static mode.

    Middle ranks relay hidden states forward and relay next-token ids backward.
    The last rank computes logits and samples/greedily chooses the next token.
    """
    prev_rank = rank - 1
    next_rank = rank + 1
    is_last_rank = rank == world_size - 1

    print(f"[Rank {rank}] Waiting for hidden states from Rank {prev_rank}...")
    with torch.inference_mode():
        while True:
            message = recv_hidden(src=prev_rank, device=device, dtype=model.dtype)
            if message is None:
                if not is_last_rank:
                    send_stop(device, dst=next_rank)
                break
            if isinstance(message, int) and message == STATUS_BATCH_DONE:
                if not is_last_rank:
                    send_batch_done(device, dst=next_rank)
                continue

            hidden_states, attention_mask_2d = message
            if is_last_rank:
                logits = rank1_forward_logits(model, hidden_states, device, attention_mask_2d)
                next_token = choose_next_token(logits, args.temperature).to(torch.long).contiguous()
            else:
                hidden_states = rank_middle_forward(model, hidden_states, device, attention_mask_2d)
                send_hidden(hidden_states, dst=next_rank, attention_mask_2d=attention_mask_2d)
                next_token = recv_token(
                    src=next_rank,
                    device=device,
                    batch_size=hidden_states.shape[0],
                )

            send_token(next_token, dst=prev_rank)
    print(f"[Rank {rank}] Stop signal received.")


def pipeline_serve_dynamic(args, rank, world_size, dtype, device):
    """Dynamic-loading service loop for non-master pipeline ranks.

    Each non-master rank receives boundaries from Rank 0, records them locally,
    and compares its own [layer_start, layer_end) with the currently cached model
    partition. It only reloads weights when that interval changes.
    """
    prev_rank = rank - 1
    next_rank = rank + 1
    is_last_rank = rank == world_size - 1
    total_layers = get_total_layers_from_config(args.model_dir)
    default_boundaries = default_boundaries_for_world_size(args, world_size, total_layers)
    scheduler = Scheduler(args.allocation_csv, total_layers, default_boundaries, world_size)
    batch_number = 1
    model = None
    current_stage = None

    try:
        while True:
            boundaries = broadcast_boundaries(None, world_size, device, rank=rank)
            if boundaries[0] == -1:
                break

            allocation = scheduler.record_allocation(batch_number, boundaries)
            layer_start, layer_end = stage_from_boundaries(boundaries, rank)
            next_stage = (layer_start, layer_end)
            interval_text = " ".join(
                f"rank{stage_rank}={allocation.interval_for_rank(stage_rank)}"
                for stage_rank in range(world_size)
            )
            print(f"[Rank {rank}] Batch {batch_number}: {interval_text}")

            if model is None or next_stage != current_stage:
                if model is not None:
                    release_model(model)
                    model = None
                    print(
                        f"[Rank {rank}] Batch {batch_number}: split changed "
                        f"{current_stage}->{next_stage}; old partition released."
                    )

                model, _, load_mode = load_model_part(
                    args.model_dir,
                    rank=rank,
                    world_size=world_size,
                    layer_start=layer_start,
                    layer_end=layer_end,
                    dtype=dtype,
                    device=device,
                    lazy_load=True,
                )
                current_stage = next_stage
                print(
                    f"[Rank {rank}] Batch {batch_number} model loaded; "
                    f"stage=[{layer_start},{layer_end}); load_mode={load_mode}"
                )
            else:
                print(
                    f"[Rank {rank}] Batch {batch_number}: reuse cached model partition "
                    f"for stage=[{layer_start},{layer_end})."
                )

            with torch.inference_mode():
                while True:
                    message = recv_hidden(src=prev_rank, device=device, dtype=model.dtype)
                    if message is None:
                        if not is_last_rank:
                            send_stop(device, dst=next_rank)
                        return
                    if isinstance(message, int) and message == STATUS_BATCH_DONE:
                        if not is_last_rank:
                            send_batch_done(device, dst=next_rank)
                        break

                    hidden_states, attention_mask_2d = message
                    if is_last_rank:
                        logits = rank1_forward_logits(model, hidden_states, device, attention_mask_2d)
                        next_token = choose_next_token(logits, args.temperature).to(torch.long).contiguous()
                    else:
                        hidden_states = rank_middle_forward(model, hidden_states, device, attention_mask_2d)
                        send_hidden(hidden_states, dst=next_rank, attention_mask_2d=attention_mask_2d)
                        next_token = recv_token(
                            src=next_rank,
                            device=device,
                            batch_size=hidden_states.shape[0],
                        )

                    send_token(next_token, dst=prev_rank)

            print(f"[Rank {rank}] Batch {batch_number} complete.")
            batch_number += 1

        print(f"[Rank {rank}] Dynamic loading stop signal received.")
    finally:
        release_model(model)


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

    if args.dynamic_load and not args.lazy_load:
        raise RuntimeError("--dynamic-load requires --lazy-load.")

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
    if args.dynamic_load:
        try:
            if rank == 0:
                tokenizer = AutoTokenizer.from_pretrained(args.model_dir, local_files_only=True)
                if tokenizer.pad_token_id is None and tokenizer.eos_token_id is not None:
                    tokenizer.pad_token = tokenizer.eos_token
                rank0_generate_dynamic(args, tokenizer, dtype, world_size, device)
            else:
                pipeline_serve_dynamic(args, rank, world_size, dtype, device)

            dist.barrier()
            print(f"[Rank {rank}] SUCCESS")
            return
        finally:
            dist.destroy_process_group()

    total_layers_for_static = get_total_layers_from_config(args.model_dir)
    static_boundaries = default_boundaries_for_world_size(args, world_size, total_layers_for_static)
    layer_start, layer_end = stage_from_boundaries(static_boundaries, rank)

    model, total_layers, load_mode = load_model_part(
        args.model_dir,
        rank,
        world_size,
        layer_start,
        layer_end,
        dtype,
        device,
        lazy_load=args.lazy_load,
    )
    print(
        f"[Rank {rank}] Loaded TinyLlama from {args.model_dir}; "
        f"total_layers={total_layers}; stage=[{layer_start},{layer_end}); "
        f"load_mode={load_mode}"
    )

    try:
        if rank == 0:
            tokenizer = AutoTokenizer.from_pretrained(args.model_dir, local_files_only=True)
            if tokenizer.pad_token_id is None and tokenizer.eos_token_id is not None:
                tokenizer.pad_token = tokenizer.eos_token
            rank0_generate(args, model, tokenizer, device)
        else:
            pipeline_serve_static(args, model, rank, world_size, device)

        # Synchronize before shutdown so both ranks finish cleanly.
        dist.barrier()
        print(f"[Rank {rank}] SUCCESS")
    finally:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
