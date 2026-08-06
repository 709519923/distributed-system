"""TinyLlama stage loading utilities.

This file owns both loading modes:
- full loading: load the complete checkpoint, then prune unused modules;
- lazy loading: build the module from config and read only tensors needed by
  the current rank's [layer_start, layer_end) stage.

When a run reports missing weights, unexpected keys, wrong layer intervals, or
unexpected full-checkpoint reads, start debugging here.
"""

import contextlib
import copy
import json
from pathlib import Path

import torch
import torch.nn as nn
from transformers import AutoConfig, AutoModelForCausalLM


def get_total_layers_from_config(model_dir):
    """Read total decoder layer count without loading any checkpoint weights."""
    config = AutoConfig.from_pretrained(model_dir, local_files_only=True)
    total_layers = getattr(config, "num_hidden_layers", None)
    if total_layers is None:
        raise RuntimeError("config.json does not define num_hidden_layers.")
    return int(total_layers)


def resolve_model_dtype_from_config(model_dir, dtype):
    """Resolve CLI dtype=auto before model loading and NCCL communication."""
    if dtype != "auto":
        return dtype
    config = AutoConfig.from_pretrained(model_dir, local_files_only=True)
    return effective_lazy_dtype(dtype, config)


def validate_model_structure(model):
    """Fail early unless the model exposes the pipeline modules used here."""
    backbone = getattr(model, "model", None)
    required = {
        "model": backbone,
        "model.layers": getattr(backbone, "layers", None),
        "model.embed_tokens": getattr(backbone, "embed_tokens", None),
        "model.norm": getattr(backbone, "norm", None),
        "lm_head": getattr(model, "lm_head", None),
    }
    missing = [name for name, value in required.items() if value is None]
    if missing:
        raise RuntimeError(
            "Model does not expose the Llama-style pipeline structure: "
            + ", ".join(missing)
        )


def load_model_part_full(model_dir, rank, world_size, layer_start, layer_end, dtype, device):
    """Original loading path: load the whole checkpoint, then keep this rank's layers.

    This remains useful as a stable fallback because it delegates all checkpoint
    details to Transformers. The drawback is that both ranks read the complete
    model weights before unused layers are removed.
    """
    config = AutoConfig.from_pretrained(model_dir, local_files_only=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_dir,
        config=config,
        torch_dtype=dtype,
        local_files_only=True,
        low_cpu_mem_usage=True,
    )
    model.eval()
    validate_model_structure(model)

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
    renumber_local_layer_indices(model)

    if rank == 0:
        model.model.norm = nn.Identity()
        model.lm_head = nn.Identity()

    if rank != 0:
        model.model.embed_tokens = nn.Identity()

    if rank != world_size - 1:
        model.model.norm = nn.Identity()
        model.lm_head = nn.Identity()


def renumber_local_layer_indices(model):
    """Make pruned decoder layers use rank-local cache indices.

    Hugging Face Llama layers can keep their original global layer_idx after we
    slice model.model.layers. That is harmless when a rank creates its own cache
    from scratch, but cloud-base receives a compact rank-local KV cache whose
    layers are indexed [0, local_layer_count). Renumbering keeps transferred KV
    caches, DynamicCache.update(), and the pruned ModuleList aligned.
    """
    for local_index, layer in enumerate(model.model.layers):
        if hasattr(layer, "layer_idx"):
            layer.layer_idx = local_index
        self_attn = getattr(layer, "self_attn", None)
        if self_attn is not None and hasattr(self_attn, "layer_idx"):
            self_attn.layer_idx = local_index


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
    1. Read config.json and build only this rank's decoder-layer structure.
    2. Prune the module tree to this rank's pipeline stage.
    3. Build a safetensors loading plan for the remaining parameters.
    4. Load only those checkpoint tensors and copy them into the pruned model.

    This avoids constructing every decoder layer before pruning, which keeps
    startup and dynamic switching work proportional to the local partition.
    """
    config = AutoConfig.from_pretrained(model_dir, local_files_only=True)
    lazy_dtype = effective_lazy_dtype(dtype, config)
    total_layers = int(getattr(config, "num_hidden_layers", 0) or 0)
    validate_layer_range(layer_start, layer_end, total_layers)
    local_layer_count = layer_end - layer_start

    # Build the requested number of layers directly. Local layer 0 is mapped back
    # to checkpoint layer_start by original_checkpoint_key() below.
    stage_config = copy.deepcopy(config)
    stage_config.num_hidden_layers = local_layer_count

    # no_init_weights() avoids random initialization of parameters that are
    # immediately replaced by selected checkpoint tensors.
    try:
        from transformers.modeling_utils import no_init_weights
        init_context = no_init_weights()
    except ImportError:
        init_context = contextlib.nullcontext()

    with init_context:
        model = AutoModelForCausalLM.from_config(stage_config)
    model.eval()
    validate_model_structure(model)
    prune_model_for_rank(model, rank, world_size, 0, local_layer_count)

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
        f"[Rank {rank}] Lazy-loaded model_type={getattr(config, 'model_type', 'unknown')} "
        f"with {tensor_count} tensors from {shard_count} "
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


def load_full_model_for_prefill(model_dir, dtype, device):
    """Load the complete model used by Rank 2 in cloud-base prefill mode."""
    config = AutoConfig.from_pretrained(model_dir, local_files_only=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_dir,
        config=config,
        torch_dtype=dtype,
        local_files_only=True,
        low_cpu_mem_usage=True,
    )
    model.eval()
    validate_model_structure(model)
    model.to(device)
    return model
