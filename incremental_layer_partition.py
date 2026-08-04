"""Incremental decoder-layer switching for dynamic pipeline partitions.

Each rank owns an independent cache of decoder layers that it has loaded from
its local safetensors checkpoint. A partition change keeps overlapping layers,
reactivates locally cached layers, and reads only previously unseen layers. No
model weights are transferred between ranks.
"""

import contextlib
import gc
import inspect
import time
from dataclasses import dataclass

import torch
import torch.nn as nn
from transformers import AutoConfig

from model_loader import load_safetensors_weight_map, renumber_local_layer_indices


MIB = 1024 * 1024


@dataclass
class LayerSwitchResult:
    """Describe one rank's local work for a dynamic partition change."""

    old_stage: tuple
    new_stage: tuple
    retained_layers: list
    cache_hit_layers: list
    loaded_layers: list
    cached_outgoing_layers: list
    evicted_layers: list
    elapsed_ms: float


class IncrementalLayerPartition:
    """Maintain active and inactive decoder layers on one rank.

    Active layers are registered in ``model.model.layers``. Inactive layers are
    held only by this manager's Python dictionary, so model forward and model
    metrics see exactly the current stage while reusable weights stay resident
    on the same rank.
    """

    CUDA_SAFETY_MARGIN_BYTES = 512 * MIB

    def __init__(
        self,
        model,
        model_dir,
        rank,
        world_size,
        layer_start,
        layer_end,
        dtype,
        device,
        batch_number,
    ):
        self.model = model
        self.model_dir = model_dir
        self.rank = int(rank)
        self.world_size = int(world_size)
        self.dtype = dtype
        self.device = torch.device(device)
        self.config = AutoConfig.from_pretrained(model_dir, local_files_only=True)
        self.total_layers = int(self.config.num_hidden_layers)
        self.weight_map = load_safetensors_weight_map(model_dir)

        initial_ids = list(range(int(layer_start), int(layer_end)))
        initial_layers = list(model.model.layers)
        if len(initial_ids) != len(initial_layers):
            raise RuntimeError(
                "Initial partition layer count does not match its global interval: "
                f"stage=[{layer_start},{layer_end}), modules={len(initial_layers)}."
            )
        if not initial_layers:
            raise RuntimeError("Incremental partition requires at least one decoder layer.")

        self.layer_class = type(initial_layers[0])
        self.layers_by_global_id = dict(zip(initial_ids, initial_layers))
        self.active_layer_ids = set(initial_ids)
        self.last_used_batch = {layer_id: int(batch_number) for layer_id in initial_ids}
        self.current_stage = (int(layer_start), int(layer_end))
        self.layer_bytes_estimate = max(
            self._module_bytes(initial_layers[0]),
            1,
        )

    @classmethod
    def from_loaded_model(
        cls,
        model,
        model_dir,
        rank,
        world_size,
        layer_start,
        layer_end,
        dtype,
        device,
        batch_number,
    ):
        """Wrap the first lazily loaded stage without re-reading any weights."""
        return cls(
            model=model,
            model_dir=model_dir,
            rank=rank,
            world_size=world_size,
            layer_start=layer_start,
            layer_end=layer_end,
            dtype=dtype,
            device=device,
            batch_number=batch_number,
        )

    @staticmethod
    def _module_bytes(module):
        """Count parameter and persistent-buffer storage without double counting."""
        seen = set()
        total = 0
        for tensor in list(module.parameters()) + list(module.buffers()):
            pointer = tensor.data_ptr() if not tensor.is_meta else id(tensor)
            identity = (pointer, tensor.numel(), tensor.element_size())
            if identity in seen:
                continue
            seen.add(identity)
            total += tensor.numel() * tensor.element_size()
        return int(total)

    def _new_layer_on_meta(self, global_layer_id):
        """Construct one decoder layer without allocating its checkpoint weights."""
        try:
            from transformers.modeling_utils import no_init_weights

            init_context = no_init_weights()
        except ImportError:
            init_context = contextlib.nullcontext()

        parameters = inspect.signature(self.layer_class.__init__).parameters
        with init_context:
            with torch.device("meta"):
                if "layer_idx" in parameters:
                    layer = self.layer_class(self.config, layer_idx=global_layer_id)
                else:
                    layer = self.layer_class(self.config)
        layer.eval()
        return layer

    def _read_layer_state(self, layer, global_layer_id):
        """Read one global decoder layer from local safetensors shards."""
        from safetensors import safe_open

        plan_by_shard = {}
        missing_keys = []
        for local_key in layer.state_dict().keys():
            checkpoint_key = f"model.layers.{global_layer_id}.{local_key}"
            shard_path = self.weight_map.get(checkpoint_key)
            if shard_path is None:
                missing_keys.append(checkpoint_key)
                continue
            plan_by_shard.setdefault(shard_path, []).append((checkpoint_key, local_key))

        if missing_keys:
            raise RuntimeError(
                f"Rank {self.rank} could not find layer {global_layer_id} tensors: "
                + ", ".join(missing_keys[:8])
            )

        state_dict = {}
        for shard_path, key_pairs in plan_by_shard.items():
            with safe_open(str(shard_path), framework="pt", device="cpu") as shard:
                for checkpoint_key, local_key in key_pairs:
                    tensor = shard.get_tensor(checkpoint_key)
                    if tensor.is_floating_point():
                        tensor = tensor.to(dtype=self.dtype)
                    state_dict[local_key] = tensor.to(device=self.device, non_blocking=True)
        return state_dict

    def _load_one_layer(self, global_layer_id):
        """Instantiate and populate one missing layer from this rank's local disk."""
        layer = self._new_layer_on_meta(global_layer_id)
        state_dict = self._read_layer_state(layer, global_layer_id)

        load_signature = inspect.signature(layer.load_state_dict).parameters
        if "assign" in load_signature:
            missing_keys, unexpected_keys = layer.load_state_dict(
                state_dict,
                strict=False,
                assign=True,
            )
        else:
            # Compatibility fallback for older PyTorch versions without assign=True.
            layer.to_empty(device=self.device)
            layer.to(dtype=self.dtype)
            missing_keys, unexpected_keys = layer.load_state_dict(state_dict, strict=False)

        parameter_keys = set(dict(layer.named_parameters()).keys())
        missing_parameters = [key for key in missing_keys if key in parameter_keys]
        if missing_parameters:
            raise RuntimeError(
                f"Rank {self.rank} missed parameters for layer {global_layer_id}: "
                + ", ".join(missing_parameters[:8])
            )
        if unexpected_keys:
            raise RuntimeError(
                f"Rank {self.rank} loaded unexpected keys for layer {global_layer_id}: "
                + ", ".join(unexpected_keys[:8])
            )

        meta_parameters = [name for name, value in layer.named_parameters() if value.is_meta]
        meta_buffers = [name for name, value in layer.named_buffers() if value.is_meta]
        if meta_parameters or meta_buffers:
            names = (meta_parameters + meta_buffers)[:8]
            raise RuntimeError(
                f"Rank {self.rank} layer {global_layer_id} still has meta tensors: "
                + ", ".join(names)
            )

        del state_dict
        return layer

    def _cuda_available_bytes(self):
        """Return memory immediately usable by new PyTorch CUDA tensors."""
        if self.device.type != "cuda":
            return None
        with torch.cuda.device(self.device):
            free_bytes, _ = torch.cuda.mem_get_info()
            reserved_bytes = torch.cuda.memory_reserved(self.device)
            allocated_bytes = torch.cuda.memory_allocated(self.device)
        reusable_reserved = max(int(reserved_bytes) - int(allocated_bytes), 0)
        return int(free_bytes) + reusable_reserved

    def _evict_inactive_layers_until_ready(self, protected_layer_ids):
        """Evict least-recently-used inactive layers when one more layer needs room."""
        if self.device.type != "cuda":
            return []

        # assign=True needs roughly one layer payload. Keep an additional margin
        # for safetensors transfer buffers and transient CUDA kernels.
        required_free = self.layer_bytes_estimate + self.CUDA_SAFETY_MARGIN_BYTES
        evicted = []
        while self._cuda_available_bytes() < required_free:
            candidates = [
                layer_id
                for layer_id in self.layers_by_global_id
                if layer_id not in self.active_layer_ids
                and layer_id not in protected_layer_ids
            ]
            if not candidates:
                break
            victim = min(candidates, key=lambda layer_id: self.last_used_batch.get(layer_id, -1))
            del self.layers_by_global_id[victim]
            self.last_used_batch.pop(victim, None)
            evicted.append(victim)
            gc.collect()
            torch.cuda.empty_cache()

        available_bytes = self._cuda_available_bytes()
        if available_bytes < self.layer_bytes_estimate:
            raise RuntimeError(
                f"Rank {self.rank} cannot load another decoder layer for the target "
                f"partition: available_cuda_mb={available_bytes / MIB:.2f}, "
                f"estimated_layer_mb={self.layer_bytes_estimate / MIB:.2f}. "
                "All remaining resident layers are required by the target stage."
            )
        return evicted

    def switch_to(self, layer_start, layer_end, batch_number):
        """Activate a new interval while retaining overlap and local cache hits."""
        layer_start = int(layer_start)
        layer_end = int(layer_end)
        batch_number = int(batch_number)
        if layer_start < 0 or layer_end > self.total_layers or layer_start >= layer_end:
            raise ValueError(
                f"Invalid incremental stage [{layer_start},{layer_end}) for "
                f"total_layers={self.total_layers}."
            )

        started = time.perf_counter()
        old_stage = self.current_stage
        old_active = set(self.active_layer_ids)
        target_ids = list(range(layer_start, layer_end))
        target_set = set(target_ids)
        retained = sorted(old_active & target_set)
        outgoing = sorted(old_active - target_set)
        cache_hits = sorted((target_set - old_active) & set(self.layers_by_global_id))
        missing = sorted(target_set - set(self.layers_by_global_id))

        # Detach outgoing layers from the model before memory-pressure eviction.
        # The manager dictionary still owns them until LRU decides otherwise.
        self.model.model.layers = nn.ModuleList(
            [self.layers_by_global_id[layer_id] for layer_id in retained]
        )
        self.active_layer_ids = set(retained)

        evicted = []
        for global_layer_id in missing:
            evicted.extend(self._evict_inactive_layers_until_ready(target_set))
            print(
                f"[Rank {self.rank}] Batch {batch_number}: loading missing local "
                f"checkpoint layer {global_layer_id}."
            )
            self.layers_by_global_id[global_layer_id] = self._load_one_layer(global_layer_id)

        self.model.model.layers = nn.ModuleList(
            [self.layers_by_global_id[layer_id] for layer_id in target_ids]
        )
        self.active_layer_ids = target_set
        self.current_stage = (layer_start, layer_end)
        for layer_id in target_ids:
            self.last_used_batch[layer_id] = batch_number

        # DynamicCache partitions are compact and rank-local. Reassign layer_idx
        # after every switch so global checkpoint ids never index local caches.
        renumber_local_layer_indices(self.model)
        self.model.config.num_hidden_layers = len(target_ids)
        if hasattr(self.model.model, "config"):
            self.model.model.config.num_hidden_layers = len(target_ids)

        elapsed_ms = (time.perf_counter() - started) * 1000.0
        return LayerSwitchResult(
            old_stage=old_stage,
            new_stage=self.current_stage,
            retained_layers=retained,
            cache_hit_layers=cache_hits,
            loaded_layers=missing,
            cached_outgoing_layers=outgoing,
            evicted_layers=sorted(set(evicted)),
            elapsed_ms=elapsed_ms,
        )

    def inactive_layer_ids(self):
        """Return decoder layers resident on this rank but not currently active."""
        return sorted(set(self.layers_by_global_id) - self.active_layer_ids)

    def release(self):
        """Drop both active and inactive layers at process shutdown."""
        if self.model is not None:
            self.model.model.layers = nn.ModuleList()
        self.layers_by_global_id.clear()
        self.active_layer_ids.clear()
        self.last_used_batch.clear()
        self.model = None
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
