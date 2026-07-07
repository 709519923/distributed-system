"""Cloud-base KV-cache transfer helpers.

The normal pipeline protocol sends simple tensors such as hidden states and
tokens. Cloud-base prefill has to move a Transformers DynamicCache instead:
one key/value tensor pair per decoder layer. Keeping that protocol here makes
cache-specific shape, device, and timing issues easier to debug.
"""

import time
from types import SimpleNamespace

import torch
import torch.distributed as dist

KV_CACHE_READY = 1


def _cache_layers(past_key_values):
    """Return the layer list from a Transformers DynamicCache-like object."""
    layers = getattr(past_key_values, "layers", None)
    if layers is None:
        raise RuntimeError("KV cache object does not expose a .layers list.")
    return layers


def _layer_key_value(layer):
    """Return one layer's key/value tensors across Transformers cache variants."""
    key = getattr(layer, "keys", None)
    value = getattr(layer, "values", None)
    if key is None:
        key = getattr(layer, "key_cache", None)
    if value is None:
        value = getattr(layer, "value_cache", None)
    if key is None or value is None:
        raise RuntimeError("KV cache layer does not expose key/value tensors.")
    return key.contiguous(), value.contiguous()


def cache_layer_pairs(past_key_values, layer_start, layer_end):
    """Extract key/value pairs for original decoder layers [layer_start, layer_end)."""
    layers = _cache_layers(past_key_values)
    return [_layer_key_value(layers[index]) for index in range(layer_start, layer_end)]


def build_dynamic_cache(layer_pairs):
    """Rebuild a rank-local DynamicCache from received key/value layer pairs."""
    try:
        from transformers.cache_utils import DynamicCache, DynamicLayer
    except ImportError:
        DynamicCache = None
        DynamicLayer = None

    rebuilt_layers = []
    for key, value in layer_pairs:
        if DynamicLayer is not None:
            try:
                layer = DynamicLayer()
            except TypeError:
                layer = SimpleNamespace()
        else:
            layer = SimpleNamespace()
        layer.keys = key.contiguous()
        layer.values = value.contiguous()
        layer.is_initialized = True
        layer.dtype = key.dtype
        layer.device = key.device
        rebuilt_layers.append(layer)

    if DynamicCache is not None:
        try:
            cache = DynamicCache()
        except TypeError:
            cache = DynamicCache(layers=[])
    else:
        cache = SimpleNamespace()

    cache.layers = rebuilt_layers
    if not hasattr(cache, "layer_class_to_replicate"):
        cache.layer_class_to_replicate = None
    if not hasattr(cache, "offloading"):
        cache.offloading = False
    return cache


def split_kv_cache_by_boundaries(full_cache, boundaries, world_size):
    """Return {rank: rank-local DynamicCache} according to layer boundaries."""
    caches = {}
    for rank in range(world_size):
        pairs = cache_layer_pairs(full_cache, boundaries[rank], boundaries[rank + 1])
        caches[rank] = build_dynamic_cache(pairs)
    return caches


def estimate_kv_cache_bytes(cache):
    """Return total key/value bytes in one DynamicCache-like object."""
    total = 0
    for layer in _cache_layers(cache):
        key, value = _layer_key_value(layer)
        total += key.numel() * key.element_size()
        total += value.numel() * value.element_size()
    return int(total)


def cache_batch_seq_len(cache):
    """Return (batch_size, seq_len) from the first key tensor in a cache."""
    layers = _cache_layers(cache)
    if not layers:
        return 0, 0
    key, _ = _layer_key_value(layers[0])
    return int(key.shape[0]), int(key.shape[2])


def _metadata_tensor(cache, device):
    """Pack per-layer key/value shapes into one fixed-size metadata tensor."""
    values = [len(_cache_layers(cache))]
    for layer in _cache_layers(cache):
        key, value = _layer_key_value(layer)
        if key.dim() != 4 or value.dim() != 4:
            raise RuntimeError(
                f"Expected 4D key/value cache tensors, got {key.shape} and {value.shape}."
            )
        values.extend(list(key.shape))
        values.extend(list(value.shape))
    return torch.tensor(values, dtype=torch.long, device=device)


def _send_metadata_parallel(cache_by_rank, comm_device):
    """Send metadata to all target ranks with isend, then wait for completion."""
    works = []
    payloads = []
    for dst, cache in sorted(cache_by_rank.items()):
        meta = _metadata_tensor(cache, comm_device).contiguous()
        payloads.append(meta)
        works.append(dist.isend(meta, dst=dst))
    for work in works:
        work.wait()
    return payloads


def _send_ready_parallel(cache_by_rank, comm_device):
    """Tell receivers that KV-cache payload transfer is about to begin.

    Receiver-side timing waits for this tiny ready message first, then starts
    the recv timer. That keeps model loading, full prefill, and cache splitting
    time out of kv_cache_recv_time_ms.
    """
    works = []
    payloads = []
    for dst in sorted(cache_by_rank):
        ready = torch.tensor([KV_CACHE_READY], dtype=torch.long, device=comm_device)
        payloads.append(ready)
        works.append(dist.isend(ready, dst=dst))
    for work in works:
        work.wait()
    return payloads


def _send_payload_parallel(cache_by_rank, comm_device, comm_dtype=None):
    """Send all key/value tensors with nonblocking sends and one final wait."""
    works = []
    payloads = []
    max_layers = max(len(_cache_layers(cache)) for cache in cache_by_rank.values())
    for layer_index in range(max_layers):
        for dst, cache in sorted(cache_by_rank.items()):
            layers = _cache_layers(cache)
            if layer_index >= len(layers):
                continue
            key, value = _layer_key_value(layers[layer_index])
            key = key.to(device=comm_device, dtype=comm_dtype or key.dtype).contiguous()
            value = value.to(device=comm_device, dtype=comm_dtype or value.dtype).contiguous()
            payloads.extend([key, value])
            works.append(dist.isend(key, dst=dst))
            works.append(dist.isend(value, dst=dst))
    for work in works:
        work.wait()
    return payloads


def send_kv_caches_parallel(cache_by_rank, comm_device, comm_dtype=None):
    """Send KV-cache partitions to multiple ranks using parallel metadata/payload sends.

    Timing starts before metadata is sent and ends after all payload tensors have
    completed. With Rank 0 and Rank 1 receiving concurrently, this wall-clock
    time should be closer to the slower receiver than to the sum of both
    receiver times.
    """
    ready_payloads = _send_ready_parallel(cache_by_rank, comm_device)
    start = time.perf_counter()
    metadata_payloads = _send_metadata_parallel(cache_by_rank, comm_device)
    tensor_payloads = _send_payload_parallel(cache_by_rank, comm_device, comm_dtype)
    elapsed_ms = (time.perf_counter() - start) * 1000.0
    # Keep tensors alive until all async work has completed.
    _ = ready_payloads, metadata_payloads, tensor_payloads
    return elapsed_ms


def _recv_ready(src, comm_device):
    """Wait until the sender announces that KV-cache transfer is ready."""
    ready = torch.empty(1, dtype=torch.long, device=comm_device)
    dist.recv(ready, src=src)
    value = int(ready.item())
    if value != KV_CACHE_READY:
        raise RuntimeError(f"Expected KV cache ready signal {KV_CACHE_READY}, got {value}.")


def recv_kv_cache(
    src,
    expected_layer_count,
    comm_device,
    compute_device,
    transfer_dtype,
    compute_dtype,
):
    """Receive one rank-local KV cache and rebuild it on compute_device.

    Waiting for the sender to finish full prefill is intentionally excluded from
    kv_cache_recv_time_ms. The timer starts only after the ready signal arrives,
    so it covers metadata receive, key/value receive, device/dtype conversion,
    contiguous layout normalization, and DynamicCache reconstruction.
    """
    _recv_ready(src, comm_device)
    start = time.perf_counter()
    meta_len = 1 + int(expected_layer_count) * 8
    meta = torch.empty(meta_len, dtype=torch.long, device=comm_device)
    dist.recv(meta, src=src)
    values = [int(value) for value in meta.tolist()]
    layer_count = values[0]
    if layer_count != int(expected_layer_count):
        raise RuntimeError(
            f"Expected {expected_layer_count} KV cache layers, received {layer_count}."
        )

    layer_pairs = []
    offset = 1
    for _ in range(layer_count):
        key_shape = tuple(values[offset : offset + 4])
        value_shape = tuple(values[offset + 4 : offset + 8])
        offset += 8
        key = torch.empty(key_shape, dtype=transfer_dtype, device=comm_device)
        value = torch.empty(value_shape, dtype=transfer_dtype, device=comm_device)
        dist.recv(key, src=src)
        dist.recv(value, src=src)
        key = key.to(device=compute_device, dtype=compute_dtype).contiguous()
        value = value.to(device=compute_device, dtype=compute_dtype).contiguous()
        layer_pairs.append((key, value))

    cache = build_dynamic_cache(layer_pairs)
    elapsed_ms = (time.perf_counter() - start) * 1000.0
    return cache, elapsed_ms


def send_prefill_inputs(input_ids, attention_mask_2d, dst, comm_device):
    """Send Rank 0 tokenized prompt tensors directly to the cloud-base rank."""
    input_ids = input_ids.to(device=comm_device, dtype=torch.long).contiguous()
    attention_mask_2d = attention_mask_2d.to(device=comm_device, dtype=torch.long).contiguous()
    batch_size, seq_len = input_ids.shape
    meta = torch.tensor([batch_size, seq_len], dtype=torch.long, device=comm_device)
    start = time.perf_counter()
    dist.send(meta, dst=dst)
    dist.send(input_ids, dst=dst)
    dist.send(attention_mask_2d, dst=dst)
    return (time.perf_counter() - start) * 1000.0


def recv_prefill_inputs(src, device):
    """Receive tokenized prompt tensors from Rank 0 on the cloud-base rank."""
    meta = torch.empty(2, dtype=torch.long, device=device)
    dist.recv(meta, src=src)
    batch_size, seq_len = [int(value) for value in meta.tolist()]
    input_ids = torch.empty((batch_size, seq_len), dtype=torch.long, device=device)
    attention_mask_2d = torch.empty((batch_size, seq_len), dtype=torch.long, device=device)
    dist.recv(input_ids, src=src)
    dist.recv(attention_mask_2d, src=src)
    return input_ids.contiguous(), attention_mask_2d.contiguous()
