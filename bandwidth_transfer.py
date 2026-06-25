"""Bandwidth-limited communication wrappers.

The normal project communication path lives in pipeline_comm.py and
kv_cache_transfer.py. This module is used only when --bandwidth is provided.
Keeping the simulated-bandwidth protocol here makes the default run path stay
as close as possible to the original code: without --bandwidth, callers should
continue to call the original send/recv helpers directly.
"""

import time

import torch
import torch.distributed as dist

from kv_cache_transfer import (
    _cache_layers,
    _layer_key_value,
    _recv_ready,
    _send_metadata_parallel,
    _send_payload_parallel,
    _send_ready_parallel,
    build_dynamic_cache,
)
from pipeline_comm import send_hidden as original_send_hidden


TRANSFER_DONE = 2
BYTES_PER_MB = 1024 * 1024


def _dtype_element_size(dtype):
    """Return bytes per element for a torch dtype without allocating a real payload."""
    return torch.empty((), dtype=dtype).element_size()


def _tensor_nbytes(tensor, dtype=None):
    """Return the byte size a tensor will have during transfer."""
    element_size = _dtype_element_size(dtype or tensor.dtype)
    return int(tensor.numel() * element_size)


def _cache_payload_bytes(cache, transfer_dtype=None):
    """Return key/value payload bytes for one rank-local KV cache."""
    total = 0
    for layer in _cache_layers(cache):
        key, value = _layer_key_value(layer)
        total += _tensor_nbytes(key, transfer_dtype)
        total += _tensor_nbytes(value, transfer_dtype)
    return int(total)


def _sleep_until_bandwidth_target(payload_bytes, real_start_time, bandwidth_mbps):
    """Sleep only for the part of target transfer time not spent in real NCCL.

    The simulated target time is payload_bytes / bandwidth. Real NCCL transfer
    time is already paid before this helper is called, so we add only the
    remaining time. If the real transfer is slower than the target, no extra
    sleep is added.
    """
    if bandwidth_mbps is None:
        return
    target_seconds = float(payload_bytes) / (float(bandwidth_mbps) * BYTES_PER_MB)
    real_elapsed = time.perf_counter() - real_start_time
    extra_sleep = max(0.0, target_seconds - real_elapsed)
    if extra_sleep > 0:
        time.sleep(extra_sleep)


def _send_done_parallel(cache_by_rank, comm_device):
    """Tell receivers the bandwidth-limited transfer window has ended."""
    works = []
    payloads = []
    for dst in sorted(cache_by_rank):
        done = torch.tensor([TRANSFER_DONE], dtype=torch.long, device=comm_device)
        payloads.append(done)
        works.append(dist.isend(done, dst=dst))
    for work in works:
        work.wait()
    return payloads


def _send_done(dst, comm_device):
    """Tell one receiver the bandwidth-limited transfer window has ended."""
    done = torch.tensor([TRANSFER_DONE], dtype=torch.long, device=comm_device)
    dist.send(done, dst=dst)


def _recv_done(src, comm_device):
    """Wait for the sender-side bandwidth window to finish."""
    done = torch.empty(1, dtype=torch.long, device=comm_device)
    dist.recv(done, src=src)
    value = int(done.item())
    if value != TRANSFER_DONE:
        raise RuntimeError(f"Expected transfer done signal {TRANSFER_DONE}, got {value}.")


def send_hidden_limited(hidden_states, dst, attention_mask_2d=None, bandwidth_mbps=None):
    """Send hidden states through the original protocol, then simulate bandwidth.

    This wrapper is intentionally used only when --bandwidth is set. It keeps
    pipeline_comm.send_hidden untouched for unlimited runs.
    """
    payload_bytes = _tensor_nbytes(hidden_states)
    if attention_mask_2d is not None:
        payload_bytes += int(attention_mask_2d.numel() * _dtype_element_size(torch.long))
    start = time.perf_counter()
    original_send_hidden(hidden_states, dst=dst, attention_mask_2d=attention_mask_2d)
    _sleep_until_bandwidth_target(payload_bytes, start, bandwidth_mbps)


def send_prefill_inputs_limited(
    input_ids,
    attention_mask_2d,
    dst,
    comm_device,
    bandwidth_mbps=None,
):
    """Send Rank 0 tokenized prompt tensors with optional bandwidth simulation."""
    input_ids = input_ids.to(device=comm_device, dtype=torch.long).contiguous()
    attention_mask_2d = attention_mask_2d.to(device=comm_device, dtype=torch.long).contiguous()
    payload_bytes = _tensor_nbytes(input_ids) + _tensor_nbytes(attention_mask_2d)

    batch_size, seq_len = input_ids.shape
    meta = torch.tensor([batch_size, seq_len], dtype=torch.long, device=comm_device)
    start = time.perf_counter()
    dist.send(meta, dst=dst)
    dist.send(input_ids, dst=dst)
    dist.send(attention_mask_2d, dst=dst)
    _sleep_until_bandwidth_target(payload_bytes, start, bandwidth_mbps)
    _send_done(dst, comm_device)


def recv_prefill_inputs_limited(src, device):
    """Receive tokenized prompts and wait for the sender bandwidth window."""
    meta = torch.empty(2, dtype=torch.long, device=device)
    dist.recv(meta, src=src)
    batch_size, seq_len = [int(value) for value in meta.tolist()]
    input_ids = torch.empty((batch_size, seq_len), dtype=torch.long, device=device)
    attention_mask_2d = torch.empty((batch_size, seq_len), dtype=torch.long, device=device)
    dist.recv(input_ids, src=src)
    dist.recv(attention_mask_2d, src=src)
    _recv_done(src, device)
    return input_ids.contiguous(), attention_mask_2d.contiguous()


def send_kv_caches_parallel_limited(
    cache_by_rank,
    comm_device,
    bandwidth_mbps,
    comm_dtype=None,
):
    """Send cloud-base KV-cache partitions with simulated bandwidth.

    Rank 2 sends Rank 0 and Rank 1 cache partitions concurrently. The simulated
    target wall time follows the slower branch, so the payload size is the
    largest per-destination cache size rather than the sum of both branches.
    Receivers wait for an additional done signal, which makes their
    kv_cache_recv_time_ms include the sender's extra bandwidth sleep.
    """
    max_dst_payload_bytes = max(
        _cache_payload_bytes(cache, comm_dtype) for cache in cache_by_rank.values()
    )

    ready_payloads = _send_ready_parallel(cache_by_rank, comm_device)
    start = time.perf_counter()
    metadata_payloads = _send_metadata_parallel(cache_by_rank, comm_device)
    tensor_payloads = _send_payload_parallel(cache_by_rank, comm_device, comm_dtype)
    _sleep_until_bandwidth_target(max_dst_payload_bytes, start, bandwidth_mbps)
    done_payloads = _send_done_parallel(cache_by_rank, comm_device)
    elapsed_ms = (time.perf_counter() - start) * 1000.0

    # Keep tensors alive until async send work has completed.
    _ = ready_payloads, metadata_payloads, tensor_payloads, done_payloads
    return elapsed_ms


def recv_kv_cache_limited(
    src,
    expected_layer_count,
    comm_device,
    compute_device,
    transfer_dtype,
    compute_dtype,
):
    """Receive a KV cache and wait for the sender's bandwidth done signal."""
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

    raw_layer_pairs = []
    offset = 1
    for _ in range(layer_count):
        key_shape = tuple(values[offset : offset + 4])
        value_shape = tuple(values[offset + 4 : offset + 8])
        offset += 8
        key = torch.empty(key_shape, dtype=transfer_dtype, device=comm_device)
        value = torch.empty(value_shape, dtype=transfer_dtype, device=comm_device)
        dist.recv(key, src=src)
        dist.recv(value, src=src)
        raw_layer_pairs.append((key, value))

    _recv_done(src, comm_device)
    layer_pairs = []
    for key, value in raw_layer_pairs:
        key = key.to(device=compute_device, dtype=compute_dtype).contiguous()
        value = value.to(device=compute_device, dtype=compute_dtype).contiguous()
        layer_pairs.append((key, value))

    cache = build_dynamic_cache(layer_pairs)
    elapsed_ms = (time.perf_counter() - start) * 1000.0
    return cache, elapsed_ms
