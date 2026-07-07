"""Environment-limited communication wrappers.

The normal project communication path lives in pipeline_comm.py and
kv_cache_transfer.py. This module is used only when Environment says a link has
simulated bandwidth or fixed delay. Keeping the simulation protocol here makes
the default run path stay as close as possible to the original code.
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
from pipeline_comm import send_token as original_send_token


TRANSFER_DONE = 2


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


def _sleep_until_environment_target(payload_bytes, real_start_time, environment, src, dst):
    """Sleep for the link target not already paid by real NCCL transfer time."""
    if environment is None:
        return
    environment.sleep_after_real_transfer(src, dst, payload_bytes, real_start_time)


def _send_done_parallel(cache_by_rank, comm_device):
    """Tell receivers the Environment-limited transfer window has ended."""
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
    """Tell one receiver the Environment-limited transfer window has ended."""
    done = torch.tensor([TRANSFER_DONE], dtype=torch.long, device=comm_device)
    dist.send(done, dst=dst)


def _recv_done(src, comm_device):
    """Wait for the sender-side Environment window to finish."""
    done = torch.empty(1, dtype=torch.long, device=comm_device)
    dist.recv(done, src=src)
    value = int(done.item())
    if value != TRANSFER_DONE:
        raise RuntimeError(f"Expected transfer done signal {TRANSFER_DONE}, got {value}.")


def send_hidden_limited(hidden_states, dst, attention_mask_2d=None, environment=None, src=None):
    """Send hidden states through the original protocol, then simulate Environment."""
    payload_bytes = _tensor_nbytes(hidden_states)
    if attention_mask_2d is not None:
        payload_bytes += int(attention_mask_2d.numel() * _dtype_element_size(torch.long))
    start = time.perf_counter()
    original_send_hidden(hidden_states, dst=dst, attention_mask_2d=attention_mask_2d)
    if src is not None:
        _sleep_until_environment_target(payload_bytes, start, environment, src, dst)
    return (time.perf_counter() - start) * 1000.0


def send_token_limited(next_token, dst, environment=None, src=None):
    """Send generated token with Environment delay on the Rank 2 -> Rank 0 link.

    The token path has no explicit done handshake, so fixed link latency is
    applied before the send. That makes Rank 0's recv_token block for the
    simulated one-way delay.
    """
    start = time.perf_counter()
    if src is not None and environment is not None:
        payload_bytes = _tensor_nbytes(next_token)
        environment.sleep_before_small_transfer(src, dst, payload_bytes)
    original_send_token(next_token, dst=dst)
    return (time.perf_counter() - start) * 1000.0


def send_prefill_inputs_limited(
    input_ids,
    attention_mask_2d,
    dst,
    comm_device,
    environment=None,
    src=0,
):
    """Send Rank 0 tokenized prompt tensors with Environment simulation."""
    input_ids = input_ids.to(device=comm_device, dtype=torch.long).contiguous()
    attention_mask_2d = attention_mask_2d.to(device=comm_device, dtype=torch.long).contiguous()
    payload_bytes = _tensor_nbytes(input_ids) + _tensor_nbytes(attention_mask_2d)

    batch_size, seq_len = input_ids.shape
    meta = torch.tensor([batch_size, seq_len], dtype=torch.long, device=comm_device)
    start = time.perf_counter()
    dist.send(meta, dst=dst)
    dist.send(input_ids, dst=dst)
    dist.send(attention_mask_2d, dst=dst)
    _sleep_until_environment_target(payload_bytes, start, environment, src, dst)
    _send_done(dst, comm_device)
    return (time.perf_counter() - start) * 1000.0


def recv_prefill_inputs_limited(src, device):
    """Receive tokenized prompts and wait for the sender Environment window."""
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
    environment,
    src=2,
    comm_dtype=None,
):
    """Send cloud-base KV-cache partitions with simulated Environment links.

    Rank 2 sends Rank 0 and Rank 1 cache partitions concurrently. The target
    wall time follows the slower configured branch among 2->0 and 2->1.
    Receivers wait for an additional done signal, which makes their
    kv_cache_recv_time_ms include the sender's extra environment sleep.
    """
    target_seconds = 0.0
    for dst, cache in cache_by_rank.items():
        payload_bytes = _cache_payload_bytes(cache, comm_dtype)
        if environment is not None:
            target_seconds = max(target_seconds, environment.target_seconds(src, dst, payload_bytes))

    ready_payloads = _send_ready_parallel(cache_by_rank, comm_device)
    start = time.perf_counter()
    metadata_payloads = _send_metadata_parallel(cache_by_rank, comm_device)
    tensor_payloads = _send_payload_parallel(cache_by_rank, comm_device, comm_dtype)
    real_elapsed = time.perf_counter() - start
    extra_sleep = max(0.0, target_seconds - real_elapsed)
    if extra_sleep > 0:
        time.sleep(extra_sleep)
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
    """Receive a KV cache and wait for the sender's Environment done signal."""
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
