"""NCCL point-to-point protocol helpers.

All tensor metadata and neighboring-rank message shapes live here. If a run
hangs during send/recv, if batch_size seems wrong across ranks, or if dynamic
batch boundaries are not arriving, inspect this module first.
"""

import torch
import torch.distributed as dist

from config import STATUS_BATCH_DONE, STATUS_HIDDEN, STATUS_STOP


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
