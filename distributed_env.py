"""Distributed runtime helpers.

Use this file when debugging process identity, NCCL initialization, dtype
selection, or CUDA device placement. Keeping these checks away from model code
makes startup failures easier to isolate.
"""

from datetime import timedelta
import os

import torch
import torch.distributed as dist


def get_rank_world_size():
    """Read and validate the distributed identity of this process.

    RANK and WORLD_SIZE are set outside Python so the exact same command can be
    used on all machines except for RANK. This demo currently supports two or
    three ranks because the model is split into two or three pipeline stages.
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


def resolve_cuda_device(cuda_device):
    """Return the CUDA device requested by --cuda-device and set it for this process.

    The argument accepts either "0" / "1" style indices or full names such as
    "cuda:1". Each rank still sees devices according to its own CUDA_VISIBLE_DEVICES.
    """
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for NCCL inference, but torch.cuda.is_available() is False.")

    device_text = str(cuda_device)
    if not device_text.startswith("cuda"):
        device_text = f"cuda:{device_text}"
    device = torch.device(device_text)
    torch.cuda.set_device(device)
    return device


def resolve_compute_device(compute_device, rank, comm_device):
    """Return the model-compute device for this rank.

    Only Rank 0 supports CPU compute. Worker ranks must keep CUDA compute
    because they receive CUDA tensors through NCCL and continue the GPU pipeline.
    """
    if rank == 0 and compute_device == "cpu":
        return torch.device("cpu")
    return comm_device


def init_process_group(args, rank, world_size):
    """Initialize the NCCL process group used by the pipeline.

    If startup hangs or times out, check --init-method, RANK, WORLD_SIZE,
    NCCL_SOCKET_IFNAME, and firewall/port reachability first.
    """
    dist.init_process_group(
        backend="nccl",
        init_method=args.init_method,
        rank=rank,
        world_size=world_size,
        timeout=timedelta(seconds=args.timeout_seconds),
    )
