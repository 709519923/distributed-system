"""
TinyLlama two-node / three-node NCCL pipeline inference entry point.

The implementation is split by function so debugging can start in the smallest
relevant module:
- config.py: CLI options, split layers, status constants.
- distributed_env.py: RANK/WORLD_SIZE, CUDA device, NCCL init.
- model_loader.py: full loading, lazy loading, layer pruning.
- model_forward.py: attention mask, position ids, per-rank forward pass.
- pipeline_comm.py: NCCL send/recv protocol and dynamic boundary broadcast.
- csv_io.py: prompt CSV input and output CSV writing.
- inference_loops.py: Rank 0 generation and worker service loops.
- scheduler.py: dynamic allocation.csv management.

Keep launching this file exactly as before: python distributed_tinyllama_inference.py ...
"""

import socket
import faulthandler

import torch
import torch.distributed as dist
from transformers import AutoTokenizer

from config import default_boundaries_for_world_size, parse_args, stage_from_boundaries
from distributed_env import (
    broadcast_prefill_mode,
    get_rank_world_size,
    init_process_group,
    resolve_compute_device,
    resolve_cuda_device,
    resolve_dtype,
)
from inference_loops import (
    pipeline_serve_dynamic,
    pipeline_serve_static,
    rank0_generate,
    rank0_generate_dynamic,
)
from model_loader import get_total_layers_from_config, load_model_part


def load_tokenizer(model_dir):
    """Load the tokenizer on Rank 0 and ensure batch padding has a valid token."""
    tokenizer = AutoTokenizer.from_pretrained(model_dir, local_files_only=True)
    if tokenizer.pad_token_id is None and tokenizer.eos_token_id is not None:
        tokenizer.pad_token = tokenizer.eos_token
    return tokenizer


def main():
    """Initialize NCCL, load this rank's model stage, then run the selected loop."""
    faulthandler.enable(all_threads=True)
    args = parse_args()
    rank, world_size = get_rank_world_size()
    comm_device = resolve_cuda_device(args.cuda_device)
    model_device = resolve_compute_device(args.compute_device, rank, comm_device)

    print(f"[Rank {rank}] Starting on host {socket.gethostname()}")
    print(f"[Rank {rank}] init_method={args.init_method}")
    print(f"[Rank {rank}] cuda_device={comm_device}")
    print(f"[Rank {rank}] compute_device={model_device}")
    print(f"[Rank {rank}] requested_prefill_mode={args.prefill_mode}")

    if args.dynamic_load and not args.lazy_load:
        raise RuntimeError("--dynamic-load requires --lazy-load.")

    init_process_group(args, rank, world_size)

    try:
        effective_prefill_mode = broadcast_prefill_mode(args, rank, comm_device)
        print(f"[Rank {rank}] prefill_mode={effective_prefill_mode} (broadcast from Rank 0)")
        if args.prefill_mode == "cloud-base" and world_size != 3:
            raise RuntimeError("cloud-base prefill mode currently requires WORLD_SIZE=3.")
        if args.prefill_mode == "cloud-base" and not args.dynamic_load:
            raise RuntimeError("cloud-base prefill mode currently requires --dynamic-load.")

        requested_dtype = resolve_dtype(args.dtype)
        comm_dtype = torch.float16 if requested_dtype == "auto" else requested_dtype
        dtype = requested_dtype
        if rank == 0 and model_device.type == "cpu" and dtype in (torch.float16, "auto"):
            print("[Rank 0] CPU compute uses dtype=float32 for PyTorch CPU compatibility.")
            dtype = torch.float32

        if args.dynamic_load:
            if rank == 0:
                tokenizer = load_tokenizer(args.model_dir)
                rank0_generate_dynamic(
                    args,
                    tokenizer,
                    dtype,
                    world_size,
                    model_device,
                    comm_device,
                    comm_dtype=comm_dtype,
                )
            else:
                pipeline_serve_dynamic(args, rank, world_size, dtype, comm_device)

            dist.barrier()
            print(f"[Rank {rank}] SUCCESS")
            return

        total_layers = get_total_layers_from_config(args.model_dir)
        boundaries = default_boundaries_for_world_size(args, world_size, total_layers)
        layer_start, layer_end = stage_from_boundaries(boundaries, rank)

        model, total_layers, load_mode = load_model_part(
            args.model_dir,
            rank,
            world_size,
            layer_start,
            layer_end,
            dtype,
            model_device,
            lazy_load=args.lazy_load,
        )
        print(
            f"[Rank {rank}] Loaded TinyLlama from {args.model_dir}; "
            f"total_layers={total_layers}; stage=[{layer_start},{layer_end}); "
            f"load_mode={load_mode}"
        )

        if rank == 0:
            tokenizer = load_tokenizer(args.model_dir)
            rank0_generate(
                args,
                model,
                tokenizer,
                model_device,
                world_size,
                layer_start,
                layer_end,
                boundaries=boundaries,
                comm_device=comm_device,
                comm_dtype=comm_dtype,
            )
        else:
            pipeline_serve_static(args, model, rank, world_size, comm_device, layer_start, layer_end)

        dist.barrier()
        print(f"[Rank {rank}] SUCCESS")
    finally:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
