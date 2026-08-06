"""High-level inference loops for master and pipeline worker ranks.

KV cache is now the default inference behavior. Every prompt batch has one
prefill pass that builds rank-local caches, followed by one-token decode passes
that reuse those caches until all rows hit EOS or --max-new-tokens is reached.
"""

import gc
import time

import torch
import torch.distributed as dist

from config import STATUS_BATCH_DONE, default_boundaries_for_world_size, stage_from_boundaries
from csv_io import chunk_items, read_prompts, write_output_rows
from distributed_env import broadcast_environment
from experiment_report import (
    append_experiment_log,
    append_summary_log,
    make_log_path,
    normalize_record,
    metric_to_tensor,
    recv_metric_records,
    recv_metric_tensor,
    send_metric_record,
    send_metric_tensor,
)
from kv_cache_transfer import (
    cache_batch_seq_len,
    recv_kv_cache,
    recv_prefill_inputs,
    send_kv_caches_parallel,
    send_prefill_inputs,
    split_kv_cache_by_boundaries,
)
from kv_cache_utils import (
    estimate_parameter_bytes,
    estimate_past_key_values_bytes,
    synchronize_cuda,
)
from incremental_layer_partition import IncrementalLayerPartition
from model_forward import choose_next_token, rank0_forward, rank1_forward_logits, rank_middle_forward
from model_loader import get_total_layers_from_config, load_full_model_for_prefill, load_model_part
from pipeline_comm import (
    broadcast_boundaries,
    recv_hidden,
    recv_token,
    send_batch_done,
    send_hidden,
    send_stop,
    send_token,
    stop_boundaries,
)
from scheduler import Scheduler, build_prompt_batch_contexts


def cloud_base_kv_transfer_has_effect(environment, src_rank, world_size):
    """Return True when any cloud-base KV target link uses Environment simulation."""
    if environment is None:
        return False
    return any(environment_link_has_effect(environment, src_rank, dst) for dst in range(world_size - 1))


def environment_link_has_effect(environment, src_rank, dst_rank):
    """Undefined links are treated as unlimited/no-delay for compatibility."""
    if environment is None:
        return False
    try:
        return environment.has_effect(src_rank, dst_rank)
    except ValueError:
        return False


def send_hidden_from_rank0(
    hidden_states,
    dst,
    attention_mask_2d,
    comm_device,
    comm_dtype=None,
    environment=None,
):
    """Move Rank 0 outputs to CUDA before NCCL send.

    Rank 0 can optionally compute on CPU. NCCL still requires CUDA tensors, so
    only the boundary tensor and mask are copied to comm_device for transport.
    """
    start_time = time.perf_counter()
    transfer_hidden_states = hidden_states.to(
        device=comm_device,
        dtype=comm_dtype,
        non_blocking=True,
    )
    transfer_attention_mask = attention_mask_2d.to(device=comm_device, non_blocking=True)
    if not environment_link_has_effect(environment, 0, dst):
        send_hidden(
            transfer_hidden_states,
            dst=dst,
            attention_mask_2d=transfer_attention_mask,
        )
    else:
        from bandwidth_transfer import send_hidden_limited

        send_hidden_limited(
            transfer_hidden_states,
            dst=dst,
            attention_mask_2d=transfer_attention_mask,
            environment=environment,
            src=0,
        )
        return (time.perf_counter() - start_time) * 1000.0
    return (time.perf_counter() - start_time) * 1000.0


def send_hidden_from_worker(hidden_states, dst, attention_mask_2d, environment=None, src=None):
    """Send hidden states from a non-master rank and return transfer wall time."""
    start_time = time.perf_counter()
    if not environment_link_has_effect(environment, src, dst):
        send_hidden(hidden_states, dst=dst, attention_mask_2d=attention_mask_2d)
    else:
        from bandwidth_transfer import send_hidden_limited

        send_hidden_limited(
            hidden_states,
            dst=dst,
            attention_mask_2d=attention_mask_2d,
            environment=environment,
            src=src,
        )
    return (time.perf_counter() - start_time) * 1000.0


def send_token_with_timing(next_token, dst, environment=None, src=None):
    """Send one generated token and return transfer wall time."""
    start_time = time.perf_counter()
    if environment_link_has_effect(environment, src, dst):
        from bandwidth_transfer import send_token_limited

        send_token_limited(next_token, dst=dst, environment=environment, src=src)
    else:
        send_token(next_token, dst=dst)
    return (time.perf_counter() - start_time) * 1000.0


def build_batch_metric(
    batch_number,
    rank,
    world_size,
    layer_start,
    layer_end,
    batch_size,
    prompt_count,
    input_seq_len_max,
    input_seq_len_avg,
    model,
    past_key_values,
    prefill_comp_time_ms=0.0,
    prefill_transfer_time_ms=0.0,
    cloud_prefill_rank2_time_ms=0.0,
    kv_cache_send_time_ms=0.0,
    kv_cache_recv_time_ms=0.0,
):
    """Build one rank-local metric record for the current batch."""
    return {
        "batch": batch_number,
        "rank": rank,
        "world_size": world_size,
        "layer_start": layer_start,
        "layer_end": layer_end,
        "layer_count": layer_end - layer_start,
        "batch_size": batch_size,
        "prompt_count": prompt_count,
        "input_seq_len_max": input_seq_len_max,
        "input_seq_len_avg": input_seq_len_avg,
        "prefill_param_bytes": estimate_parameter_bytes(model),
        "kv_cache_bytes_after_prefill": estimate_past_key_values_bytes(past_key_values),
        "prefill_comp_time_ms": prefill_comp_time_ms,
        "prefill_transfer_time_ms": prefill_transfer_time_ms,
        "cloud_prefill_rank2_time_ms": cloud_prefill_rank2_time_ms,
        "kv_cache_send_time_ms": kv_cache_send_time_ms,
        "kv_cache_recv_time_ms": kv_cache_recv_time_ms,
        "decode_step_count": 0,
        "decode_comp_time_ms": 0.0,
        "decode_transfer_time_ms": 0.0,
        "decode_time_per_token_ms": 0.0,
    }


def finish_decode_metric(metric, decode_step_count, decode_comp_time_ms, decode_transfer_time_ms):
    """Write decode-stage timing into a metric record."""
    metric["decode_step_count"] = int(decode_step_count)
    metric["decode_comp_time_ms"] = float(decode_comp_time_ms)
    metric["decode_transfer_time_ms"] = float(decode_transfer_time_ms)
    decode_elapsed_time_ms = float(decode_comp_time_ms) + float(decode_transfer_time_ms)
    metric["decode_time_per_token_ms"] = (
        decode_elapsed_time_ms / int(decode_step_count) if int(decode_step_count) else 0.0
    )


def force_decode_enabled(args):
    """Return True when the run should ignore EOS and execute fixed decode steps."""
    return getattr(args, "force_decode_steps", None) is not None


def decode_iteration_limit(args):
    """Return Rank 0 loop iterations needed for normal or forced decoding.

    The token immediately available after prefill is appended before the first
    decode forward. Therefore N forced decode forwards require N+1 loop
    iterations: one to append the prefill token, then N decode sends.
    """
    if force_decode_enabled(args):
        return int(args.force_decode_steps) + 1
    return int(args.max_new_tokens)


def generate_rows_for_prompts(
    args,
    model,
    tokenizer,
    device,
    prompts,
    world_size,
    layer_start,
    layer_end,
    batch_number,
    log_path,
    comm_device=None,
    comm_dtype=None,
    environment=None,
    start_index=0,
    total_count=None,
):
    """Generate one prompt batch with distributed KV cache.

    Rank 0 owns tokenization and output text. It does one full-sequence prefill,
    receives the first generated token from the last rank, writes prefill
    metrics, and then decodes one token per step using the cached state. The
    returned metric record is written after workers receive batch_done and send
    their own decode-stage timing records back to Rank 0.
    """
    comm_device = comm_device or device
    comm_dtype = comm_dtype or model.dtype
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
    prompt_count = len(prompts)
    input_seq_len_max = int(attention_mask_2d.shape[1])
    input_seq_len_avg = float(attention_mask_2d.sum(dim=1).float().mean().item())
    finished = torch.zeros(batch_size, dtype=torch.bool, device=device)
    generated_tokens = [[] for _ in range(batch_size)]
    rank0_past_key_values = None

    first_prompt_number = start_index + 1
    last_prompt_number = start_index + len(prompts)
    print(
        f"[Rank 0] Prompt batch {first_prompt_number}-{last_prompt_number}/"
        f"{total_count}; batch_size={batch_size}; kv_cache=on"
    )

    with torch.inference_mode():
        synchronize_cuda()
        prefill_compute_start_time = time.perf_counter()
        hidden_states, rank0_past_key_values = rank0_forward(
            model,
            input_ids,
            device,
            attention_mask_2d,
            past_key_values=None,
        )
        synchronize_cuda()
        prefill_comp_time_ms = (time.perf_counter() - prefill_compute_start_time) * 1000.0

        prefill_transfer_time_ms = send_hidden_from_rank0(
            hidden_states,
            dst=1,
            attention_mask_2d=attention_mask_2d,
            comm_device=comm_device,
            comm_dtype=comm_dtype,
            environment=environment,
        )
        last_rank = world_size - 1
        next_token = recv_token(src=last_rank, device=comm_device, batch_size=batch_size).to(device)

        rank0_metric = build_batch_metric(
            batch_number=batch_number,
            rank=0,
            world_size=world_size,
            layer_start=layer_start,
            layer_end=layer_end,
            batch_size=batch_size,
            prompt_count=prompt_count,
            input_seq_len_max=input_seq_len_max,
            input_seq_len_avg=input_seq_len_avg,
            model=model,
            past_key_values=rank0_past_key_values,
            prefill_comp_time_ms=prefill_comp_time_ms,
            prefill_transfer_time_ms=prefill_transfer_time_ms,
        )
        records = [normalize_record(rank0_metric, str(model.dtype))]
        rank0_decode_comp_time_ms = 0.0
        rank0_decode_transfer_time_ms = 0.0
        rank0_decode_step_count = 0
        force_decode = force_decode_enabled(args)
        for token_index in range(decode_iteration_limit(args)):
            synchronize_cuda()
            rank0_decode_compute_start_time = time.perf_counter()
            if force_decode:
                active = torch.ones(batch_size, dtype=torch.bool, device=device)
            else:
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
                    if not force_decode and eos_token_id is not None and token_value == eos_token_id:
                        finished[row_index] = True

            attention_mask_2d = torch.cat([attention_mask_2d, mask_to_append], dim=1)
            if force_decode:
                should_stop = rank0_decode_step_count >= int(args.force_decode_steps)
            else:
                should_stop = bool(finished.all().item()) or token_index == args.max_new_tokens - 1
            if should_stop:
                break

            decode_input_ids = tokens_to_append
            hidden_states, rank0_past_key_values = rank0_forward(
                model,
                decode_input_ids,
                device,
                attention_mask_2d,
                past_key_values=rank0_past_key_values,
            )
            synchronize_cuda()
            rank0_decode_comp_time_ms += (
                time.perf_counter() - rank0_decode_compute_start_time
            ) * 1000.0
            rank0_decode_transfer_time_ms += send_hidden_from_rank0(
                hidden_states,
                dst=1,
                attention_mask_2d=attention_mask_2d,
                comm_device=comm_device,
                comm_dtype=comm_dtype,
                environment=environment,
            )
            rank0_decode_step_count += 1
            next_token = recv_token(src=last_rank, device=comm_device, batch_size=batch_size).to(device)

        finish_decode_metric(
            records[0],
            rank0_decode_step_count,
            rank0_decode_comp_time_ms,
            rank0_decode_transfer_time_ms,
        )

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

    return rows, records


def generate_rows_for_prompts_cloud_base(
    args,
    model,
    tokenizer,
    device,
    prompts,
    world_size,
    layer_start,
    layer_end,
    batch_number,
    boundaries,
    comm_device,
    comm_dtype,
    environment=None,
    start_index=0,
    total_count=None,
):
    """Generate one prompt batch after Rank 2 builds and transfers KV cache."""
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
    prompt_count = len(prompts)
    input_seq_len_max = int(attention_mask_2d.shape[1])
    input_seq_len_avg = float(attention_mask_2d.sum(dim=1).float().mean().item())
    first_prompt_number = start_index + 1
    last_prompt_number = start_index + len(prompts)
    last_rank = world_size - 1
    print(
        f"[Rank 0] Prompt batch {first_prompt_number}-{last_prompt_number}/"
        f"{total_count}; batch_size={batch_size}; kv_cache=cloud-base"
    )

    print(
        f"[Rank 0] Cloud-base: sending prefill inputs to Rank {last_rank}; "
        f"input_ids_shape={tuple(input_ids.shape)}"
    )
    if not environment_link_has_effect(environment, 0, last_rank):
        send_prefill_inputs(
            input_ids,
            attention_mask_2d,
            dst=last_rank,
            comm_device=comm_device,
        )
    else:
        from bandwidth_transfer import send_prefill_inputs_limited

        send_prefill_inputs_limited(
            input_ids,
            attention_mask_2d,
            dst=last_rank,
            comm_device=comm_device,
            environment=environment,
            src=0,
        )
    print("[Rank 0] Cloud-base: prefill inputs sent; waiting for local KV cache from Rank 2...")
    synchronize_cuda()
    if not cloud_base_kv_transfer_has_effect(environment, last_rank, world_size):
        rank0_past_key_values, kv_cache_recv_time_ms = recv_kv_cache(
            src=last_rank,
            expected_layer_count=layer_end - layer_start,
            comm_device=comm_device,
            compute_device=device,
            transfer_dtype=comm_dtype,
            compute_dtype=getattr(model, "dtype", comm_dtype),
        )
    else:
        from bandwidth_transfer import recv_kv_cache_limited

        rank0_past_key_values, kv_cache_recv_time_ms = recv_kv_cache_limited(
            src=last_rank,
            expected_layer_count=layer_end - layer_start,
            comm_device=comm_device,
            compute_device=device,
            transfer_dtype=comm_dtype,
            compute_dtype=getattr(model, "dtype", comm_dtype),
        )
    synchronize_cuda()
    print(
        f"[Rank 0] Cloud-base: received local KV cache in "
        f"{kv_cache_recv_time_ms:.2f} ms; waiting for first token from Rank {last_rank}..."
    )
    next_token = recv_token(src=last_rank, device=comm_device, batch_size=batch_size).to(device)
    print("[Rank 0] Cloud-base: received first token; entering decode loop.")

    rank0_metric = build_batch_metric(
        batch_number=batch_number,
        rank=0,
        world_size=world_size,
        layer_start=layer_start,
        layer_end=layer_end,
        batch_size=batch_size,
        prompt_count=prompt_count,
        input_seq_len_max=input_seq_len_max,
        input_seq_len_avg=input_seq_len_avg,
        model=model,
        past_key_values=rank0_past_key_values,
        prefill_comp_time_ms=0.0,
        prefill_transfer_time_ms=0.0,
        kv_cache_recv_time_ms=kv_cache_recv_time_ms,
    )
    records = [normalize_record(rank0_metric, str(model.dtype))]
    finished = torch.zeros(batch_size, dtype=torch.bool, device=device)
    generated_tokens = [[] for _ in range(batch_size)]
    rank0_decode_comp_time_ms = 0.0
    rank0_decode_transfer_time_ms = 0.0
    rank0_decode_step_count = 0

    with torch.inference_mode():
        force_decode = force_decode_enabled(args)
        for token_index in range(decode_iteration_limit(args)):
            synchronize_cuda()
            rank0_decode_compute_start_time = time.perf_counter()
            if force_decode:
                active = torch.ones(batch_size, dtype=torch.bool, device=device)
            else:
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
                    if not force_decode and eos_token_id is not None and token_value == eos_token_id:
                        finished[row_index] = True

            attention_mask_2d = torch.cat([attention_mask_2d, mask_to_append], dim=1)
            if force_decode:
                should_stop = rank0_decode_step_count >= int(args.force_decode_steps)
            else:
                should_stop = bool(finished.all().item()) or token_index == args.max_new_tokens - 1
            if should_stop:
                break

            hidden_states, rank0_past_key_values = rank0_forward(
                model,
                tokens_to_append,
                device,
                attention_mask_2d,
                past_key_values=rank0_past_key_values,
            )
            synchronize_cuda()
            rank0_decode_comp_time_ms += (
                time.perf_counter() - rank0_decode_compute_start_time
            ) * 1000.0
            rank0_decode_transfer_time_ms += send_hidden_from_rank0(
                hidden_states,
                dst=1,
                attention_mask_2d=attention_mask_2d,
                comm_device=comm_device,
                comm_dtype=comm_dtype,
                environment=environment,
            )
            rank0_decode_step_count += 1
            next_token = recv_token(src=last_rank, device=comm_device, batch_size=batch_size).to(device)

    finish_decode_metric(
        records[0],
        rank0_decode_step_count,
        rank0_decode_comp_time_ms,
        rank0_decode_transfer_time_ms,
    )

    rows = []
    for local_index, prompt in enumerate(prompts, start=1):
        global_index = start_index + local_index
        generated_ids = generated_tokens[local_index - 1]
        generated_text = tokenizer.decode(generated_ids, skip_special_tokens=True)
        rows.append(
            {
                "prompt": prompt,
                "generated_text": generated_text,
                "full_text": prompt + generated_text,
            }
        )
        print(f"[Rank 0] Output {global_index}: {generated_text}")

    return rows, records


def release_model(model):
    """Release the current model partition before loading a different one."""
    if model is None:
        return
    del model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def wait_for_all_model_partitions(rank, batch_number, comm_device):
    """Do not start batch communication until every rank has finished loading."""
    if comm_device.type != "cuda":
        raise RuntimeError("NCCL model-ready barrier requires a CUDA communication device.")
    device_index = comm_device.index
    if device_index is None:
        device_index = torch.cuda.current_device()
    print(f"[Rank {rank}] Batch {batch_number}: waiting at model-ready barrier.")
    dist.barrier(device_ids=[device_index])
    print(f"[Rank {rank}] Batch {batch_number}: all model partitions are ready.")


def receive_cloud_base_cache_metric(
    args,
    model,
    rank,
    world_size,
    device,
    layer_start,
    layer_end,
    batch_number,
    environment=None,
):
    """Receive this worker's cloud-base KV cache from Rank 2 and build metrics."""
    print(f"[Rank {rank}] Cloud-base: waiting for local KV cache from Rank {world_size - 1}...")
    synchronize_cuda()
    if not cloud_base_kv_transfer_has_effect(environment, world_size - 1, world_size):
        past_key_values, kv_cache_recv_time_ms = recv_kv_cache(
            src=world_size - 1,
            expected_layer_count=layer_end - layer_start,
            comm_device=device,
            compute_device=device,
            transfer_dtype=getattr(model, "dtype", torch.float16),
            compute_dtype=getattr(model, "dtype", torch.float16),
        )
    else:
        from bandwidth_transfer import recv_kv_cache_limited

        past_key_values, kv_cache_recv_time_ms = recv_kv_cache_limited(
            src=world_size - 1,
            expected_layer_count=layer_end - layer_start,
            comm_device=device,
            compute_device=device,
            transfer_dtype=getattr(model, "dtype", torch.float16),
            compute_dtype=getattr(model, "dtype", torch.float16),
        )
    synchronize_cuda()
    print(f"[Rank {rank}] Cloud-base: received local KV cache in {kv_cache_recv_time_ms:.2f} ms.")
    batch_size, seq_len = cache_batch_seq_len(past_key_values)
    metric = build_batch_metric(
        batch_number=batch_number,
        rank=rank,
        world_size=world_size,
        layer_start=layer_start,
        layer_end=layer_end,
        batch_size=batch_size,
        prompt_count=batch_size,
        input_seq_len_max=seq_len,
        input_seq_len_avg=float(seq_len),
        model=model,
        past_key_values=past_key_values,
        prefill_comp_time_ms=0.0,
        prefill_transfer_time_ms=0.0,
        kv_cache_recv_time_ms=kv_cache_recv_time_ms,
    )
    return past_key_values, metric


def run_rank2_cloud_base_prefill(
    args,
    decode_model,
    full_prefill_model,
    world_size,
    device,
    layer_start,
    layer_end,
    batch_number,
    boundaries,
    comm_dtype,
    environment=None,
):
    """Let Rank 2 full-prefill a batch, distribute KV cache, and return first token."""
    print("[Rank 2] Cloud-base: waiting for prefill inputs from Rank 0...")
    if not environment_link_has_effect(environment, 0, world_size - 1):
        input_ids, attention_mask_2d = recv_prefill_inputs(src=0, device=device)
    else:
        from bandwidth_transfer import recv_prefill_inputs_limited

        input_ids, attention_mask_2d = recv_prefill_inputs_limited(src=0, device=device)
    batch_size = int(input_ids.shape[0])
    input_seq_len_max = int(attention_mask_2d.shape[1])
    input_seq_len_avg = float(attention_mask_2d.sum(dim=1).float().mean().item())
    print(
        f"[Rank 2] Cloud-base: received prefill inputs; "
        f"input_ids_shape={tuple(input_ids.shape)}"
    )

    synchronize_cuda()
    prefill_start = time.perf_counter()
    print("[Rank 2] Cloud-base: running full-model prefill...")
    hidden_states, full_past_key_values = rank0_forward(
        full_prefill_model,
        input_ids,
        device,
        attention_mask_2d,
        past_key_values=None,
    )
    logits = full_prefill_model.lm_head(hidden_states)[:, -1, :]
    next_token = choose_next_token(logits, args.temperature).to(torch.long).contiguous()
    synchronize_cuda()
    cloud_prefill_rank2_time_ms = (time.perf_counter() - prefill_start) * 1000.0
    print(
        f"[Rank 2] Cloud-base: full-model prefill finished in "
        f"{cloud_prefill_rank2_time_ms:.2f} ms; splitting KV cache..."
    )

    cache_by_rank = split_kv_cache_by_boundaries(full_past_key_values, boundaries, world_size)
    rank2_past_key_values = cache_by_rank[world_size - 1]
    transfer_targets = {0: cache_by_rank[0], 1: cache_by_rank[1]}
    print("[Rank 2] Cloud-base: sending KV cache partitions to Rank 0 and Rank 1...")
    if not cloud_base_kv_transfer_has_effect(environment, world_size - 1, world_size):
        kv_cache_send_time_ms = send_kv_caches_parallel(
            transfer_targets,
            comm_device=device,
            comm_dtype=comm_dtype,
        )
    else:
        from bandwidth_transfer import send_kv_caches_parallel_limited

        kv_cache_send_time_ms = send_kv_caches_parallel_limited(
            transfer_targets,
            comm_device=device,
            environment=environment,
            src=world_size - 1,
            comm_dtype=comm_dtype,
        )
    print(
        f"[Rank 2] Cloud-base: KV cache partitions sent in "
        f"{kv_cache_send_time_ms:.2f} ms; sending first token to Rank 0."
    )
    send_token_with_timing(
        next_token,
        dst=0,
        environment=environment,
        src=world_size - 1,
    )
    synchronize_cuda()

    metric = build_batch_metric(
        batch_number=batch_number,
        rank=world_size - 1,
        world_size=world_size,
        layer_start=layer_start,
        layer_end=layer_end,
        batch_size=batch_size,
        prompt_count=batch_size,
        input_seq_len_max=input_seq_len_max,
        input_seq_len_avg=input_seq_len_avg,
        model=decode_model,
        past_key_values=rank2_past_key_values,
        prefill_comp_time_ms=0.0,
        prefill_transfer_time_ms=0.0,
        cloud_prefill_rank2_time_ms=cloud_prefill_rank2_time_ms,
        kv_cache_send_time_ms=kv_cache_send_time_ms,
    )
    return rank2_past_key_values, metric


def rank0_generate(
    args,
    model,
    tokenizer,
    device,
    world_size,
    layer_start,
    layer_end,
    boundaries=None,
    comm_device=None,
    comm_dtype=None,
    environment=None,
):
    """Rank 0 static-split driver loop."""
    comm_device = comm_device or device
    comm_dtype = comm_dtype or model.dtype
    prompts = read_prompts(args.input_csv, args.csv_has_header, args.prompt_column)
    if not prompts:
        print(f"[Rank 0] No prompts found in {args.input_csv}")
        send_stop(comm_device)
        return

    log_path = make_log_path()
    print(f"[Rank 0] KV-cache experiment log: {log_path}")
    all_rows = []
    for batch_number, start_index, prompt_batch in chunk_items(prompts, args.batch_size):
        print(f"[Rank 0] Static batch {batch_number}; prompts={len(prompt_batch)}")
        if args.prefill_mode == "cloud-base":
            rows, records = generate_rows_for_prompts_cloud_base(
                args,
                model,
                tokenizer,
                device,
                prompt_batch,
                world_size=world_size,
                layer_start=layer_start,
                layer_end=layer_end,
                batch_number=batch_number,
                boundaries=boundaries,
                comm_device=comm_device,
                comm_dtype=comm_dtype,
                environment=environment,
                start_index=start_index,
                total_count=len(prompts),
            )
        else:
            rows, records = generate_rows_for_prompts(
                args,
                model,
                tokenizer,
                device,
                prompt_batch,
                world_size=world_size,
                layer_start=layer_start,
                layer_end=layer_end,
                batch_number=batch_number,
                log_path=log_path,
                comm_device=comm_device,
                comm_dtype=comm_dtype,
                environment=environment,
                start_index=start_index,
                total_count=len(prompts),
            )
        all_rows.extend(rows)
        send_batch_done(comm_device)
        records.extend(recv_metric_records(world_size, comm_device, str(comm_dtype)))
        append_experiment_log(log_path, records)
        append_summary_log(
            log_path,
            records,
            batch_number,
            environment=environment,
            boundaries=boundaries,
            prefill_mode=args.prefill_mode,
        )
        print(f"[Rank 0] Batch {batch_number} log written to {log_path}")

    write_output_rows(args.output_csv, all_rows)
    send_stop(comm_device)


def boundaries_for_batch(args, scheduler, default_boundaries, batch_number):
    """Return Scheduler boundaries when configured, otherwise fixed boundaries."""
    if args.allocation_csv:
        allocation = scheduler.get_or_create(batch_number)
        return allocation.boundaries, allocation
    return default_boundaries, None


def interval_text_from_boundaries(boundaries, world_size):
    """Format rank intervals for logs without requiring Scheduler allocation rows."""
    return " ".join(
        f"rank{rank}=[{boundaries[rank]},{boundaries[rank + 1]})"
        for rank in range(world_size)
    )


def rank0_generate_dynamic(
    args,
    tokenizer,
    dtype,
    world_size,
    device,
    comm_device=None,
    comm_dtype=None,
    environment=None,
):
    """Rank 0 dynamic-loading driver.

    With the standard run.sh entry point, Rank 0 always provides scheduler.csv
    and the Scheduler controls each batch's partition. If the script is started
    manually without --allocation-csv, every batch uses --split-layers as a
    fixed partition. KV cache is always per-batch and is rebuilt after every
    batch.
    """
    comm_device = comm_device or device
    comm_dtype = comm_dtype or dtype
    prompts = read_prompts(args.input_csv, args.csv_has_header, args.prompt_column)
    if not prompts:
        print(f"[Rank 0] No prompts found in {args.input_csv}")
        broadcast_boundaries(stop_boundaries(world_size), world_size, comm_device, rank=0)
        return

    total_layers = get_total_layers_from_config(args.model_dir)
    default_boundaries = default_boundaries_for_world_size(args, world_size, total_layers)
    scheduler = None
    batch_contexts = {}
    if args.allocation_csv:
        scheduler = Scheduler(
            args.allocation_csv,
            total_layers,
            default_boundaries,
            world_size,
            bandit_policy=args.bandit_policy,
        )
        batch_contexts = build_prompt_batch_contexts(
            prompts=prompts,
            tokenizer=tokenizer,
            batch_size=args.batch_size,
            max_input_tokens=args.max_input_tokens,
        )

    log_path = make_log_path()
    print(f"[Rank 0] KV-cache experiment log: {log_path}")
    all_rows = []
    model = None
    layer_partition = None
    current_stage = None

    try:
        for batch_number, start_index, prompt_batch in chunk_items(prompts, args.batch_size):
            batch_context = batch_contexts.get(batch_number)
            if scheduler is not None:
                selected_arm = scheduler.select_arm_before_batch(
                    batch=batch_number,
                    context=batch_context,
                )
                if selected_arm is None:
                    boundaries, allocation = boundaries_for_batch(
                        args,
                        scheduler,
                        default_boundaries,
                        batch_number,
                    )
                else:
                    boundaries = scheduler.reallocate_layer(
                        batch=batch_number,
                        arm=selected_arm,
                    )
                    allocation = scheduler.get_or_create(batch_number)
            else:
                selected_arm = None
                boundaries, allocation = boundaries_for_batch(
                    args,
                    scheduler,
                    default_boundaries,
                    batch_number,
                )
            layer_start, layer_end = stage_from_boundaries(boundaries, rank=0)
            next_stage = (layer_start, layer_end)
            if allocation is None:
                interval_text = interval_text_from_boundaries(boundaries, world_size)
            else:
                interval_text = " ".join(
                    f"rank{rank}={allocation.interval_for_rank(rank)}"
                    for rank in range(world_size)
                )
            print(
                f"[Rank 0] Batch {batch_number}: {interval_text}; "
                f"prompts={len(prompt_batch)}"
            )

            print(
                f"[Rank 0] Batch {batch_number}: broadcasting boundaries "
                f"{boundaries}; prefill_mode={args.prefill_mode}"
            )
            broadcast_boundaries(boundaries, world_size, comm_device, rank=0)
            if environment is not None:
                environment.apply_batch(batch_number)
                environment = broadcast_environment(environment, rank=0, device=comm_device)
                print(f"[Rank 0] Batch {batch_number}: environment={environment.describe()}")
            if model is None:
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
                layer_partition = IncrementalLayerPartition.from_loaded_model(
                    model=model,
                    model_dir=args.model_dir,
                    rank=0,
                    world_size=world_size,
                    layer_start=layer_start,
                    layer_end=layer_end,
                    dtype=dtype,
                    device=device,
                    batch_number=batch_number,
                )
                current_stage = next_stage
                print(
                    f"[Rank 0] Batch {batch_number} model loaded; "
                    f"stage=[{layer_start},{layer_end}); load_mode={load_mode}"
                )
            elif next_stage != current_stage:
                switch_result = layer_partition.switch_to(
                    layer_start,
                    layer_end,
                    batch_number,
                )
                model = layer_partition.model
                current_stage = next_stage
                print(
                    f"[Rank 0] Batch {batch_number}: incremental split "
                    f"{switch_result.old_stage}->{switch_result.new_stage}; "
                    f"retained={switch_result.retained_layers}; "
                    f"cache_hits={switch_result.cache_hit_layers}; "
                    f"local_loaded={switch_result.loaded_layers}; "
                    f"inactive_cache={layer_partition.inactive_layer_ids()}; "
                    f"evicted={switch_result.evicted_layers}; "
                    f"elapsed_ms={switch_result.elapsed_ms:.2f}."
                )
            else:
                print(
                    f"[Rank 0] Batch {batch_number}: reuse cached model partition "
                    f"for stage=[{layer_start},{layer_end})."
                )

            wait_for_all_model_partitions(0, batch_number, comm_device)

            if args.prefill_mode == "cloud-base":
                rows, records = generate_rows_for_prompts_cloud_base(
                    args,
                    model,
                    tokenizer,
                    device,
                    prompt_batch,
                    world_size=world_size,
                    layer_start=layer_start,
                    layer_end=layer_end,
                    batch_number=batch_number,
                    boundaries=boundaries,
                    start_index=start_index,
                    total_count=len(prompts),
                    comm_device=comm_device,
                    comm_dtype=comm_dtype,
                    environment=environment,
                )
            else:
                rows, records = generate_rows_for_prompts(
                    args,
                    model,
                    tokenizer,
                    device,
                    prompt_batch,
                    world_size=world_size,
                    layer_start=layer_start,
                    layer_end=layer_end,
                    batch_number=batch_number,
                    log_path=log_path,
                    start_index=start_index,
                    total_count=len(prompts),
                    comm_device=comm_device,
                    comm_dtype=comm_dtype,
                    environment=environment,
                )
            all_rows.extend(rows)

            send_batch_done(comm_device)
            records.extend(recv_metric_records(world_size, comm_device, str(comm_dtype)))
            if scheduler is not None:
                scheduler.collect_batch_summary(
                    batch=batch_number,
                    prefill_mode=args.prefill_mode,
                    boundaries=boundaries,
                    records=records,
                    context=batch_context,
                    selected_arm=selected_arm,
                )
                scheduler.update_policy_after_batch(batch_number)
            append_experiment_log(log_path, records)
            append_summary_log(
                log_path,
                records,
                batch_number,
                environment=environment,
                boundaries=boundaries,
                prefill_mode=args.prefill_mode,
            )
            print(f"[Rank 0] Batch {batch_number} log written to {log_path}")
            print(f"[Rank 0] Batch {batch_number} complete.")

        write_output_rows(args.output_csv, all_rows)
        broadcast_boundaries(stop_boundaries(world_size), world_size, comm_device, rank=0)
    finally:
        if layer_partition is not None:
            layer_partition.release()
            model = None
        release_model(model)


def handle_worker_batch(
    args,
    model,
    rank,
    world_size,
    device,
    layer_start,
    layer_end,
    batch_number,
    initial_past_key_values=None,
    initial_metric=None,
    environment=None,
):
    """Serve one worker batch and keep rank-local KV cache until batch_done."""
    prev_rank = rank - 1
    next_rank = rank + 1
    is_last_rank = rank == world_size - 1
    past_key_values = initial_past_key_values
    metric = initial_metric
    decode_comp_time_ms = 0.0
    decode_transfer_time_ms = 0.0
    decode_step_count = 0

    with torch.inference_mode():
        while True:
            message = recv_hidden(src=prev_rank, device=device, dtype=model.dtype)
            if message is None:
                if not is_last_rank:
                    send_stop(device, dst=next_rank)
                return False
            if isinstance(message, int) and message == STATUS_BATCH_DONE:
                downstream_metric_tensors = []
                if not is_last_rank:
                    send_batch_done(device, dst=next_rank)
                    downstream_metric_tensors = [
                        recv_metric_tensor(src=next_rank, device=device)
                        for _ in range(world_size - rank - 1)
                    ]
                if metric is not None:
                    finish_decode_metric(
                        metric,
                        decode_step_count,
                        decode_comp_time_ms,
                        decode_transfer_time_ms,
                    )
                    if is_last_rank:
                        send_metric_record(metric, device, dst=prev_rank)
                    else:
                        send_metric_tensor(metric_to_tensor(metric, device), dst=prev_rank)
                        for downstream_metric_tensor in downstream_metric_tensors:
                            send_metric_tensor(downstream_metric_tensor, dst=prev_rank)
                return True

            hidden_states, attention_mask_2d = message
            is_prefill = past_key_values is None
            if is_prefill:
                batch_size = int(hidden_states.shape[0])
                prompt_count = batch_size
                input_seq_len_max = int(attention_mask_2d.shape[1])
                input_seq_len_avg = float(attention_mask_2d.sum(dim=1).float().mean().item())

            synchronize_cuda()
            compute_start_time = time.perf_counter()

            if is_last_rank:
                logits, past_key_values = rank1_forward_logits(
                    model,
                    hidden_states,
                    device,
                    attention_mask_2d,
                    past_key_values=past_key_values,
                )
                next_token = choose_next_token(logits, args.temperature).to(torch.long).contiguous()
                synchronize_cuda()
                comp_time_ms = (time.perf_counter() - compute_start_time) * 1000.0
                transfer_time_ms = send_token_with_timing(
                    next_token,
                    dst=0,
                    environment=environment,
                    src=rank,
                )
            else:
                output_hidden_states, past_key_values = rank_middle_forward(
                    model,
                    hidden_states,
                    device,
                    attention_mask_2d,
                    past_key_values=past_key_values,
                )
                synchronize_cuda()
                comp_time_ms = (time.perf_counter() - compute_start_time) * 1000.0
                transfer_time_ms = send_hidden_from_worker(
                    output_hidden_states,
                    dst=next_rank,
                    attention_mask_2d=attention_mask_2d,
                    environment=environment,
                    src=rank,
                )

            if is_prefill and metric is None:
                metric = build_batch_metric(
                    batch_number=batch_number,
                    rank=rank,
                    world_size=world_size,
                    layer_start=layer_start,
                    layer_end=layer_end,
                    batch_size=batch_size,
                    prompt_count=prompt_count,
                    input_seq_len_max=input_seq_len_max,
                    input_seq_len_avg=input_seq_len_avg,
                    model=model,
                    past_key_values=past_key_values,
                    prefill_comp_time_ms=comp_time_ms,
                    prefill_transfer_time_ms=transfer_time_ms,
                )
            elif not is_prefill:
                decode_comp_time_ms += comp_time_ms
                decode_transfer_time_ms += transfer_time_ms
                decode_step_count += 1


def pipeline_serve_static(
    args,
    model,
    rank,
    world_size,
    device,
    layer_start,
    layer_end,
    boundaries=None,
    dtype=None,
    comm_dtype=None,
    environment=None,
):
    """Serve a non-master pipeline rank in static mode."""
    batch_number = 1
    full_prefill_model = None
    if args.prefill_mode == "cloud-base" and rank == world_size - 1:
        print(f"[Rank {rank}] Loading full model for cloud-base KV prefill...")
        full_prefill_model = load_full_model_for_prefill(args.model_dir, dtype, device)
        print(f"[Rank {rank}] Loaded full model for cloud-base KV prefill.")

    print(f"[Rank {rank}] Waiting for hidden states from Rank {rank - 1}...")
    while True:
        initial_past_key_values = None
        initial_metric = None
        if args.prefill_mode == "cloud-base":
            if rank == world_size - 1:
                initial_past_key_values, initial_metric = run_rank2_cloud_base_prefill(
                    args,
                    model,
                    full_prefill_model,
                    world_size,
                    device,
                    layer_start,
                    layer_end,
                    batch_number,
                    boundaries,
                    comm_dtype or getattr(model, "dtype", torch.float16),
                    environment=environment,
                )
            else:
                initial_past_key_values, initial_metric = receive_cloud_base_cache_metric(
                    args,
                    model,
                    rank,
                    world_size,
                    device,
                    layer_start,
                    layer_end,
                    batch_number,
                    environment=environment,
                )

        should_continue = handle_worker_batch(
            args,
            model,
            rank,
            world_size,
            device,
            layer_start,
            layer_end,
            batch_number,
            initial_past_key_values=initial_past_key_values,
            initial_metric=initial_metric,
            environment=environment,
        )
        if not should_continue:
            break
        print(f"[Rank {rank}] Batch {batch_number} complete.")
        batch_number += 1
    print(f"[Rank {rank}] Stop signal received.")


def pipeline_serve_dynamic(args, rank, world_size, dtype, device, environment=None):
    """Dynamic-loading service loop for non-master pipeline ranks."""
    total_layers = get_total_layers_from_config(args.model_dir)
    default_boundaries = default_boundaries_for_world_size(args, world_size, total_layers)
    scheduler = None
    if args.allocation_csv:
        scheduler = Scheduler(args.allocation_csv, total_layers, default_boundaries, world_size)
    batch_number = 1
    model = None
    layer_partition = None
    full_prefill_model = None
    current_stage = None

    try:
        while True:
            boundaries = broadcast_boundaries(None, world_size, device, rank=rank)
            if boundaries[0] == -1:
                break
            print(
                f"[Rank {rank}] Batch {batch_number}: received boundaries "
                f"{boundaries}; prefill_mode={args.prefill_mode}"
            )
            if environment is not None:
                environment = broadcast_environment(environment, rank=rank, device=device)
                print(f"[Rank {rank}] Batch {batch_number}: environment={environment.describe()}")

            if scheduler is None:
                allocation = None
            else:
                allocation = scheduler.record_allocation(batch_number, boundaries)
            layer_start, layer_end = stage_from_boundaries(boundaries, rank)
            next_stage = (layer_start, layer_end)
            if allocation is None:
                interval_text = interval_text_from_boundaries(boundaries, world_size)
            else:
                interval_text = " ".join(
                    f"rank{stage_rank}={allocation.interval_for_rank(stage_rank)}"
                    for stage_rank in range(world_size)
                )
            print(f"[Rank {rank}] Batch {batch_number}: {interval_text}")

            if model is None:
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
                layer_partition = IncrementalLayerPartition.from_loaded_model(
                    model=model,
                    model_dir=args.model_dir,
                    rank=rank,
                    world_size=world_size,
                    layer_start=layer_start,
                    layer_end=layer_end,
                    dtype=dtype,
                    device=device,
                    batch_number=batch_number,
                )
                current_stage = next_stage
                print(
                    f"[Rank {rank}] Batch {batch_number} model loaded; "
                    f"stage=[{layer_start},{layer_end}); load_mode={load_mode}"
                )
            elif next_stage != current_stage:
                switch_result = layer_partition.switch_to(
                    layer_start,
                    layer_end,
                    batch_number,
                )
                model = layer_partition.model
                current_stage = next_stage
                print(
                    f"[Rank {rank}] Batch {batch_number}: incremental split "
                    f"{switch_result.old_stage}->{switch_result.new_stage}; "
                    f"retained={switch_result.retained_layers}; "
                    f"cache_hits={switch_result.cache_hit_layers}; "
                    f"local_loaded={switch_result.loaded_layers}; "
                    f"inactive_cache={layer_partition.inactive_layer_ids()}; "
                    f"evicted={switch_result.evicted_layers}; "
                    f"elapsed_ms={switch_result.elapsed_ms:.2f}."
                )
            else:
                print(
                    f"[Rank {rank}] Batch {batch_number}: reuse cached model partition "
                    f"for stage=[{layer_start},{layer_end})."
                )

            # Rank 2's cloud-base full model is also a batch prerequisite. Load
            # it before the barrier so Rank 0/1 do not enter P2P communication
            # while Rank 2 is still reading the checkpoint.
            if (
                args.prefill_mode == "cloud-base"
                and rank == world_size - 1
                and full_prefill_model is None
            ):
                print(f"[Rank {rank}] Loading full model for cloud-base KV prefill...")
                full_prefill_model = load_full_model_for_prefill(
                    args.model_dir,
                    dtype,
                    device,
                )
                print(f"[Rank {rank}] Loaded full model for cloud-base KV prefill.")

            wait_for_all_model_partitions(rank, batch_number, device)

            initial_past_key_values = None
            initial_metric = None
            if args.prefill_mode == "cloud-base":
                if rank == world_size - 1:
                    initial_past_key_values, initial_metric = run_rank2_cloud_base_prefill(
                        args,
                        model,
                        full_prefill_model,
                        world_size,
                        device,
                        layer_start,
                        layer_end,
                        batch_number,
                        boundaries,
                        getattr(model, "dtype", torch.float16),
                        environment=environment,
                    )
                else:
                    initial_past_key_values, initial_metric = receive_cloud_base_cache_metric(
                        args,
                        model,
                        rank,
                        world_size,
                        device,
                        layer_start,
                        layer_end,
                        batch_number,
                        environment=environment,
                    )

            should_continue = handle_worker_batch(
                args,
                model,
                rank,
                world_size,
                device,
                layer_start,
                layer_end,
                batch_number,
                initial_past_key_values=initial_past_key_values,
                initial_metric=initial_metric,
                environment=environment,
            )
            if not should_continue:
                return

            print(f"[Rank {rank}] Batch {batch_number} complete.")
            batch_number += 1

        print(f"[Rank {rank}] Dynamic loading stop signal received.")
    finally:
        if layer_partition is not None:
            layer_partition.release()
            model = None
        release_model(model)
        release_model(full_prefill_model)
