"""High-level inference loops for master and pipeline worker ranks.

KV cache is now the default inference behavior. Every prompt batch has one
prefill pass that builds rank-local caches, followed by one-token decode passes
that reuse those caches until all rows hit EOS or --max-new-tokens is reached.
"""

import gc
import time

import torch

from config import STATUS_BATCH_DONE, default_boundaries_for_world_size, stage_from_boundaries
from csv_io import chunk_items, read_prompts, write_output_rows
from experiment_report import (
    append_experiment_log,
    make_log_path,
    normalize_record,
    metric_to_tensor,
    recv_metric_records,
    recv_metric_tensor,
    send_metric_record,
    send_metric_tensor,
)
from kv_cache_utils import (
    count_model_parameters,
    cuda_memory_allocated,
    cuda_memory_reserved,
    estimate_parameter_bytes,
    estimate_past_key_values_bytes,
    synchronize_cuda,
    tensor_bytes,
)
from model_forward import choose_next_token, rank0_forward, rank1_forward_logits, rank_middle_forward
from model_loader import get_total_layers_from_config, load_model_part
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
from scheduler import Scheduler


def build_prefill_metric(
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
    hidden_states,
    memory_before,
    memory_after,
    memory_reserved_after,
    prefill_time_ms,
):
    """Build one record for the text experiment log."""
    hidden_shape = list(hidden_states.shape)
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
        "prefill_param_count": count_model_parameters(model),
        "prefill_param_bytes": estimate_parameter_bytes(model),
        "kv_cache_bytes_after_prefill": estimate_past_key_values_bytes(past_key_values),
        "hidden_prefill_batch": hidden_shape[0],
        "hidden_prefill_seq_len": hidden_shape[1],
        "hidden_prefill_hidden_size": hidden_shape[2],
        "hidden_prefill_bytes": tensor_bytes(hidden_states),
        "cuda_memory_allocated_before_prefill": memory_before,
        "cuda_memory_allocated_after_prefill": memory_after,
        "cuda_memory_reserved_after_prefill": memory_reserved_after,
        "prefill_time_ms": prefill_time_ms,
        "decode_time_per_token_ms": 0.0,
    }


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
        memory_before = cuda_memory_allocated(device)
        start_time = time.perf_counter()
        hidden_states, rank0_past_key_values = rank0_forward(
            model,
            input_ids,
            device,
            attention_mask_2d,
            past_key_values=None,
        )
        synchronize_cuda()
        prefill_time_ms = (time.perf_counter() - start_time) * 1000.0
        memory_after = cuda_memory_allocated(device)
        memory_reserved_after = cuda_memory_reserved(device)

        send_hidden(hidden_states, dst=1, attention_mask_2d=attention_mask_2d)
        next_token = recv_token(src=1, device=device, batch_size=batch_size)

        rank0_metric = build_prefill_metric(
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
            hidden_states=hidden_states,
            memory_before=memory_before,
            memory_after=memory_after,
            memory_reserved_after=memory_reserved_after,
            prefill_time_ms=prefill_time_ms,
        )
        records = [normalize_record(rank0_metric, str(model.dtype))]
        rank0_decode_time_ms = 0.0
        rank0_decode_step_count = 0
        for token_index in range(args.max_new_tokens):
            synchronize_cuda()
            rank0_decode_start_time = time.perf_counter()
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

            attention_mask_2d = torch.cat([attention_mask_2d, mask_to_append], dim=1)
            if bool(finished.all().item()) or token_index == args.max_new_tokens - 1:
                break

            decode_input_ids = tokens_to_append
            hidden_states, rank0_past_key_values = rank0_forward(
                model,
                decode_input_ids,
                device,
                attention_mask_2d,
                past_key_values=rank0_past_key_values,
            )
            send_hidden(hidden_states, dst=1, attention_mask_2d=attention_mask_2d)
            synchronize_cuda()
            rank0_decode_time_ms += (time.perf_counter() - rank0_decode_start_time) * 1000.0
            rank0_decode_step_count += 1
            next_token = recv_token(src=1, device=device, batch_size=batch_size)

        rank0_decode_time_per_token_ms = (
            rank0_decode_time_ms / rank0_decode_step_count if rank0_decode_step_count else 0.0
        )
        records[0]["decode_time_per_token_ms"] = rank0_decode_time_per_token_ms

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


def release_model(model):
    """Release the current model partition before loading a different one."""
    if model is None:
        return
    del model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def rank0_generate(args, model, tokenizer, device, world_size, layer_start, layer_end):
    """Rank 0 static-split driver loop."""
    prompts = read_prompts(args.input_csv, args.csv_has_header, args.prompt_column)
    if not prompts:
        print(f"[Rank 0] No prompts found in {args.input_csv}")
        send_stop(device)
        return

    log_path = make_log_path()
    print(f"[Rank 0] KV-cache experiment log: {log_path}")
    all_rows = []
    for batch_number, start_index, prompt_batch in chunk_items(prompts, args.batch_size):
        print(f"[Rank 0] Static batch {batch_number}; prompts={len(prompt_batch)}")
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
        )
        all_rows.extend(rows)
        send_batch_done(device)
        records.extend(recv_metric_records(world_size, device, str(model.dtype)))
        append_experiment_log(log_path, records)

    write_output_rows(args.output_csv, all_rows)
    send_stop(device)


def boundaries_for_batch(args, scheduler, default_boundaries, batch_number):
    """Return fixed split boundaries unless --allocation-csv was explicitly set."""
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


def rank0_generate_dynamic(args, tokenizer, dtype, world_size, device):
    """Rank 0 dynamic-loading driver.

    If --allocation-csv is omitted, every batch uses --split-layers as a fixed
    partition. If --allocation-csv is provided, Scheduler controls each batch's
    partition. KV cache is always per-batch and is rebuilt after every batch.
    """
    prompts = read_prompts(args.input_csv, args.csv_has_header, args.prompt_column)
    if not prompts:
        print(f"[Rank 0] No prompts found in {args.input_csv}")
        broadcast_boundaries(stop_boundaries(world_size), world_size, device, rank=0)
        return

    total_layers = get_total_layers_from_config(args.model_dir)
    default_boundaries = default_boundaries_for_world_size(args, world_size, total_layers)
    scheduler = None
    if args.allocation_csv:
        scheduler = Scheduler(args.allocation_csv, total_layers, default_boundaries, world_size)

    log_path = make_log_path()
    print(f"[Rank 0] KV-cache experiment log: {log_path}")
    all_rows = []
    model = None
    current_stage = None

    try:
        for batch_number, start_index, prompt_batch in chunk_items(prompts, args.batch_size):
            boundaries, allocation = boundaries_for_batch(args, scheduler, default_boundaries, batch_number)
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
            )
            all_rows.extend(rows)

            send_batch_done(device)
            records.extend(recv_metric_records(world_size, device, str(model.dtype)))
            append_experiment_log(log_path, records)
            print(f"[Rank 0] Batch {batch_number} complete.")

        write_output_rows(args.output_csv, all_rows)
        broadcast_boundaries(stop_boundaries(world_size), world_size, device, rank=0)
    finally:
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
):
    """Serve one worker batch and keep rank-local KV cache until batch_done."""
    prev_rank = rank - 1
    next_rank = rank + 1
    is_last_rank = rank == world_size - 1
    past_key_values = None
    metric = None
    decode_time_ms = 0.0
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
                    metric["decode_time_per_token_ms"] = (
                        decode_time_ms / decode_step_count if decode_step_count else 0.0
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
                memory_before = cuda_memory_allocated(device)
                start_time = time.perf_counter()
            else:
                synchronize_cuda()
                decode_start_time = time.perf_counter()

            if is_last_rank:
                logits, past_key_values = rank1_forward_logits(
                    model,
                    hidden_states,
                    device,
                    attention_mask_2d,
                    past_key_values=past_key_values,
                )
                output_hidden_states = hidden_states
                if is_prefill and metric is None:
                    synchronize_cuda()
                    prefill_time_ms = (time.perf_counter() - start_time) * 1000.0
                    memory_after = cuda_memory_allocated(device)
                    memory_reserved_after = cuda_memory_reserved(device)
                next_token = choose_next_token(logits, args.temperature).to(torch.long).contiguous()
                if not is_prefill:
                    synchronize_cuda()
                    decode_time_ms += (time.perf_counter() - decode_start_time) * 1000.0
                    decode_step_count += 1
            else:
                output_hidden_states, past_key_values = rank_middle_forward(
                    model,
                    hidden_states,
                    device,
                    attention_mask_2d,
                    past_key_values=past_key_values,
                )
                if is_prefill and metric is None:
                    synchronize_cuda()
                    prefill_time_ms = (time.perf_counter() - start_time) * 1000.0
                    memory_after = cuda_memory_allocated(device)
                    memory_reserved_after = cuda_memory_reserved(device)
                send_hidden(output_hidden_states, dst=next_rank, attention_mask_2d=attention_mask_2d)
                if not is_prefill:
                    synchronize_cuda()
                    decode_time_ms += (time.perf_counter() - decode_start_time) * 1000.0
                    decode_step_count += 1
                next_token = recv_token(
                    src=next_rank,
                    device=device,
                    batch_size=output_hidden_states.shape[0],
                )

            if is_prefill and metric is None:
                metric = build_prefill_metric(
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
                    hidden_states=output_hidden_states,
                    memory_before=memory_before,
                    memory_after=memory_after,
                    memory_reserved_after=memory_reserved_after,
                    prefill_time_ms=prefill_time_ms,
                )

            send_token(next_token, dst=prev_rank)


def pipeline_serve_static(args, model, rank, world_size, device, layer_start, layer_end):
    """Serve a non-master pipeline rank in static mode."""
    batch_number = 1
    print(f"[Rank {rank}] Waiting for hidden states from Rank {rank - 1}...")
    while True:
        should_continue = handle_worker_batch(
            args,
            model,
            rank,
            world_size,
            device,
            layer_start,
            layer_end,
            batch_number,
        )
        if not should_continue:
            break
        print(f"[Rank {rank}] Batch {batch_number} complete.")
        batch_number += 1
    print(f"[Rank {rank}] Stop signal received.")


def pipeline_serve_dynamic(args, rank, world_size, dtype, device):
    """Dynamic-loading service loop for non-master pipeline ranks."""
    total_layers = get_total_layers_from_config(args.model_dir)
    default_boundaries = default_boundaries_for_world_size(args, world_size, total_layers)
    scheduler = None
    if args.allocation_csv:
        scheduler = Scheduler(args.allocation_csv, total_layers, default_boundaries, world_size)
    batch_number = 1
    model = None
    current_stage = None

    try:
        while True:
            boundaries = broadcast_boundaries(None, world_size, device, rank=rank)
            if boundaries[0] == -1:
                break

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

            should_continue = handle_worker_batch(
                args,
                model,
                rank,
                world_size,
                device,
                layer_start,
                layer_end,
                batch_number,
            )
            if not should_continue:
                return

            print(f"[Rank {rank}] Batch {batch_number} complete.")
            batch_number += 1

        print(f"[Rank {rank}] Dynamic loading stop signal received.")
    finally:
        release_model(model)
