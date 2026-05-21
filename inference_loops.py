"""High-level inference loops for master and pipeline worker ranks.

This module glues together CSV batches, Scheduler allocations, model loading,
forward passes, and NCCL communication. Use it when debugging run flow: which
batch is active, whether a partition was reused, or where a dynamic-loading
cycle stops.
"""

import gc

import torch

from config import default_boundaries_for_world_size, stage_from_boundaries, STATUS_BATCH_DONE
from csv_io import chunk_items, read_prompts, write_output_rows
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


def generate_rows_for_prompts(args, model, tokenizer, device, prompts, start_index=0, total_count=None):
    """Generate outputs for an in-memory list of prompts using the active model.

    This function is shared by normal mode and dynamic-loading mode. Rank 0 owns
    tokenization and receives one next-token id per prompt from Rank 1 for every
    generation step. The whole prompt list passed here is one tensor batch.
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
    finished = torch.zeros(batch_size, dtype=torch.bool, device=device)
    generated_tokens = [[] for _ in range(batch_size)]

    first_prompt_number = start_index + 1
    last_prompt_number = start_index + len(prompts)
    print(
        f"[Rank 0] Prompt batch {first_prompt_number}-{last_prompt_number}/"
        f"{total_count}; batch_size={batch_size}"
    )

    with torch.inference_mode():
        for _ in range(args.max_new_tokens):
            hidden_states = rank0_forward(model, input_ids, device, attention_mask_2d)
            send_hidden(hidden_states, dst=1, attention_mask_2d=attention_mask_2d)

            next_token = recv_token(src=1, device=device, batch_size=batch_size)
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

            input_ids = torch.cat([input_ids, tokens_to_append], dim=1)
            attention_mask_2d = torch.cat([attention_mask_2d, mask_to_append], dim=1)

            if bool(finished.all().item()):
                break

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

    return rows


def release_model(model):
    """Release the current model partition before loading a different one."""
    if model is None:
        return
    del model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def rank0_generate(args, model, tokenizer, device):
    """Rank 0 driver loop.

    For every prompt, Rank 0 repeatedly:
    1. Encodes or extends the current token sequence.
    2. Runs embeddings and early decoder layers.
    3. Sends hidden states to Rank 1.
    4. Receives one next-token id from Rank 1.
    5. Appends that token and continues until EOS or max_new_tokens.

    Only Rank 0 writes outputs.csv because it owns the original prompts and final
    generated token sequence.
    """
    prompts = read_prompts(args.input_csv, args.csv_has_header, args.prompt_column)
    if not prompts:
        print(f"[Rank 0] No prompts found in {args.input_csv}")
        send_stop(device)
        return

    all_rows = []
    for batch_number, start_index, prompt_batch in chunk_items(prompts, args.batch_size):
        print(f"[Rank 0] Static batch {batch_number}; prompts={len(prompt_batch)}")
        rows = generate_rows_for_prompts(
            args,
            model,
            tokenizer,
            device,
            prompt_batch,
            start_index=start_index,
            total_count=len(prompts),
        )
        all_rows.extend(rows)

    write_output_rows(args.output_csv, all_rows)
    send_stop(device)


def rank0_generate_dynamic(args, tokenizer, dtype, world_size, device):
    """Rank 0 dynamic-loading driver.

    Rank 0 reads all prompts, groups them into batches, asks Scheduler which
    layer split should be used for each batch, broadcasts that split to Rank 1,
    and keeps the current model partition cached. It only releases and reloads
    weights when the scheduler midpoint changes between batches.
    """
    prompts = read_prompts(args.input_csv, args.csv_has_header, args.prompt_column)
    if not prompts:
        print(f"[Rank 0] No prompts found in {args.input_csv}")
        broadcast_boundaries(stop_boundaries(world_size), world_size, device, rank=0)
        return

    total_layers = get_total_layers_from_config(args.model_dir)
    default_boundaries = default_boundaries_for_world_size(args, world_size, total_layers)
    scheduler = Scheduler(args.allocation_csv, total_layers, default_boundaries, world_size)
    all_rows = []
    model = None
    current_stage = None

    try:
        for batch_number, start_index, prompt_batch in chunk_items(prompts, args.batch_size):
            allocation = scheduler.get_or_create(batch_number)
            boundaries = allocation.boundaries
            layer_start, layer_end = stage_from_boundaries(boundaries, rank=0)
            next_stage = (layer_start, layer_end)
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

            rows = generate_rows_for_prompts(
                args,
                model,
                tokenizer,
                device,
                prompt_batch,
                start_index=start_index,
                total_count=len(prompts),
            )
            all_rows.extend(rows)

            send_batch_done(device)
            print(f"[Rank 0] Batch {batch_number} complete.")

        write_output_rows(args.output_csv, all_rows)
        broadcast_boundaries(stop_boundaries(world_size), world_size, device, rank=0)
    finally:
        release_model(model)


def pipeline_serve_static(args, model, rank, world_size, device):
    """Serve a non-master pipeline rank in static mode.

    Middle ranks relay hidden states forward and relay next-token ids backward.
    The last rank computes logits and samples/greedily chooses the next token.
    """
    prev_rank = rank - 1
    next_rank = rank + 1
    is_last_rank = rank == world_size - 1

    print(f"[Rank {rank}] Waiting for hidden states from Rank {prev_rank}...")
    with torch.inference_mode():
        while True:
            message = recv_hidden(src=prev_rank, device=device, dtype=model.dtype)
            if message is None:
                if not is_last_rank:
                    send_stop(device, dst=next_rank)
                break
            if isinstance(message, int) and message == STATUS_BATCH_DONE:
                if not is_last_rank:
                    send_batch_done(device, dst=next_rank)
                continue

            hidden_states, attention_mask_2d = message
            if is_last_rank:
                logits = rank1_forward_logits(model, hidden_states, device, attention_mask_2d)
                next_token = choose_next_token(logits, args.temperature).to(torch.long).contiguous()
            else:
                hidden_states = rank_middle_forward(model, hidden_states, device, attention_mask_2d)
                send_hidden(hidden_states, dst=next_rank, attention_mask_2d=attention_mask_2d)
                next_token = recv_token(
                    src=next_rank,
                    device=device,
                    batch_size=hidden_states.shape[0],
                )

            send_token(next_token, dst=prev_rank)
    print(f"[Rank {rank}] Stop signal received.")


def pipeline_serve_dynamic(args, rank, world_size, dtype, device):
    """Dynamic-loading service loop for non-master pipeline ranks.

    Each non-master rank receives boundaries from Rank 0, records them locally,
    and compares its own [layer_start, layer_end) with the currently cached model
    partition. It only reloads weights when that interval changes.
    """
    prev_rank = rank - 1
    next_rank = rank + 1
    is_last_rank = rank == world_size - 1
    total_layers = get_total_layers_from_config(args.model_dir)
    default_boundaries = default_boundaries_for_world_size(args, world_size, total_layers)
    scheduler = Scheduler(args.allocation_csv, total_layers, default_boundaries, world_size)
    batch_number = 1
    model = None
    current_stage = None

    try:
        while True:
            boundaries = broadcast_boundaries(None, world_size, device, rank=rank)
            if boundaries[0] == -1:
                break

            allocation = scheduler.record_allocation(batch_number, boundaries)
            layer_start, layer_end = stage_from_boundaries(boundaries, rank)
            next_stage = (layer_start, layer_end)
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

            with torch.inference_mode():
                while True:
                    message = recv_hidden(src=prev_rank, device=device, dtype=model.dtype)
                    if message is None:
                        if not is_last_rank:
                            send_stop(device, dst=next_rank)
                        return
                    if isinstance(message, int) and message == STATUS_BATCH_DONE:
                        if not is_last_rank:
                            send_batch_done(device, dst=next_rank)
                        break

                    hidden_states, attention_mask_2d = message
                    if is_last_rank:
                        logits = rank1_forward_logits(model, hidden_states, device, attention_mask_2d)
                        next_token = choose_next_token(logits, args.temperature).to(torch.long).contiguous()
                    else:
                        hidden_states = rank_middle_forward(model, hidden_states, device, attention_mask_2d)
                        send_hidden(hidden_states, dst=next_rank, attention_mask_2d=attention_mask_2d)
                        next_token = recv_token(
                            src=next_rank,
                            device=device,
                            batch_size=hidden_states.shape[0],
                        )

                    send_token(next_token, dst=prev_rank)

            print(f"[Rank {rank}] Batch {batch_number} complete.")
            batch_number += 1

        print(f"[Rank {rank}] Dynamic loading stop signal received.")
    finally:
        release_model(model)
