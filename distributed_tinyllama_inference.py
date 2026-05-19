"""
TinyLlama two-node pipeline inference demo.

This script is intentionally written as a small, explicit pipeline instead of
using a high-level serving framework. The goal is to show how a causal language
model can be split across two NCCL ranks:

- Rank 0 owns input/output work: CSV prompts, tokenizer, token generation loop,
  embedding, and the first decoder layers.
- Rank 1 owns the later decoder layers, final norm, and lm_head.
- Rank 0 sends intermediate hidden states to Rank 1.
- Rank 1 sends the selected next token back to Rank 0.

Important note: each rank currently loads the full Hugging Face checkpoint first,
then removes the layers it does not execute. That keeps the code simple and easy
to verify. It means startup still reads the full model on both nodes.
"""

import argparse
import csv
import inspect
import os
import socket
from datetime import timedelta
from pathlib import Path

import torch
import torch.distributed as dist
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer


# Default distributed rendezvous address. Rank 1 connects to this address when
# torch.distributed initializes the process group. It can also be overridden by
# --init-method or the DIST_INIT_METHOD environment variable.
DEFAULT_INIT_METHOD = "tcp://10.50.0.57:29500"

# Rank 1 starts from this decoder layer index. With the default value 5:
# Rank 0 runs layers 0, 1, 2, 3, 4; Rank 1 runs layers 5 ... last.
DEFAULT_SPLIT_LAYER = 5


# -----------------------------------------------------------------------------
# Command-line configuration
# -----------------------------------------------------------------------------

def parse_args():
    """Parse runtime options shared by both ranks.

    The same script is launched on both nodes. The RANK environment variable
    decides which half of the pipeline this process runs.
    """
    parser = argparse.ArgumentParser(
        description="Run TinyLlama pipeline inference across two NCCL ranks."
    )
    parser.add_argument(
        "--model-dir",
        default="model/tinyllama",
        help="Local TinyLlama model directory. Default: model/tinyllama",
    )
    parser.add_argument(
        "--input-csv",
        default="prompts.csv",
        help="CSV file. One row is one prompt. Default: prompts.csv",
    )
    parser.add_argument(
        "--output-csv",
        default="outputs.csv",
        help="Rank 0 writes generated results here. Default: outputs.csv",
    )
    parser.add_argument(
        "--prompt-column",
        default=None,
        help=(
            "Prompt column name when --csv-has-header is set, or zero-based column "
            "index when there is no header. Default: first column."
        ),
    )
    parser.add_argument(
        "--csv-has-header",
        action="store_true",
        help="Treat the first CSV row as a header row.",
    )
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=64,
        help="Maximum generated tokens for each prompt. Default: 64",
    )
    parser.add_argument(
        "--max-input-tokens",
        type=int,
        default=1024,
        help="Truncate prompts to this many tokens. Default: 1024",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.0,
        help="0 means greedy decoding. Values > 0 enable sampling. Default: 0",
    )
    parser.add_argument(
        "--split-layer",
        type=int,
        default=DEFAULT_SPLIT_LAYER,
        help="Layer index where Rank 1 starts. Default: 5",
    )
    parser.add_argument(
        "--init-method",
        default=os.environ.get("DIST_INIT_METHOD", DEFAULT_INIT_METHOD),
        help=f"torch.distributed init method. Default: {DEFAULT_INIT_METHOD}",
    )
    parser.add_argument(
        "--timeout-seconds",
        type=int,
        default=120,
        help="Distributed initialization timeout. Default: 120",
    )
    parser.add_argument(
        "--dtype",
        choices=("auto", "float16", "bfloat16", "float32"),
        default="float16",
        help="Model dtype. Default: float16",
    )
    return parser.parse_args()


# -----------------------------------------------------------------------------
# Distributed environment and model loading
# -----------------------------------------------------------------------------

def get_rank_world_size():
    """Read and validate the distributed identity of this process.

    RANK and WORLD_SIZE are set outside Python so the exact same command can be
    used on both machines except for RANK. This demo is deliberately limited to
    two ranks because the model is split into two pipeline stages.
    """
    try:
        rank = int(os.environ["RANK"])
        world_size = int(os.environ["WORLD_SIZE"])
    except KeyError as exc:
        raise RuntimeError("Please set RANK and WORLD_SIZE before running this script.") from exc

    if world_size != 2:
        raise RuntimeError("This script expects WORLD_SIZE=2.")
    if rank not in (0, 1):
        raise RuntimeError("This script only supports RANK=0 or RANK=1.")
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


def load_model_part(model_dir, rank, split_layer, dtype, device):
    """Load TinyLlama and keep only the layers this rank will execute.

    This function does not implement true checkpoint sharding. Both ranks call
    from_pretrained(), so both ranks read the full checkpoint from disk. After
    loading, we replace model.model.layers with a shorter ModuleList:

    - Rank 0 keeps layers before split_layer.
    - Rank 1 keeps layers from split_layer onward.

    The benefit is that the code stays close to the standard Hugging Face model
    object and is easy to debug. The tradeoff is that startup memory and disk IO
    are not reduced yet.
    """
    model = AutoModelForCausalLM.from_pretrained(
        model_dir,
        torch_dtype=dtype,
        local_files_only=True,
        low_cpu_mem_usage=True,
    )
    model.eval()

    total_layers = len(model.model.layers)
    if split_layer <= 0 or split_layer >= total_layers:
        raise ValueError(
            f"--split-layer must be between 1 and {total_layers - 1}; got {split_layer}."
        )

    if rank == 0:
        model.model.layers = nn.ModuleList(list(model.model.layers[:split_layer]))
    else:
        model.model.layers = nn.ModuleList(list(model.model.layers[split_layer:]))

    model.to(device)
    return model, total_layers


# -----------------------------------------------------------------------------
# TinyLlama forward helpers
# -----------------------------------------------------------------------------

def maybe_rotary_embeddings(model, hidden_states, position_ids):
    """Build rotary position embeddings when the installed Transformers needs them.

    Different Transformers versions expose Llama/TinyLlama decoder-layer forward
    signatures slightly differently. Newer versions may pass precomputed rotary
    embeddings through position_embeddings; older versions only use position_ids.
    Returning None is fine for the older path.
    """
    rotary = getattr(model.model, "rotary_emb", None)
    if rotary is None:
        return None
    try:
        return rotary(hidden_states, position_ids)
    except TypeError:
        return None


def make_causal_mask(batch_size, seq_len, dtype, device):
    """Create a standard causal attention mask.

    Shape is [batch, heads, query_length, key_length]. Values above the diagonal
    are set to a very negative number so a token cannot attend to future tokens.
    """
    min_value = torch.finfo(dtype).min
    mask = torch.full((seq_len, seq_len), min_value, dtype=dtype, device=device)
    mask = torch.triu(mask, diagonal=1)
    return mask.view(1, 1, seq_len, seq_len).expand(batch_size, 1, seq_len, seq_len)


def run_decoder_layers(model, hidden_states, position_ids, attention_mask):
    """Run whichever decoder layers remain in this rank's model object.

    The same helper is used by both ranks. Rank 0's model contains only early
    layers; Rank 1's model contains only later layers. The inspect.signature()
    logic makes this script tolerate small API differences across Transformers
    versions without changing the core distributed logic.
    """
    position_embeddings = maybe_rotary_embeddings(model, hidden_states, position_ids)
    cache_position = torch.arange(hidden_states.shape[1], device=hidden_states.device)

    for layer in model.model.layers:
        signature = inspect.signature(layer.forward)
        kwargs = {}
        if "attention_mask" in signature.parameters:
            kwargs["attention_mask"] = attention_mask
        if "position_ids" in signature.parameters:
            kwargs["position_ids"] = position_ids
        if "past_key_value" in signature.parameters:
            kwargs["past_key_value"] = None
        if "output_attentions" in signature.parameters:
            kwargs["output_attentions"] = False
        if "use_cache" in signature.parameters:
            kwargs["use_cache"] = False
        if "cache_position" in signature.parameters:
            kwargs["cache_position"] = cache_position
        if "position_embeddings" in signature.parameters and position_embeddings is not None:
            kwargs["position_embeddings"] = position_embeddings

        layer_outputs = layer(hidden_states, **kwargs)
        hidden_states = layer_outputs[0] if isinstance(layer_outputs, tuple) else layer_outputs

    return hidden_states


def rank0_forward(model, input_ids, device):
    """Run Rank 0's part of the model and return hidden states for Rank 1.

    Rank 0 starts from token ids, so it must apply token embedding first. It then
    runs the early decoder layers and sends the resulting hidden_states tensor to
    Rank 1 through NCCL.
    """
    batch_size, seq_len = input_ids.shape
    position_ids = torch.arange(seq_len, device=device, dtype=torch.long).unsqueeze(0)
    attention_mask = make_causal_mask(batch_size, seq_len, model.dtype, device)

    hidden_states = model.model.embed_tokens(input_ids)
    hidden_states = run_decoder_layers(model, hidden_states, position_ids, attention_mask)
    return hidden_states.contiguous()


def rank1_forward_logits(model, hidden_states, device):
    """Run Rank 1's part of the model and return logits for the last token.

    Rank 1 receives hidden states, not token ids. Therefore it skips embeddings,
    runs the later decoder layers, applies final norm and lm_head, then returns
    only the last-token logits needed to choose the next generated token.
    """
    batch_size, seq_len, _ = hidden_states.shape
    position_ids = torch.arange(seq_len, device=device, dtype=torch.long).unsqueeze(0)
    attention_mask = make_causal_mask(batch_size, seq_len, model.dtype, device)

    hidden_states = run_decoder_layers(model, hidden_states, position_ids, attention_mask)
    hidden_states = model.model.norm(hidden_states)
    logits = model.lm_head(hidden_states)
    return logits[:, -1, :]


# -----------------------------------------------------------------------------
# NCCL point-to-point message protocol
# -----------------------------------------------------------------------------

def send_stop(device):
    """Tell Rank 1 that Rank 0 has no more prompts to process.

    The first value in meta is a status code:
    - 0 means a hidden_states tensor will follow.
    - 1 means stop serving and exit the receive loop.
    """
    meta = torch.tensor([1, 0, 0, 0], dtype=torch.long, device=device)
    dist.send(meta, dst=1)


def send_hidden_to_rank1(hidden_states):
    """Send tensor metadata first, then the hidden_states tensor itself.

    dist.recv() needs the receiver to allocate a correctly shaped tensor before
    receiving payload data. The small meta tensor carries [status, batch, seq,
    hidden_size] so Rank 1 knows exactly what buffer to allocate.
    """
    batch_size, seq_len, hidden_size = hidden_states.shape
    meta = torch.tensor([0, batch_size, seq_len, hidden_size], dtype=torch.long, device=hidden_states.device)
    dist.send(meta, dst=1)
    dist.send(hidden_states, dst=1)


def recv_hidden_from_rank0(device, dtype):
    """Receive one message from Rank 0.

    Returns a CUDA hidden_states tensor, or None when Rank 0 sends the stop code.
    """
    meta = torch.empty(4, dtype=torch.long, device=device)
    dist.recv(meta, src=0)
    status, batch_size, seq_len, hidden_size = meta.tolist()
    if status == 1:
        return None
    if status != 0:
        raise RuntimeError(f"Unknown message status from Rank 0: {status}")

    hidden_states = torch.empty(
        (batch_size, seq_len, hidden_size), dtype=dtype, device=device
    )
    dist.recv(hidden_states, src=0)
    return hidden_states


def choose_next_token(logits, temperature):
    """Convert last-token logits into one token id.

    temperature=0 uses greedy decoding. A positive temperature samples from the
    softmax distribution, which makes output less deterministic.
    """
    if temperature and temperature > 0:
        probs = torch.softmax(logits / temperature, dim=-1)
        return torch.multinomial(probs, num_samples=1)
    return torch.argmax(logits, dim=-1, keepdim=True)


# -----------------------------------------------------------------------------
# CSV input and output
# -----------------------------------------------------------------------------

def read_prompts(csv_path, has_header, prompt_column):
    """Read prompts from CSV.

    Supported formats:
    - Without header: read the first column by default, or --prompt-column as a
      zero-based column index.
    - With header: read --prompt-column as a column name, or use the first header
      column if --prompt-column is omitted.
    """
    prompts = []
    with open(csv_path, "r", encoding="utf-8-sig", newline="") as f:
        if has_header:
            reader = csv.DictReader(f)
            if prompt_column is None:
                if not reader.fieldnames:
                    return []
                prompt_column = reader.fieldnames[0]
            for row in reader:
                value = (row.get(prompt_column) or "").strip()
                if value:
                    prompts.append(value)
        else:
            column_index = int(prompt_column) if prompt_column is not None else 0
            reader = csv.reader(f)
            for row in reader:
                if len(row) <= column_index:
                    continue
                value = row[column_index].strip()
                if value:
                    prompts.append(value)
    return prompts


# -----------------------------------------------------------------------------
# Rank-specific execution loops
# -----------------------------------------------------------------------------

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

    output_path = Path(args.output_csv)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    eos_token_id = tokenizer.eos_token_id
    rows = []

    for index, prompt in enumerate(prompts, start=1):
        encoded = tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=args.max_input_tokens,
        )
        input_ids = encoded["input_ids"].to(device)
        prompt_len = input_ids.shape[1]

        print(f"[Rank 0] Prompt {index}/{len(prompts)}: {prompt}")

        with torch.inference_mode():
            for _ in range(args.max_new_tokens):
                hidden_states = rank0_forward(model, input_ids, device)
                send_hidden_to_rank1(hidden_states)

                next_token = torch.empty((1, 1), dtype=torch.long, device=device)
                dist.recv(next_token, src=1)
                input_ids = torch.cat([input_ids, next_token], dim=1)

                if eos_token_id is not None and int(next_token.item()) == eos_token_id:
                    break

        generated_ids = input_ids[0, prompt_len:]
        generated_text = tokenizer.decode(generated_ids, skip_special_tokens=True)
        full_text = tokenizer.decode(input_ids[0], skip_special_tokens=True)
        rows.append(
            {
                "prompt": prompt,
                "generated_text": generated_text,
                "full_text": full_text,
            }
        )
        print(f"[Rank 0] Output {index}: {generated_text}")

    with open(output_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["prompt", "generated_text", "full_text"])
        writer.writeheader()
        writer.writerows(rows)

    print(f"[Rank 0] Wrote {len(rows)} rows to {output_path}")
    send_stop(device)


def rank1_serve(args, model, device):
    """Rank 1 service loop.

    Rank 1 does not read prompts and does not write the output CSV. It waits for
    hidden states from Rank 0, runs the second half of the model, sends one token
    id back, and repeats until a stop message arrives.
    """
    print("[Rank 1] Waiting for hidden states from Rank 0...")
    with torch.inference_mode():
        while True:
            hidden_states = recv_hidden_from_rank0(device, model.dtype)
            if hidden_states is None:
                break
            logits = rank1_forward_logits(model, hidden_states, device)
            next_token = choose_next_token(logits, args.temperature).to(torch.long).contiguous()
            dist.send(next_token, dst=0)
    print("[Rank 1] Stop signal received.")


# -----------------------------------------------------------------------------
# Program entry point
# -----------------------------------------------------------------------------

def main():
    """Initialize NCCL, load this rank's model part, then run the right loop."""
    args = parse_args()
    rank, world_size = get_rank_world_size()

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for NCCL inference, but torch.cuda.is_available() is False.")

    device = torch.device("cuda:0")
    torch.cuda.set_device(device)

    print(f"[Rank {rank}] Starting on host {socket.gethostname()}")
    print(f"[Rank {rank}] init_method={args.init_method}")

    # NCCL is used because all tensors passed between ranks are CUDA tensors.
    # The init_method address must be reachable from both nodes.
    dist.init_process_group(
        backend="nccl",
        init_method=args.init_method,
        rank=rank,
        world_size=world_size,
        timeout=timedelta(seconds=args.timeout_seconds),
    )

    dtype = resolve_dtype(args.dtype)
    model, total_layers = load_model_part(args.model_dir, rank, args.split_layer, dtype, device)
    print(
        f"[Rank {rank}] Loaded TinyLlama from {args.model_dir}; "
        f"total_layers={total_layers}; split_layer={args.split_layer}"
    )

    try:
        if rank == 0:
            tokenizer = AutoTokenizer.from_pretrained(args.model_dir, local_files_only=True)
            if tokenizer.pad_token_id is None and tokenizer.eos_token_id is not None:
                tokenizer.pad_token = tokenizer.eos_token
            rank0_generate(args, model, tokenizer, device)
        else:
            rank1_serve(args, model, device)

        # Synchronize before shutdown so both ranks finish cleanly.
        dist.barrier()
        print(f"[Rank {rank}] SUCCESS")
    finally:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
