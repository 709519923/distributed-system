"""Command-line options and layer partition helpers.

This module is the first place to check when a run uses an unexpected split,
batch size, model path, rendezvous address, or CUDA device. It intentionally
contains no NCCL calls and no model loading, so configuration problems can be
reasoned about without touching distributed state.
"""

import argparse
import os


DEFAULT_INIT_METHOD = "tcp://10.50.1.228:29500"
DEFAULT_SPLIT_LAYER = 5
DEFAULT_SECOND_SPLIT_LAYER = 15

# Small message protocol used between neighboring pipeline ranks.
STATUS_HIDDEN = 0
STATUS_STOP = 1
STATUS_BATCH_DONE = 2


def parse_args():
    """Parse runtime options shared by both ranks.

    The same script is launched on both nodes. The RANK environment variable
    decides which half of the pipeline this process runs.
    """
    parser = argparse.ArgumentParser(
        description="Run TinyLlama pipeline inference across two or three NCCL ranks."
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
        "--context-manifest",
        default=None,
        help=(
            "DEF interleaved sidecar CSV with per-batch scenario, request type, "
            "and target output tokens. Requires --bandit-policy contextual."
        ),
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
        default=512,
        help="Maximum generated tokens for each prompt. Default: 512",
    )
    parser.add_argument(
        "--force-decode-steps",
        type=int,
        default=None,
        help=(
            "Ignore EOS and force this many decode forward steps after prefill. "
            "When set, --max-new-tokens is not used as the decode loop limit."
        ),
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
        help="Two-node split layer, or first split in three-node mode. Default: 5",
    )
    parser.add_argument(
        "--split-layers",
        default=None,
        help=(
            "Comma-separated split layers. Use one value for WORLD_SIZE=2 "
            "(example: 5), or two values for WORLD_SIZE=3 (example: 5,15)."
        ),
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
    parser.add_argument(
        "--lazy-load",
        action="store_true",
        help=(
            "Only load the checkpoint tensors used by this rank. "
            "Requires a safetensors-format Hugging Face checkpoint."
        ),
    )
    parser.add_argument(
        "--dynamic-load",
        action="store_true",
        help=(
            "Process prompts in batches and reload the rank-local layer partition "
            "only when Scheduler changes the split point. Requires --lazy-load."
        ),
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=64,
        help="Number of prompts per tensor batch in both normal and dynamic-loading mode. Default: 64",
    )
    parser.add_argument(
        "--allocation-csv",
        default=None,
        help=(
            "Rank 0 Scheduler CSV path. When provided, missing batches are created "
            "from the latest allocation, starting with --split-layers."
        ),
    )
    parser.add_argument(
        "--bandit-policy",
        choices=(
            "ucb",
            "contextual",
            "contextual_controlled",
            "contextual_woscenario",
            "lipschitz",
            "contextual_lipschitz",
        ),
        default="ucb",
        help=(
            "Rank 0 scheduler bandit policy. Default: ucb. contextual uses "
            "current-batch prompt context before selecting the current arm. "
            "contextual_controlled isolates online models by inferred DEF key. "
            "contextual_woscenario pools all controlled DEF batches into one "
            "intercept-only online model."
        ),
    )
    parser.add_argument(
        "--top-k-arms",
        type=int,
        default=5,
        help=(
            "Number of highest-scoring arms to store per scheduler snapshot. "
            "Default: 5."
        ),
    )
    parser.add_argument(
        "--experiment-scenario",
        type=lambda value: value.strip().upper(),
        choices=("A", "B", "C", "D", "E", "F"),
        default=None,
        help=(
            "Single-scenario controlled run for ucb or lipschitz. The selected "
            "A-F scenario fixes the generated token count for every prompt."
        ),
    )
    parser.add_argument(
        "--cuda-device",
        default="0",
        help="CUDA device index used by this process. Default: 0",
    )
    parser.add_argument(
        "--compute-device",
        choices=("cuda", "cpu"),
        default="cuda",
        help=(
            "Rank 0 compute device. Default: cuda. Use cpu to run Rank 0 model "
            "compute on CPU while keeping CUDA/NCCL tensors for communication."
        ),
    )
    parser.add_argument(
        "--prefill-mode",
        choices=("distributed", "cloud-base"),
        default="distributed",
        help=(
            "KV-cache prefill strategy. distributed keeps the current behavior "
            "where each rank prefill-computes its own layer partition. cloud-base "
            "lets Rank 2 compute the full KV cache and transfer layer partitions "
            "to Rank 0 and Rank 1."
        ),
    )
    args = parser.parse_args()
    if args.force_decode_steps is not None and args.force_decode_steps < 0:
        parser.error("--force-decode-steps must be a non-negative integer.")
    if args.top_k_arms < 1:
        parser.error("--top-k-arms must be a positive integer.")
    return args


def parse_split_layers(value):
    """Parse --split-layers into a list of integer split points."""
    if value is None:
        return None
    splits = [item.strip() for item in value.split(",") if item.strip()]
    if not splits:
        return None
    return [int(item) for item in splits]


def default_boundaries_for_world_size(args, world_size, total_layers):
    """Build [0, split..., total_layers] for WORLD_SIZE=2 or WORLD_SIZE=3."""
    explicit_splits = parse_split_layers(args.split_layers)

    if explicit_splits is None:
        if world_size == 2:
            explicit_splits = [args.split_layer]
        else:
            second_split = DEFAULT_SECOND_SPLIT_LAYER
            if second_split <= args.split_layer or second_split >= total_layers:
                second_split = max(args.split_layer + 1, (2 * total_layers) // 3)
            explicit_splits = [args.split_layer, second_split]

    expected_count = world_size - 1
    if len(explicit_splits) != expected_count:
        raise ValueError(
            f"WORLD_SIZE={world_size} expects {expected_count} split value(s); "
            f"got {explicit_splits}"
        )

    boundaries = [0] + explicit_splits + [total_layers]
    validate_boundaries(boundaries, world_size, total_layers)
    return boundaries


def validate_boundaries(boundaries, world_size, total_layers):
    """Validate a full pipeline boundary list."""
    if len(boundaries) != world_size + 1:
        raise ValueError(f"Expected {world_size + 1} boundaries, got {boundaries}")
    if boundaries[0] != 0 or boundaries[-1] != total_layers:
        raise ValueError(f"Boundaries must start at 0 and end at {total_layers}: {boundaries}")
    for left, right in zip(boundaries, boundaries[1:]):
        if left >= right:
            raise ValueError(f"Boundaries must be strictly increasing: {boundaries}")


def stage_from_boundaries(boundaries, rank):
    """Return (layer_start, layer_end) for this rank."""
    return int(boundaries[rank]), int(boundaries[rank + 1])
