import argparse


def parse_args():
    parser = argparse.ArgumentParser(description="Distributed inference benchmark runner")
    parser.add_argument("--model_id", type=str, default="/home/dingcong/models/google/gemma-2-2b-it")
    parser.add_argument("--input_file", type=str, default="input.txt")
    parser.add_argument("--output_file", type=str, default="outputs.txt")
    parser.add_argument("--summary_file", type=str, default="single_run_summary.txt")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--sequence_length", type=int, default=128)
    parser.add_argument("--num_cpu_layers", type=int, default=3)
    parser.add_argument("--mid_gpu_layer", type=int, default=12)
    parser.add_argument("--total_layers", type=int, default=26)
    parser.add_argument("--max_new_tokens", type=int, default=125)
    parser.add_argument("--simulated_bandwidth_bps", type=float, default=1_000_000_000)
    parser.add_argument("--simulated_gpu_to_gpu_bandwidth_bps", type=float, default=50_000_000_000)
    parser.add_argument("--fixed_latency_sec", type=float, default=0.0001)
    parser.add_argument(
        "--build_dataset",
        action="store_true",
        help="Build synthetic prompts to match batch_size from the input prompt pool",
    )
    return parser.parse_args()


def validate_args(args):
    if args.num_cpu_layers < 1:
        raise ValueError("--num_cpu_layers must be >= 1 so hook registration is valid.")
    if args.mid_gpu_layer < 1 or args.mid_gpu_layer > args.total_layers:
        raise ValueError("--mid_gpu_layer must be in [1, total_layers].")
    if args.num_cpu_layers > args.total_layers:
        raise ValueError("--num_cpu_layers must be <= total_layers.")


def build_device_map(args):
    return {
        "model.embed_tokens": "cpu",
        **{f"model.layers.{i}": "cpu" for i in range(args.num_cpu_layers)},
        **{f"model.layers.{i}": "cuda:0" for i in range(args.num_cpu_layers, args.mid_gpu_layer)},
        **{f"model.layers.{i}": "cuda:0" for i in range(args.mid_gpu_layer, args.total_layers)},
        "model.norm": "cuda:0",
        "lm_head": "cuda:0",
    }
