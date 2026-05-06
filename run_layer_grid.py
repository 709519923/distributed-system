import argparse
import datetime
import os
import re
import subprocess
import sys


def parse_int_list(text):
    values = []
    for part in text.split(","):
        part = part.strip()
        if not part:
            continue
        values.append(int(part))
    if not values:
        raise ValueError("Empty list is not allowed.")
    return values


def extract_input_sl(input_file):
    # Expected example: input_64.csv -> inputsl64
    base = os.path.basename(input_file)
    m = re.search(r"input_(\d+)\.csv$", base, flags=re.IGNORECASE)
    if m:
        return m.group(1)
    return "unknown"


def main():
    parser = argparse.ArgumentParser(
        description="Sweep num_cpu_layers and mid_gpu_layer for main5.py"
    )
    parser.add_argument("--python_bin", type=str, default=sys.executable)
    parser.add_argument("--main_script", type=str, default="main5.py")
    parser.add_argument("--model_id", type=str, required=True)
    parser.add_argument("--input_file", type=str, default="input.txt")
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--sequence_length", type=int, default=256)
    parser.add_argument("--max_new_tokens", type=int, default=125)
    parser.add_argument("--total_layers", type=int, default=26)
    parser.add_argument("--num_cpu_layers_list", type=str, default="1,3,12")
    parser.add_argument("--mid_gpu_layer_list", type=str, default="3,12,20")
    parser.add_argument("--enable_build_dataset", action="store_true")
    parser.add_argument("--outputs_dir", type=str, default="outputs_runs")
    parser.add_argument("--summaries_dir", type=str, default="summaries_runs")
    args = parser.parse_args()

    num_cpu_layers_values = parse_int_list(args.num_cpu_layers_list)
    mid_gpu_layer_values = parse_int_list(args.mid_gpu_layer_list)

    ts = datetime.datetime.now().strftime("%Y%m%d%H%M")
    outputs_dir = f"{args.outputs_dir}_{ts}"
    summaries_dir = f"{args.summaries_dir}_{ts}"

    os.makedirs(outputs_dir, exist_ok=True)
    os.makedirs(summaries_dir, exist_ok=True)
    input_sl = extract_input_sl(args.input_file)

    valid_pairs = []
    for num_cpu_layers in num_cpu_layers_values:
        for mid_gpu_layer in mid_gpu_layer_values:
            if num_cpu_layers < mid_gpu_layer:
                valid_pairs.append((num_cpu_layers, mid_gpu_layer))

    if not valid_pairs:
        raise ValueError(
            "No valid combinations found. Requirement: num_cpu_layers < mid_gpu_layer."
        )

    print(
        f"Total valid runs: {len(valid_pairs)} "
        f"(filtered by num_cpu_layers < mid_gpu_layer)"
    )

    total_runs = len(valid_pairs)

    for idx, (num_cpu_layers, mid_gpu_layer) in enumerate(valid_pairs, start=1):
        output_file = os.path.join(
            outputs_dir,
            f"bs{args.batch_size}_sl{args.sequence_length}_cpu{num_cpu_layers}_mid{mid_gpu_layer}_inputsl{input_sl}.txt",
        )
        summary_file = os.path.join(
            summaries_dir,
            f"summary_bs{args.batch_size}_sl{args.sequence_length}_cpu{num_cpu_layers}_mid{mid_gpu_layer}_inputsl{input_sl}.txt",
        )

        cmd = [
            args.python_bin,
            args.main_script,
            "--model_id",
            args.model_id,
            "--input_file",
            args.input_file,
            "--output_file",
            output_file,
            "--summary_file",
            summary_file,
            "--batch_size",
            str(args.batch_size),
            "--sequence_length",
            str(args.sequence_length),
            "--num_cpu_layers",
            str(num_cpu_layers),
            "--mid_gpu_layer",
            str(mid_gpu_layer),
            "--max_new_tokens",
            str(args.max_new_tokens),
            "--total_layers",
            str(args.total_layers),
        ]
        if args.enable_build_dataset:
            cmd.append("--build_dataset")

        print(f"[RUN] num_cpu_layers={num_cpu_layers}, mid_gpu_layer={mid_gpu_layer}")
        print(" ".join(cmd))
        subprocess.run(cmd, check=True)
        done = idx
        remaining = total_runs - done
        print(
            f"[PROGRESS] done={done}, remaining={remaining}, total={total_runs}"
        )

    print("\nAll runs completed.")
    print(f"Outputs folder: {outputs_dir}")
    print(f"Summaries folder: {summaries_dir}")


if __name__ == "__main__":
    main()
