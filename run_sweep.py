import argparse
import csv
import os
import subprocess
import sys


def parse_summary(path):
    data = {}
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if "=" not in line:
                continue
            k, v = line.split("=", 1)
            data[k.strip()] = v.strip()
    return data


def main():
    parser = argparse.ArgumentParser(description="Run parameter sweep for distributed inference benchmark.")
    parser.add_argument("--python", type=str, default=sys.executable)
    parser.add_argument("--main_script", type=str, default="main5.py")
    parser.add_argument("--input_file", type=str, default="input.txt")
    parser.add_argument("--output_dir", type=str, default="sweep_outputs")
    parser.add_argument("--csv_file", type=str, default="sweep_results.csv")
    parser.add_argument("--model_id", type=str, default="/home/dingcong/models/google/gemma-2-2b-it")
    parser.add_argument("--total_layers", type=int, default=26)
    parser.add_argument("--max_new_tokens", type=int, default=125)
    args = parser.parse_args()

    batch_sizes = [1, 32, 64, 128]
    sequence_lengths = [64, 128, 256, 512]
    groups = [
        {"num_cpu_layers": 1, "mid_gpu_layer": 3},
        {"num_cpu_layers": 3, "mid_gpu_layer": 12},
        {"num_cpu_layers": 12, "mid_gpu_layer": 20},
    ]

    os.makedirs(args.output_dir, exist_ok=True)
    rows = []

    for group_idx, group in enumerate(groups, start=1):
        for batch_size in batch_sizes:
            for sequence_length in sequence_lengths:
                run_tag = (
                    f"g{group_idx}_cpu{group['num_cpu_layers']}_mid{group['mid_gpu_layer']}"
                    f"_bs{batch_size}_sl{sequence_length}"
                )
                output_file = os.path.join(args.output_dir, f"{run_tag}.txt")
                summary_file = os.path.join(args.output_dir, f"{run_tag}.summary.txt")

                cmd = [
                    args.python,
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
                    str(batch_size),
                    "--sequence_length",
                    str(sequence_length),
                    "--num_cpu_layers",
                    str(group["num_cpu_layers"]),
                    "--mid_gpu_layer",
                    str(group["mid_gpu_layer"]),
                    "--total_layers",
                    str(args.total_layers),
                    "--max_new_tokens",
                    str(args.max_new_tokens),
                    "--build_dataset",
                ]

                print(f"[RUN] {run_tag}")
                print(" ".join(cmd))
                result = subprocess.run(cmd, capture_output=True, text=True)

                row = {
                    "run_tag": run_tag,
                    "group": group_idx,
                    "num_cpu_layers": group["num_cpu_layers"],
                    "mid_gpu_layer": group["mid_gpu_layer"],
                    "batch_size": batch_size,
                    "sequence_length_target": sequence_length,
                    "return_code": result.returncode,
                }

                if result.returncode == 0 and os.path.exists(summary_file):
                    summary = parse_summary(summary_file)
                    row["sequence_length_effective"] = summary.get("sequence_length", "")
                    row["total_time_sec"] = summary.get("total_time_sec", "")
                else:
                    row["sequence_length_effective"] = ""
                    row["total_time_sec"] = ""
                    err_file = os.path.join(args.output_dir, f"{run_tag}.stderr.txt")
                    with open(err_file, "w", encoding="utf-8") as ef:
                        ef.write(result.stderr or "")
                        ef.write("\n--- STDOUT ---\n")
                        ef.write(result.stdout or "")
                    row["error_file"] = err_file

                rows.append(row)

    fieldnames = [
        "run_tag",
        "group",
        "num_cpu_layers",
        "mid_gpu_layer",
        "batch_size",
        "sequence_length_target",
        "sequence_length_effective",
        "total_time_sec",
        "return_code",
        "error_file",
    ]

    with open(args.csv_file, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)

    print(f"\nSweep completed. Results saved to: {args.csv_file}")


if __name__ == "__main__":
    main()
