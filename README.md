# Distributed Inference Benchmark (Refactored)

This project runs distributed-inference benchmarks from `main5.py` using command-line arguments.
It processes the full dataset in mini-batches (`batch_size`) unless `--build_dataset` is enabled.

## Project structure

- `main5.py`: thin entrypoint, orchestrates full run.
- `config.py`: CLI args, argument validation, and device map building.
- `data_utils.py`: prompt loading (`.txt` or `.csv`), synthetic dataset construction, batching, tokenization.
- `model_utils.py`: model/tokenizer loading and latency-hook registration.
- `inference_utils.py`: generation run and output/summary file writing.
- `input.txt`: source prompts.

## How to run

Run with command-line arguments like this (full dataset mode):

```bash
python main5.py \
  --model_id /home/dingcong/models/google/gemma-2-2b-it \
  --input_file input_64.csv \
  --output_file outputs_bs64_sl256.txt \
  --summary_file summary_bs64_sl256.txt \
  --batch_size 64 \
  --sequence_length 256 \
  --num_cpu_layers 3 \
  --mid_gpu_layer 12
```

If you add `--build_dataset`, prompts will be repeated/truncated to exactly `batch_size`, so typically:
- `Total Samples == batch_size`
- `Num Batches == 1`

## Important arguments

- `--batch_size`: mini-batch size used to iterate through the dataset.
- `--sequence_length`: tokenizer truncation max length.
- `--num_cpu_layers`: number of front layers mapped to CPU.
- `--mid_gpu_layer`: split point where second transfer hook is attached.
- `--build_dataset`: if set, repeat prompts from input until exactly `batch_size` samples are built.
- `--output_file`: generated text output file.
- `--summary_file`: single-run key-value summary.

## Input format

- `.txt`: one prompt per line.
- `.csv`: must contain a `prompt` column.

## build_qwen_inference_dataset usage

Use `build_qwen_inference_dataset` to generate a CSV dataset for inference, then pass that CSV to `main5.py`.

Typical flow:

1. Build dataset (example):

```bash
python build_qwen_inference_dataset.py
```

2. Confirm output CSV has a `prompt` column (for example: `input_64.csv`).

3. Run inference with that CSV:

```bash
python main5.py \
  --model_id /home/dingcong/models/google/gemma-2-2b-it \
  --input_file input_64.csv \
  --output_file outputs_bs64_sl256.txt \
  --summary_file summary_bs64_sl256.txt \
  --batch_size 64 \
  --sequence_length 256 \
  --num_cpu_layers 3 \
  --mid_gpu_layer 12
```

Notes:
- Keep `--build_dataset` disabled if you want to process all rows in the CSV.
- Enable `--build_dataset` only when you want exactly one synthetic batch (size = `batch_size`).

## run_layer_grid.py usage

Use this script to sweep `num_cpu_layers` and `mid_gpu_layer` automatically.

Example command:

```bash
python run_layer_grid.py \
  --model_id /home/dingcong/models/google/gemma-2-2b-it \
  --input_file input.txt \
  --batch_size 64 \
  --sequence_length 256 \
  --num_cpu_layers_list 1,3,12 \
  --mid_gpu_layer_list 3,12,20 \
  --enable_build_dataset
```

Behavior:
- Only valid combinations are run: `num_cpu_layers < mid_gpu_layer`.
- With `1,3,12` and `3,12,20`, total valid runs are `6`.
- Progress is printed after each run: done / remaining / total.
- Output folders are timestamped:
  - `outputs_runs_YYYYMMDDHHMM`
  - `summaries_runs_YYYYMMDDHHMM`

## Output files

- `output_file`: includes config + decoded generated samples.
- `summary_file`: includes compact metrics:
  - `batch_size`
  - `sequence_length` (effective after tokenization)
  - `total_samples`
  - `num_batches`
  - `num_cpu_layers`
  - `mid_gpu_layer`
  - `total_time_sec`
