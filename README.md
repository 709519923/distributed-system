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
