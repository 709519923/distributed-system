import time

import torch


def run_inference(model, inputs, max_new_tokens):
    # Run greedy decoding and return generated token ids + elapsed seconds.
    print("\nStarting batch inference...")
    start_time = time.time()
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
        )
    end_time = time.time()
    return outputs, (end_time - start_time)


def write_outputs(
    output_file,
    texts,
    batch_size,
    sequence_length,
    num_cpu_layers,
    mid_gpu_layer,
    total_time,
    total_samples,
    num_batches,
):
    # Save human-readable generation results and run config.
    with open(output_file, "w", encoding="utf-8") as f:
        f.write(f"===== Batch Size: {batch_size} =====\n")
        f.write(f"Sequence Length: {sequence_length}\n")
        f.write(f"Total Samples: {total_samples}\n")
        f.write(f"Num Batches: {num_batches}\n")
        f.write(f"NUM_CPU_LAYERS: {num_cpu_layers}\n")
        f.write(f"MID_GPU_LAYER: {mid_gpu_layer}\n")
        f.write(f"Total Time: {total_time:.6f} s\n\n")
        for i, text in enumerate(texts):
            f.write(f"[Sample {i}]\n{text}\n\n")


def write_summary(
    summary_file,
    batch_size,
    sequence_length,
    num_cpu_layers,
    mid_gpu_layer,
    total_time,
    total_samples,
    num_batches,
):
    # Save compact machine-friendly run summary.
    with open(summary_file, "w", encoding="utf-8") as f:
        f.write(f"batch_size={batch_size}\n")
        f.write(f"sequence_length={sequence_length}\n")
        f.write(f"total_samples={total_samples}\n")
        f.write(f"num_batches={num_batches}\n")
        f.write(f"num_cpu_layers={num_cpu_layers}\n")
        f.write(f"mid_gpu_layer={mid_gpu_layer}\n")
        f.write(f"total_time_sec={total_time:.6f}\n")
