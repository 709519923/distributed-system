import os

from config import parse_args, validate_args, build_device_map
from data_utils import load_prompts, build_dataset_prompts, build_tokenized_inputs, chunk_prompts
from inference_utils import run_inference, write_outputs, write_summary
from model_utils import load_model_and_tokenizer, register_latency_hooks

os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"


def main():
    # 1) Parse runtime arguments and build device placement.
    args = parse_args()
    validate_args(args)
    device_map = build_device_map(args)

    # 2) Load model/tokenizer and attach transfer-latency simulation hooks.
    model, tokenizer = load_model_and_tokenizer(args.model_id, device_map)
    register_latency_hooks(model, args)

    # 3) Build inference prompts (optional repeat-to-batch synthetic dataset).
    base_prompts = load_prompts(args.input_file)
    if len(base_prompts) == 0:
        raise ValueError("Input file is empty!")

    prompts = build_dataset_prompts(base_prompts, args.batch_size) if args.build_dataset else base_prompts

    first_layer_device = device_map["model.layers.0"]
    all_texts = []
    total_time = 0.0
    max_effective_seq_len = 0
    batches = list(chunk_prompts(prompts, args.batch_size))
    num_batches = len(batches)
    total_samples = len(prompts)

    # 4) Run generation for all batches in the dataset.
    for batch_idx, batch_prompts in enumerate(batches, start=1):
        print(f"\n[BATCH] {batch_idx}/{num_batches} size={len(batch_prompts)}")
        inputs = build_tokenized_inputs(tokenizer, batch_prompts, args.sequence_length)
        effective_seq_len = inputs["input_ids"].shape[1]
        if effective_seq_len > max_effective_seq_len:
            max_effective_seq_len = effective_seq_len

        inputs = {k: v.to(first_layer_device) for k, v in inputs.items()}
        outputs, batch_time = run_inference(model, inputs, args.max_new_tokens)
        total_time += batch_time
        texts = tokenizer.batch_decode(outputs, skip_special_tokens=True)
        all_texts.extend(texts)

    # 5) Persist full output + compact summary.
    write_outputs(
        output_file=args.output_file,
        texts=all_texts,
        batch_size=args.batch_size,
        sequence_length=max_effective_seq_len,
        num_cpu_layers=args.num_cpu_layers,
        mid_gpu_layer=args.mid_gpu_layer,
        total_time=total_time,
        total_samples=total_samples,
        num_batches=num_batches,
    )
    write_summary(
        summary_file=args.summary_file,
        batch_size=args.batch_size,
        sequence_length=max_effective_seq_len,
        num_cpu_layers=args.num_cpu_layers,
        mid_gpu_layer=args.mid_gpu_layer,
        total_time=total_time,
        total_samples=total_samples,
        num_batches=num_batches,
    )

    print("\n" + "=" * 60)
    print(f"Configured Batch Size: {args.batch_size}")
    print(f"Max Effective Sequence Length: {max_effective_seq_len}")
    print(f"Total Samples: {total_samples}")
    print(f"Num Batches: {num_batches}")
    print(f"NUM_CPU_LAYERS: {args.num_cpu_layers}")
    print(f"MID_GPU_LAYER: {args.mid_gpu_layer}")
    print(f"Total Inference Time: {total_time:.2f} s")
    print(f"Results saved to: {args.output_file}")
    print(f"Summary saved to: {args.summary_file}")
    print("=" * 60)
    print("Done.")


if __name__ == "__main__":
    main()
