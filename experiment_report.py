"""Write KV-cache prefill experiment records to a plain text log.

Rank 0 owns the final file. Worker ranks send compact numeric tensors back along
the pipeline path after each batch's prefill pass. The output is intentionally
one key=value per line so it is easy to inspect over SSH without opening a
spreadsheet.
"""

from datetime import datetime
import os
from pathlib import Path

import torch
import torch.distributed as dist

from kv_cache_utils import bytes_to_mb


METRIC_FIELDS = [
    "batch",
    "rank",
    "world_size",
    "layer_start",
    "layer_end",
    "layer_count",
    "batch_size",
    "prompt_count",
    "input_seq_len_max",
    "input_seq_len_avg",
    "prefill_param_count",
    "prefill_param_bytes",
    "kv_cache_bytes_after_prefill",
    "hidden_prefill_batch",
    "hidden_prefill_seq_len",
    "hidden_prefill_hidden_size",
    "hidden_prefill_bytes",
    "cuda_memory_allocated_before_prefill",
    "cuda_memory_allocated_after_prefill",
    "cuda_memory_reserved_after_prefill",
    "prefill_time_ms",
    "prefill_time_total_ms",
    "decode_step_count",
    "decode_time_total_ms",
    "decode_time_per_token_ms",
    "inference_compute_total_ms",
]


def make_log_path(directory="logs"):
    """Return logs/log_YYYY-MM-DD-HH-MM.txt and create the log directory."""
    timestamp = datetime.now().strftime("%Y-%m-%d-%H-%M")
    log_dir = Path(directory)
    log_dir.mkdir(parents=True, exist_ok=True)
    return log_dir / f"log_{timestamp}.txt"


def metric_to_tensor(record, device):
    """Convert one numeric metric record to a CUDA float64 tensor for NCCL send."""
    values = [float(record[field]) for field in METRIC_FIELDS]
    return torch.tensor(values, dtype=torch.float64, device=device)


def send_metric_tensor(tensor, dst):
    """Forward an already-built metric tensor to the previous pipeline rank."""
    dist.send(tensor.contiguous(), dst=dst)


def recv_metric_tensor(src, device):
    """Receive one raw metric tensor from the next pipeline rank."""
    tensor = torch.empty(len(METRIC_FIELDS), dtype=torch.float64, device=device)
    dist.recv(tensor, src=src)
    return tensor


def tensor_to_metric(tensor, dtype_name):
    """Convert a received metric tensor back to a Python dict."""
    values = tensor.detach().cpu().tolist()
    record = {field: values[index] for index, field in enumerate(METRIC_FIELDS)}
    integer_fields = {
        "batch",
        "rank",
        "world_size",
        "layer_start",
        "layer_end",
        "layer_count",
        "batch_size",
        "prompt_count",
        "input_seq_len_max",
        "hidden_prefill_batch",
        "hidden_prefill_seq_len",
        "hidden_prefill_hidden_size",
        "decode_step_count",
    }
    for field in integer_fields:
        record[field] = int(record[field])
    record["dtype"] = dtype_name
    record["timestamp"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    return record


def send_metric_record(record, device, dst=0):
    """Send one worker-rank prefill metric record to Rank 0."""
    dist.send(metric_to_tensor(record, device), dst=dst)


def recv_metric_records(world_size, device, dtype_name):
    """Receive worker metric records through the pipeline.

    For WORLD_SIZE=3, Rank 2 sends its metric to Rank 1, and Rank 1 forwards
    both Rank 1 and Rank 2 records to Rank 0. Metrics intentionally keep this
    low-frequency chain path, while per-token results are returned directly from
    the last rank to Rank 0.
    """
    records = []
    for _ in range(1, world_size):
        tensor = recv_metric_tensor(src=1, device=device)
        records.append(tensor_to_metric(tensor, dtype_name))
    return records


def normalize_record(record, dtype_name):
    """Add display-only fields used by the text writer."""
    normalized = dict(record)
    normalized["dtype"] = dtype_name
    normalized["timestamp"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    return normalized


def _format_mb(record, byte_field):
    return f"{bytes_to_mb(record[byte_field]):.2f}"


def append_experiment_log(log_path, records):
    """Append metric records to the text log.

    Each record is written as a block. Inside a block, each line contains exactly
    one parameter, which keeps the file easy to read and grep.
    """
    log_path = Path(log_path)
    with open(log_path, "a", encoding="utf-8") as f:
        for record in sorted(records, key=lambda item: (item["batch"], item["rank"])):
            f.write("--- record ---\n")
            f.write(f"timestamp={record['timestamp']}\n")
            f.write(f"batch={int(record['batch'])}\n")
            f.write(f"rank={int(record['rank'])}\n")
            f.write(f"world_size={int(record['world_size'])}\n")
            f.write(f"layer_start={int(record['layer_start'])}\n")
            f.write(f"layer_end={int(record['layer_end'])}\n")
            f.write(f"layer_count={int(record['layer_count'])}\n")
            f.write(f"batch_size={int(record['batch_size'])}\n")
            f.write(f"prompt_count={int(record['prompt_count'])}\n")
            f.write(f"input_seq_len_max={int(record['input_seq_len_max'])}\n")
            f.write(f"input_seq_len_avg={float(record['input_seq_len_avg']):.2f}\n")
            f.write(f"dtype={record['dtype']}\n")
            f.write(f"prefill_param_count={int(record['prefill_param_count'])}\n")
            f.write(f"prefill_param_size_mb={_format_mb(record, 'prefill_param_bytes')}\n")
            f.write(
                "kv_cache_size_mb_after_prefill="
                f"{_format_mb(record, 'kv_cache_bytes_after_prefill')}\n"
            )
            f.write(
                "hidden_prefill_shape="
                f"[{int(record['hidden_prefill_batch'])},"
                f"{int(record['hidden_prefill_seq_len'])},"
                f"{int(record['hidden_prefill_hidden_size'])}]\n"
            )
            f.write(f"hidden_prefill_size_mb={_format_mb(record, 'hidden_prefill_bytes')}\n")
            f.write(
                "cuda_memory_allocated_before_prefill_mb="
                f"{_format_mb(record, 'cuda_memory_allocated_before_prefill')}\n"
            )
            f.write(
                "cuda_memory_allocated_after_prefill_mb="
                f"{_format_mb(record, 'cuda_memory_allocated_after_prefill')}\n"
            )
            f.write(
                "cuda_memory_reserved_after_prefill_mb="
                f"{_format_mb(record, 'cuda_memory_reserved_after_prefill')}\n"
            )
            f.write(f"prefill_time_ms={float(record['prefill_time_ms']):.2f}\n")
            f.write(f"prefill_time_total_ms={float(record['prefill_time_total_ms']):.2f}\n")
            f.write(f"decode_step_count={int(record['decode_step_count'])}\n")
            f.write(f"decode_time_total_ms={float(record['decode_time_total_ms']):.2f}\n")
            f.write(
                "decode_time_per_token_ms="
                f"{float(record['decode_time_per_token_ms']):.2f}\n"
            )
            f.write(
                "inference_compute_total_ms="
                f"{float(record['inference_compute_total_ms']):.2f}\n\n"
            )
        f.flush()
        os.fsync(f.fileno())


def create_summary(world_size):
    """Create Rank 0's per-rank cumulative timing summary."""
    return {
        rank: {
            "record_count": 0,
            "total_prefill_time_ms": 0.0,
            "total_decode_time_ms": 0.0,
            "total_inference_compute_time_ms": 0.0,
            "total_decode_step_count": 0,
        }
        for rank in range(world_size)
    }


def update_summary(summary_by_rank, records):
    """Accumulate one completed batch of metric records into summary_by_rank."""
    for record in records:
        rank = int(record["rank"])
        summary = summary_by_rank.setdefault(
            rank,
            {
                "record_count": 0,
                "total_prefill_time_ms": 0.0,
                "total_decode_time_ms": 0.0,
                "total_inference_compute_time_ms": 0.0,
                "total_decode_step_count": 0,
            },
        )
        summary["record_count"] += 1
        summary["total_prefill_time_ms"] += float(record["prefill_time_total_ms"])
        summary["total_decode_time_ms"] += float(record["decode_time_total_ms"])
        summary["total_inference_compute_time_ms"] += float(record["inference_compute_total_ms"])
        summary["total_decode_step_count"] += int(record["decode_step_count"])


def append_summary_log(log_path, summary_by_rank, batch_number):
    """Append cumulative per-rank timing totals after a completed batch."""
    log_path = Path(log_path)
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    with open(log_path, "a", encoding="utf-8") as f:
        f.write(f"--- summary after batch {int(batch_number)} ---\n")
        f.write(f"timestamp={timestamp}\n")
        for rank in sorted(summary_by_rank):
            summary = summary_by_rank[rank]
            total_steps = int(summary["total_decode_step_count"])
            avg_decode_ms = (
                summary["total_decode_time_ms"] / total_steps if total_steps else 0.0
            )
            f.write(f"rank={int(rank)}\n")
            f.write(f"summary_record_count={int(summary['record_count'])}\n")
            f.write(
                "total_prefill_time_ms="
                f"{float(summary['total_prefill_time_ms']):.2f}\n"
            )
            f.write(
                "total_decode_time_ms="
                f"{float(summary['total_decode_time_ms']):.2f}\n"
            )
            f.write(
                "total_inference_compute_time_ms="
                f"{float(summary['total_inference_compute_time_ms']):.2f}\n"
            )
            f.write(f"total_decode_step_count={total_steps}\n")
            f.write(f"decode_time_per_token_avg_ms={avg_decode_ms:.2f}\n")
        f.write("\n")
        f.flush()
        os.fsync(f.fileno())
