"""Write one-batch experiment records to a plain text log.

Rank 0 owns the final log file. Worker ranks send compact numeric tensors back
after each batch. Timing fields are intentionally split into compute and
transfer parts so Scheduler can later use the same records for adaptive
decisions without guessing what a mixed total means.
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
    "prefill_param_bytes",
    "kv_cache_bytes_after_prefill",
    "prefill_comp_time_ms",
    "prefill_transfer_time_ms",
    "cloud_prefill_rank2_time_ms",
    "kv_cache_send_time_ms",
    "kv_cache_recv_time_ms",
    "decode_step_count",
    "decode_comp_time_ms",
    "decode_transfer_time_ms",
    "decode_time_per_token_ms",
]


LINK_LABELS = ["0_to_1", "1_to_2", "2_to_0", "0_to_2", "2_to_1"]


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
        "decode_step_count",
    }
    for field in integer_fields:
        record[field] = int(record[field])
    record["dtype"] = dtype_name
    return record


def send_metric_record(record, device, dst=0):
    """Send one worker-rank batch metric record to Rank 0."""
    dist.send(metric_to_tensor(record, device), dst=dst)


def recv_metric_records(world_size, device, dtype_name):
    """Receive worker metric records through the low-frequency metric chain."""
    records = []
    for _ in range(1, world_size):
        tensor = recv_metric_tensor(src=1, device=device)
        records.append(tensor_to_metric(tensor, dtype_name))
    return records


def normalize_record(record, dtype_name):
    """Add display-only fields used by the text writer."""
    normalized = dict(record)
    normalized["dtype"] = dtype_name
    return normalized


def _format_mb(record, byte_field):
    return f"{bytes_to_mb(record[byte_field]):.2f}"


def _format_float(value):
    return f"{float(value):.2f}"


def _records_by_rank(records):
    return {int(record["rank"]): record for record in records}


def _allocation_text(records, boundaries=None):
    if boundaries:
        return " ".join(
            f"rank{rank}=[{int(boundaries[rank])},{int(boundaries[rank + 1])})"
            for rank in range(len(boundaries) - 1)
        )
    return " ".join(
        f"rank{int(record['rank'])}=[{int(record['layer_start'])},{int(record['layer_end'])})"
        for record in sorted(records, key=lambda item: int(item["rank"]))
    )


def _environment_snapshot(environment):
    if environment is None:
        return None
    if hasattr(environment, "snapshot"):
        return environment.snapshot()
    return environment


def _write_environment(f, environment):
    snapshot = _environment_snapshot(environment)
    if not snapshot:
        f.write("environment=none\n")
        return

    bandwidths = list(snapshot.get("Bandwidth", []))
    delays = list(snapshot.get("time_comm_delay", []))
    f.write("environment:\n")
    for index, label in enumerate(LINK_LABELS):
        bandwidth = bandwidths[index] if index < len(bandwidths) else None
        delay = delays[index] if index < len(delays) else 0.0
        bandwidth_text = "unlimited" if bandwidth is None else _format_float(bandwidth)
        f.write(f"bandwidth_{label}_MBps={bandwidth_text}\n")
        f.write(f"time_comm_delay_{label}_ms={_format_float(delay)}\n")


def append_experiment_log(log_path, records):
    """Append per-rank records for one completed batch."""
    log_path = Path(log_path)
    with open(log_path, "a", encoding="utf-8") as f:
        for record in sorted(records, key=lambda item: (item["batch"], item["rank"])):
            f.write("--- record ---\n")
            f.write(f"batch={int(record['batch'])}\n")
            f.write(f"rank={int(record['rank'])}\n")
            f.write(f"world_size={int(record['world_size'])}\n")
            f.write(f"batch_size={int(record['batch_size'])}\n")
            f.write(f"prompt_count={int(record['prompt_count'])}\n")
            f.write(f"input_seq_len_max={int(record['input_seq_len_max'])}\n")
            f.write(f"input_seq_len_avg={float(record['input_seq_len_avg']):.2f}\n")
            f.write(f"dtype={record['dtype']}\n")
            f.write(f"prefill_param_size_mb={_format_mb(record, 'prefill_param_bytes')}\n")
            f.write(
                "kv_cache_size_mb_after_prefill="
                f"{_format_mb(record, 'kv_cache_bytes_after_prefill')}\n"
            )
            f.write("distributed parameter:\n")
            f.write(
                "prefill_comp_time_ms="
                f"{_format_float(record['prefill_comp_time_ms'])}\n"
            )
            f.write(
                "prefill_transfer_time_ms="
                f"{_format_float(record['prefill_transfer_time_ms'])}\n"
            )
            f.write("Cloud-base parameter:\n")
            f.write(
                "cloud_prefill_rank2_time_ms="
                f"{_format_float(record['cloud_prefill_rank2_time_ms'])}\n"
            )
            f.write(f"kv_cache_send_time_ms={_format_float(record['kv_cache_send_time_ms'])}\n")
            f.write(f"kv_cache_recv_time_ms={_format_float(record['kv_cache_recv_time_ms'])}\n")
            f.write("Common parameter:\n")
            f.write(f"decode_step_count={int(record['decode_step_count'])}\n")
            f.write(f"decode_comp_time_ms={_format_float(record['decode_comp_time_ms'])}\n")
            f.write(
                "decode_transfer_time_ms="
                f"{_format_float(record['decode_transfer_time_ms'])}\n"
            )
            f.write(
                "decode_time_per_token_ms="
                f"{_format_float(record['decode_time_per_token_ms'])}\n\n"
            )
        f.flush()
        os.fsync(f.fileno())


def append_summary_log(log_path, records, batch_number, environment=None, boundaries=None, prefill_mode=None):
    """Append a current-batch summary; no values are accumulated across batches."""
    log_path = Path(log_path)
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    records_by_rank = _records_by_rank(records)
    with open(log_path, "a", encoding="utf-8") as f:
        f.write(f"--- summary after batch {int(batch_number)} ---\n")
        f.write(f"timestamp={timestamp}\n")
        if prefill_mode is not None:
            f.write(f"prefill_mode={prefill_mode}\n")
        f.write(f"layer_allocation={_allocation_text(records, boundaries)}\n")
        _write_environment(f, environment)
        for rank in sorted(records_by_rank):
            record = records_by_rank[rank]
            is_distributed_mode = prefill_mode == "distributed"
            is_cloud_base_mode = prefill_mode == "cloud-base"
            distributed_compute = (
                float(record["prefill_comp_time_ms"])
                + float(record["decode_comp_time_ms"])
            )
            distributed_transfer = (
                float(record["prefill_transfer_time_ms"])
                + float(record["decode_transfer_time_ms"])
            )
            distributed_total = distributed_compute + distributed_transfer
            cloud_total = (
                float(record["cloud_prefill_rank2_time_ms"])
                + float(record["kv_cache_send_time_ms"])
                + float(record["kv_cache_recv_time_ms"])
                + float(record["decode_comp_time_ms"])
                + float(record["decode_transfer_time_ms"])
            )
            f.write(f"rank={int(rank)}\n")
            f.write("Distributed:\n")
            if is_cloud_base_mode:
                f.write("not applicable\n")
            else:
                f.write(
                    "Tcompute_plus_Ttransfer_plus_Tcomm_ms="
                    f"{distributed_total:.2f}\n"
                )
                f.write(f"Tcompute_ms={distributed_compute:.2f}\n")
                f.write(f"Ttransfer_plus_Tcomm_ms={distributed_transfer:.2f}\n")
            f.write("Cloud-base:\n")
            if is_distributed_mode:
                f.write("not applicable\n")
            else:
                f.write(
                    "Tdecode_plus_Ttransfer_plus_Tcomm_ms="
                    f"{cloud_total:.2f}\n"
                )
        f.write("\n")
        f.flush()
        os.fsync(f.fileno())
