"""Offline cold-start estimator for distributed TinyLlama experiment logs.

The inference program already records rank-local compute times. This analyzer
does not change the online inference path. It reads one text log, joins
per-batch summary metadata with per-rank record blocks, and estimates cold
start time from the first two observations of the same rank-local partition.
"""

import argparse
import csv
import re
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path


SUMMARY_RE = re.compile(r"^--- summary after batch\s+(\d+)\s+---$")
INTERVAL_RE = re.compile(r"rank(\d+)=\[(\d+),(\d+)\)")
LOG_TIMESTAMP_RE = re.compile(r"log_(\d{4}-\d{2}-\d{2}-\d{2}-\d{2})")


@dataclass
class SummaryInfo:
    """Metadata shared by all rank records in one batch."""

    batch: int
    prefill_mode: str
    rank_layers: dict


@dataclass
class RecordInfo:
    """One parsed --- record --- block."""

    batch: int
    rank: int
    batch_size: int
    input_seq_len_max: int
    prefill_comp_time_ms: float
    decode_step_count: int
    decode_comp_time_ms: float


class ColdStartAnalyzer:
    """Estimate cold-start time from the first two matching log records."""

    output_fields = [
        "prefill_mode",
        "rank",
        "layer_start",
        "layer_end",
        "batch_size",
        "input_seq_len_max",
        "first_batch",
        "second_batch",
        "cold_start_type",
        "first_observed_ms",
        "second_observed_ms",
        "cold_start_ms",
    ]

    def __init__(self, log_path):
        self.log_path = Path(log_path)
        self.summaries = {}
        self.records = []

    def run(self):
        """Parse the log, compute cold starts, write CSV, and return its path."""
        self.parse()
        rows = self.compute()
        output_path = self.default_output_path()
        self.write_csv(output_path, rows)
        return output_path

    def parse(self):
        """Parse summary and record blocks from the log file."""
        if not self.log_path.exists():
            raise FileNotFoundError(f"Log file not found: {self.log_path}")

        lines = self.log_path.read_text(encoding="utf-8").splitlines()
        index = 0
        while index < len(lines):
            line = lines[index].strip()
            summary_match = SUMMARY_RE.match(line)
            if summary_match:
                index = self._parse_summary(lines, index, int(summary_match.group(1)))
                continue
            if line == "--- record ---":
                index = self._parse_record(lines, index)
                continue
            index += 1

    def compute(self):
        """Compute one cold-start row per matching partition key."""
        groups = {}
        for record in self.records:
            summary = self.summaries.get(record.batch)
            if summary is None:
                continue
            layer = summary.rank_layers.get(record.rank)
            if layer is None:
                continue

            layer_start, layer_end = layer
            key = (
                summary.prefill_mode,
                record.rank,
                layer_start,
                layer_end,
                record.batch_size,
                record.input_seq_len_max,
            )
            groups.setdefault(key, []).append(record)

        rows = []
        for key in sorted(groups):
            samples = sorted(groups[key], key=lambda item: item.batch)
            if len(samples) < 2:
                continue
            first, second = samples[0], samples[1]
            row = self._compute_pair(key, first, second)
            if row is not None:
                rows.append(row)
        return rows

    def default_output_path(self):
        """Use the input log timestamp when possible; otherwise use now."""
        match = LOG_TIMESTAMP_RE.search(self.log_path.stem)
        timestamp = match.group(1) if match else datetime.now().strftime("%Y-%m-%d-%H-%M")
        return self.log_path.parent / f"cold_start_time_log_{timestamp}.csv"

    def write_csv(self, output_path, rows):
        """Write estimated cold-start rows to CSV."""
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=self.output_fields)
            writer.writeheader()
            for row in rows:
                writer.writerow(row)

    def _parse_summary(self, lines, start_index, batch):
        block = {}
        index = start_index + 1
        while index < len(lines):
            line = lines[index].strip()
            if line.startswith("--- summary after batch") or line == "--- record ---":
                break
            if "=" in line:
                key, value = line.split("=", 1)
                block[key.strip()] = value.strip()
            index += 1

        prefill_mode = block.get("prefill_mode", "")
        rank_layers = self._parse_layer_allocation(block.get("layer_allocation", ""))
        if prefill_mode and rank_layers:
            self.summaries[batch] = SummaryInfo(
                batch=batch,
                prefill_mode=prefill_mode,
                rank_layers=rank_layers,
            )
        return index

    def _parse_record(self, lines, start_index):
        block = {}
        index = start_index + 1
        while index < len(lines):
            line = lines[index].strip()
            if line.startswith("--- summary after batch") or line == "--- record ---":
                break
            if "=" in line:
                key, value = line.split("=", 1)
                block[key.strip()] = value.strip()
            index += 1

        required = [
            "batch",
            "rank",
            "batch_size",
            "input_seq_len_max",
            "prefill_comp_time_ms",
            "decode_step_count",
            "decode_comp_time_ms",
        ]
        if all(field in block for field in required):
            self.records.append(
                RecordInfo(
                    batch=int(block["batch"]),
                    rank=int(block["rank"]),
                    batch_size=int(block["batch_size"]),
                    input_seq_len_max=int(block["input_seq_len_max"]),
                    prefill_comp_time_ms=float(block["prefill_comp_time_ms"]),
                    decode_step_count=int(block["decode_step_count"]),
                    decode_comp_time_ms=float(block["decode_comp_time_ms"]),
                )
            )
        return index

    @staticmethod
    def _parse_layer_allocation(value):
        rank_layers = {}
        for match in INTERVAL_RE.finditer(value):
            rank_layers[int(match.group(1))] = (int(match.group(2)), int(match.group(3)))
        return rank_layers

    def _compute_pair(self, key, first, second):
        prefill_mode, rank, layer_start, layer_end, batch_size, input_seq_len_max = key
        if prefill_mode == "distributed":
            first_observed = first.prefill_comp_time_ms
            second_observed = second.prefill_comp_time_ms
            cold_start_type = "prefill"
            cold_start_ms = max(0.0, first_observed - second_observed)
        elif prefill_mode == "cloud-base":
            if first.decode_step_count <= 0 or second.decode_step_count <= 0:
                return None
            first_observed = first.decode_comp_time_ms
            second_observed = second.decode_comp_time_ms
            cold_start_type = "decode"
            stable_decode_ms_per_step = second_observed / second.decode_step_count
            expected_first_decode_ms = stable_decode_ms_per_step * first.decode_step_count
            cold_start_ms = max(0.0, first_observed - expected_first_decode_ms)
        else:
            return None

        return {
            "prefill_mode": prefill_mode,
            "rank": rank,
            "layer_start": layer_start,
            "layer_end": layer_end,
            "batch_size": batch_size,
            "input_seq_len_max": input_seq_len_max,
            "first_batch": first.batch,
            "second_batch": second.batch,
            "cold_start_type": cold_start_type,
            "first_observed_ms": f"{first_observed:.6f}",
            "second_observed_ms": f"{second_observed:.6f}",
            "cold_start_ms": f"{cold_start_ms:.6f}",
        }


def parse_args():
    parser = argparse.ArgumentParser(
        description="Estimate cold-start time from a distributed inference text log."
    )
    parser.add_argument("log_path", help="Path to logs/log_YYYY-MM-DD-HH-MM.txt")
    parser.add_argument(
        "--output-csv",
        default=None,
        help="Optional output CSV path. Defaults to logs/cold_start_time_log_<timestamp>.csv",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    analyzer = ColdStartAnalyzer(args.log_path)
    analyzer.parse()
    rows = analyzer.compute()
    output_path = Path(args.output_csv) if args.output_csv else analyzer.default_output_path()
    analyzer.write_csv(output_path, rows)
    print(f"Wrote {len(rows)} cold-start rows to {output_path}")


if __name__ == "__main__":
    main()
