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
    layer_allocation: str
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
        "layer_allocation",
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
        """Compute cold starts only for first-seen consecutive allocations.

        A cold-start estimate is valid only when a full layer allocation appears
        for the first time and the immediately following batch uses the exact
        same allocation. This avoids comparing non-consecutive samples that may
        have gone through a different model partition in between.
        """
        records_by_batch_rank = {
            (record.batch, record.rank): record
            for record in self.records
        }
        seen_allocations = set()
        rows = []

        for batch in sorted(self.summaries):
            summary = self.summaries[batch]
            allocation_key = (summary.prefill_mode, summary.layer_allocation)
            if allocation_key in seen_allocations:
                continue
            seen_allocations.add(allocation_key)

            next_summary = self.summaries.get(batch + 1)
            if next_summary is None:
                continue
            next_key = (next_summary.prefill_mode, next_summary.layer_allocation)
            if next_key != allocation_key:
                continue

            for rank in sorted(summary.rank_layers):
                first = records_by_batch_rank.get((batch, rank))
                second = records_by_batch_rank.get((batch + 1, rank))
                if first is None or second is None:
                    continue
                layer = summary.rank_layers.get(rank)
                next_layer = next_summary.rank_layers.get(rank)
                if layer is None or next_layer is None or layer != next_layer:
                    continue
                row = self._compute_pair(summary, rank, layer, first, second)
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
        layer_allocation = self._normalize_layer_allocation(block.get("layer_allocation", ""))
        rank_layers = self._parse_layer_allocation(layer_allocation)
        if prefill_mode and layer_allocation and rank_layers:
            self.summaries[batch] = SummaryInfo(
                batch=batch,
                prefill_mode=prefill_mode,
                layer_allocation=layer_allocation,
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

    @staticmethod
    def _normalize_layer_allocation(value):
        intervals = []
        for match in INTERVAL_RE.finditer(value):
            rank = int(match.group(1))
            start = int(match.group(2))
            end = int(match.group(3))
            intervals.append((rank, start, end))
        if not intervals:
            return ""
        return " ".join(
            f"rank{rank}=[{start},{end})"
            for rank, start, end in sorted(intervals)
        )

    def _compute_pair(self, summary, rank, layer, first, second):
        prefill_mode = summary.prefill_mode
        layer_start, layer_end = layer
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
            "layer_allocation": summary.layer_allocation,
            "rank": rank,
            "layer_start": layer_start,
            "layer_end": layer_end,
            "batch_size": first.batch_size,
            "input_seq_len_max": first.input_seq_len_max,
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
