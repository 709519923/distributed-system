"""
Batch allocation scheduler for dynamic two-rank TinyLlama inference.

The scheduler owns allocation.csv. Each row records which decoder-layer interval
belongs to Rank 0 and Rank 1 for a given batch:

    batch,rank0,rank1
    123,"[0,5)","[5,22)"

If allocation.csv already contains a row for a batch, that row wins. Otherwise
the scheduler inherits the latest earlier allocation and continues inference
with that split. The default midpoint is only used when there is no earlier
allocation at all.
"""

import csv
import re
from dataclasses import dataclass
from pathlib import Path


_INTERVAL_RE = re.compile(r"^\[\s*(\d+)\s*,\s*(\d+)\s*[\)\]]$")


@dataclass(frozen=True)
class Allocation:
    """Layer allocation for one batch.

    Intervals use Python slicing style: [start, end). Rank 0 owns the front part
    of the model and Rank 1 owns the tail part. For TinyLlama with 22 layers and
    midpoint 5, Rank 0 owns [0,5), Rank 1 owns [5,22).
    """

    batch: int
    rank0_start: int
    rank0_end: int
    rank1_start: int
    rank1_end: int

    @property
    def midpoint(self):
        """The split layer where Rank 1 starts."""
        return self.rank0_end

    @property
    def rank0_interval(self):
        return format_interval(self.rank0_start, self.rank0_end)

    @property
    def rank1_interval(self):
        return format_interval(self.rank1_start, self.rank1_end)


def format_interval(start, end):
    """Return the allocation.csv interval representation."""
    return f"[{start},{end})"


def parse_interval(value):
    """Parse an interval string like [0,5) or [5,22].

    The scheduler writes [start,end), but accepting a closing ']' makes manual
    edits less fragile.
    """
    match = _INTERVAL_RE.match((value or "").strip())
    if not match:
        raise ValueError(f"Invalid interval: {value!r}")
    return int(match.group(1)), int(match.group(2))


class Scheduler:
    """Maintain per-batch layer allocations in allocation.csv.

    Rank 0 calls get_or_create() to choose a batch allocation, then broadcasts the
    midpoint to Rank 1. Rank 1 calls record_allocation() with the received
    midpoint so both nodes keep a local allocation.csv for debugging.
    """

    fieldnames = ["batch", "rank0", "rank1"]

    def __init__(self, allocation_csv, total_layers, default_midpoint):
        self.path = Path(allocation_csv)
        self.total_layers = int(total_layers)
        self.default_midpoint = int(default_midpoint)
        self.allocations = {}

        self._validate_midpoint(self.default_midpoint)
        self._load_existing_file()

    def get_or_create(self, batch):
        """Return the allocation for batch, inheriting the latest split if needed.

        Example: if allocation.csv only defines batch 1 and batch 20, then batch
        2-19 inherit batch 1, and batch 21+ inherit batch 20. This matches the
        runtime meaning of "latest way to continue inference".
        """
        batch = int(batch)
        if batch not in self.allocations:
            self.allocations[batch] = self._make_allocation(
                batch,
                self._latest_midpoint_before(batch),
            )
            self.save()
        return self.allocations[batch]

    def record_allocation(self, batch, midpoint):
        """Record an externally chosen midpoint, usually broadcast by Rank 0."""
        batch = int(batch)
        midpoint = int(midpoint)
        allocation = self._make_allocation(batch, midpoint)
        self.allocations[batch] = allocation
        self.save()
        return allocation

    def save(self):
        """Write all known allocations to allocation.csv in batch order."""
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.path, "w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=self.fieldnames)
            writer.writeheader()
            for batch in sorted(self.allocations):
                allocation = self.allocations[batch]
                writer.writerow(
                    {
                        "batch": allocation.batch,
                        "rank0": allocation.rank0_interval,
                        "rank1": allocation.rank1_interval,
                    }
                )

    def _load_existing_file(self):
        """Load allocation.csv if it already exists.

        Existing rows allow manual scheduling experiments. For example, you can
        put a different midpoint on batch 10 and the scheduler will use it.
        """
        if not self.path.exists():
            return

        with open(self.path, "r", encoding="utf-8-sig", newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                if not row:
                    continue
                batch = int(str(row.get("batch", "")).strip())
                rank0_start, rank0_end = parse_interval(row.get("rank0", ""))
                rank1_start, rank1_end = parse_interval(row.get("rank1", ""))
                allocation = Allocation(
                    batch=batch,
                    rank0_start=rank0_start,
                    rank0_end=rank0_end,
                    rank1_start=rank1_start,
                    rank1_end=rank1_end,
                )
                self._validate_allocation(allocation)
                self.allocations[batch] = allocation

    def _make_allocation(self, batch, midpoint):
        self._validate_midpoint(midpoint)
        return Allocation(
            batch=batch,
            rank0_start=0,
            rank0_end=midpoint,
            rank1_start=midpoint,
            rank1_end=self.total_layers,
        )

    def _latest_midpoint_before(self, batch):
        """Find the most recent allocation before batch.

        If no previous allocation exists, fall back to the constructor's default
        midpoint so the scheduler can bootstrap an empty allocation.csv.
        """
        earlier_batches = [known for known in self.allocations if known < batch]
        if not earlier_batches:
            return self.default_midpoint
        latest_batch = max(earlier_batches)
        return self.allocations[latest_batch].midpoint

    def _validate_midpoint(self, midpoint):
        if midpoint <= 0 or midpoint >= self.total_layers:
            raise ValueError(
                f"midpoint must be between 1 and {self.total_layers - 1}; got {midpoint}"
            )

    def _validate_allocation(self, allocation):
        """Validate that a CSV row describes one clean two-way split."""
        if allocation.rank0_start != 0:
            raise ValueError("rank0 interval must start at layer 0")
        if allocation.rank0_end != allocation.rank1_start:
            raise ValueError("rank0 end must equal rank1 start")
        if allocation.rank1_end != self.total_layers:
            raise ValueError(f"rank1 interval must end at total_layers={self.total_layers}")
        self._validate_midpoint(allocation.midpoint)
