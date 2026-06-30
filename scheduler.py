"""
Batch allocation scheduler for two-rank and three-rank TinyLlama inference.

The scheduler owns scheduler.csv. Each row records which decoder-layer interval
belongs to each rank for a given batch. Rank 0 is the scheduler owner; worker
ranks use the boundaries broadcast by Rank 0.

Two-node example:

    batch,rank0,rank1
    1,"[0,5)","[5,22)"

Three-node example:

    batch,rank0,rank1,rank2
    1,"[0,5)","[5,15)","[15,22)"

If scheduler.csv already contains a row for a batch, that row wins. Otherwise
the scheduler inherits the latest earlier allocation and continues inference
with that split. The default boundaries are only used when there is no earlier
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

    intervals is a tuple of (start, end) pairs indexed by rank. Intervals follow
    Python slicing style: [start, end). For a 3-node 22-layer TinyLlama split at
    5 and 15, intervals are ((0, 5), (5, 15), (15, 22)).
    """

    batch: int
    intervals: tuple

    @property
    def boundaries(self):
        """Return [0, split..., total_layers]."""
        return [self.intervals[0][0]] + [end for _, end in self.intervals]

    @property
    def midpoint(self):
        """Backward-compatible two-node split point."""
        if len(self.intervals) != 2:
            raise ValueError("midpoint is only defined for two-rank allocations")
        return self.intervals[0][1]

    @property
    def rank0_interval(self):
        return self.interval_for_rank(0)

    @property
    def rank1_interval(self):
        return self.interval_for_rank(1)

    @property
    def rank2_interval(self):
        return self.interval_for_rank(2)

    def interval_for_rank(self, rank):
        start, end = self.intervals[rank]
        return format_interval(start, end)


def format_interval(start, end):
    """Return the scheduler.csv interval representation."""
    return f"[{start},{end})"


def parse_interval(value):
    """Parse an interval string like [0,5) or [5,22]."""
    match = _INTERVAL_RE.match((value or "").strip())
    if not match:
        raise ValueError(f"Invalid interval: {value!r}")
    return int(match.group(1)), int(match.group(2))


def boundaries_to_intervals(boundaries):
    """Convert [0, a, b, total] to ((0, a), (a, b), (b, total))."""
    return tuple(
        (int(boundaries[index]), int(boundaries[index + 1]))
        for index in range(len(boundaries) - 1)
    )


class Scheduler:
    """Maintain per-batch layer allocations in scheduler.csv.

    Rank 0 calls get_or_create() to choose a batch allocation, then broadcasts
    allocation.boundaries to the other ranks. Worker ranks do not need a local
    scheduler file; the Rank 0 scheduler file is the single source of truth.
    """

    def __init__(self, allocation_csv, total_layers, default_boundaries, world_size):
        self.path = Path(allocation_csv)
        self.total_layers = int(total_layers)
        self.world_size = int(world_size)
        self.default_boundaries = [int(value) for value in default_boundaries]
        self.fieldnames = ["batch"] + [f"rank{rank}" for rank in range(self.world_size)]
        self.allocations = {}
        # These fields are reserved for future adaptive scheduling. Today the
        # scheduler records the latest per-rank metrics but does not change the
        # layer split automatically.
        self.rank_metrics = {rank: None for rank in range(self.world_size)}
        self.rank0_data = None
        self.rank1_data = None
        self.rank2_data = None
        self.latest_rank_metrics_batch = None

        self._validate_boundaries(self.default_boundaries)
        self._load_existing_file()

    def get_or_create(self, batch):
        """Return the allocation for batch, inheriting the latest split if needed.

        Example: if scheduler.csv only defines batch 1 and batch 20, then batch
        2-19 inherit batch 1, and batch 21+ inherit batch 20.
        """
        batch = int(batch)
        if batch not in self.allocations:
            self.allocations[batch] = self._make_allocation(
                batch,
                self._latest_boundaries_before(batch),
            )
            self.save()
        return self.allocations[batch]

    def record_allocation(self, batch, boundaries):
        """Record boundaries chosen by Rank 0."""
        batch = int(batch)
        allocation = self._make_allocation(batch, boundaries)
        self.allocations[batch] = allocation
        self.save()
        return allocation

    def update_rank_metrics(self, rank, metrics, batch=None):
        """Store the latest metrics reported by one rank.

        The current project logs metrics after every batch. Future scheduling
        policies can read rank0_data/rank1_data/rank2_data or rank_metrics to
        decide whether the next batch should move layers between ranks.
        """
        rank = int(rank)
        if rank < 0 or rank >= self.world_size:
            raise ValueError(f"rank {rank} is outside WORLD_SIZE={self.world_size}")

        snapshot = dict(metrics) if metrics is not None else None
        self.rank_metrics[rank] = snapshot
        if rank == 0:
            self.rank0_data = snapshot
        elif rank == 1:
            self.rank1_data = snapshot
        elif rank == 2:
            self.rank2_data = snapshot

        if batch is not None:
            self.latest_rank_metrics_batch = int(batch)

    def update_rank_metrics_from_records(self, records, batch=None):
        """Store all per-rank metric records received after one batch."""
        for record in records:
            if not isinstance(record, dict) or "rank" not in record:
                continue
            record_batch = batch if batch is not None else record.get("batch")
            self.update_rank_metrics(record["rank"], record, batch=record_batch)

    def reallocate_layer(self, batch=None, rank_metrics=None):
        """Reserved hook for future adaptive layer reallocation.

        For now this method intentionally keeps the latest known allocation.
        It returns the boundaries that would be used for batch without mutating
        scheduler.csv. When the adaptive policy is designed, this is the method
        to extend with rules based on rank0/rank1/rank2 metrics.
        """
        _ = rank_metrics if rank_metrics is not None else self.rank_metrics
        if batch is None:
            if self.allocations:
                batch = max(self.allocations) + 1
            else:
                return list(self.default_boundaries)
        return list(self._latest_boundaries_before(int(batch)))

    def save(self):
        """Write all known allocations to scheduler.csv in batch order."""
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.path, "w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=self.fieldnames)
            writer.writeheader()
            for batch in sorted(self.allocations):
                allocation = self.allocations[batch]
                row = {"batch": allocation.batch}
                for rank in range(self.world_size):
                    row[f"rank{rank}"] = allocation.interval_for_rank(rank)
                writer.writerow(row)

    def _load_existing_file(self):
        """Load scheduler.csv if it already exists."""
        if not self.path.exists():
            return

        with open(self.path, "r", encoding="utf-8-sig", newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                if not row:
                    continue
                batch = int(str(row.get("batch", "")).strip())
                intervals = []
                for rank in range(self.world_size):
                    key = f"rank{rank}"
                    if key not in row or not row[key]:
                        raise ValueError(
                            f"{self.path} is missing column {key!r} for WORLD_SIZE={self.world_size}"
                        )
                    intervals.append(parse_interval(row[key]))
                allocation = Allocation(batch=batch, intervals=tuple(intervals))
                self._validate_allocation(allocation)
                self.allocations[batch] = allocation

    def _make_allocation(self, batch, boundaries):
        self._validate_boundaries(boundaries)
        return Allocation(
            batch=batch,
            intervals=boundaries_to_intervals(boundaries),
        )

    def _latest_boundaries_before(self, batch):
        """Find the most recent allocation before batch."""
        earlier_batches = [known for known in self.allocations if known < batch]
        if not earlier_batches:
            return self.default_boundaries
        latest_batch = max(earlier_batches)
        return self.allocations[latest_batch].boundaries

    def _validate_boundaries(self, boundaries):
        if len(boundaries) != self.world_size + 1:
            raise ValueError(
                f"expected {self.world_size + 1} boundaries for WORLD_SIZE={self.world_size}; "
                f"got {boundaries}"
            )
        if boundaries[0] != 0:
            raise ValueError("first boundary must be 0")
        if boundaries[-1] != self.total_layers:
            raise ValueError(f"last boundary must be total_layers={self.total_layers}")
        for left, right in zip(boundaries, boundaries[1:]):
            if left >= right:
                raise ValueError(f"boundaries must be strictly increasing; got {boundaries}")

    def _validate_allocation(self, allocation):
        """Validate that a CSV row describes one clean pipeline split."""
        self._validate_boundaries(allocation.boundaries)
