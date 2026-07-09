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
import math
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


class LayerBanditPolicy:
    """Small UCB-style policy for choosing 3-rank layer split arms.

    An arm is represented as (p1, p2), which maps to:

        rank0: [0, p1)
        rank1: [p1, p2)
        rank2: [p2, total_layers)

    The policy evaluates one arm for six completed batches. The first batch is
    treated as warmup, and the following five batches are used to update the
    arm reward. This keeps model reload, CUDA warmup, and communication setup
    noise from dominating the online decision.
    """

    def __init__(
        self,
        total_layers,
        default_boundaries,
        world_size,
        window_size=6,
        warmup_skip=1,
        exploration_weight=0.5,
    ):
        self.total_layers = int(total_layers)
        self.default_boundaries = [int(value) for value in default_boundaries]
        self.world_size = int(world_size)
        self.window_size = int(window_size)
        self.warmup_skip = int(warmup_skip)
        self.exploration_weight = float(exploration_weight)
        self.enabled = self.world_size == 3
        self.current_arm = self.boundaries_to_arm(self.default_boundaries)
        self.active_batches = []
        self.total_pulls = 0

        self.arms = self._build_candidate_arms()
        if self.enabled and self.current_arm not in self.arms:
            self.arms.insert(0, self.current_arm)
        self.stats = {arm: self._new_stats() for arm in self.arms}

    @staticmethod
    def _new_stats():
        return {
            "pulls": 0,
            "mean_cost": 0.0,
            "last_cost": None,
            "reward": 0.5,
        }

    def observe_and_select(self, batch, batch_summary_history):
        """Observe one completed batch and return the arm for the next batch."""
        if not self.enabled:
            return None

        batch = int(batch)
        summary = batch_summary_history.get(batch)
        if not summary:
            return self.current_arm

        completed_arm = summary.get("arm")
        if completed_arm is None:
            completed_arm = self.boundaries_to_arm(summary.get("boundaries"))
        if completed_arm is None:
            return self.current_arm
        completed_arm = tuple(int(value) for value in completed_arm)
        self._ensure_arm(completed_arm)

        # If a manual scheduler.csv row changes the split, treat it as the new
        # active arm and start a fresh six-batch measurement window.
        if completed_arm != self.current_arm:
            self.current_arm = completed_arm
            self.active_batches = []

        if batch not in self.active_batches:
            self.active_batches.append(batch)

        if len(self.active_batches) < self.window_size:
            return self.current_arm

        window_batches = self.active_batches[-self.window_size :]
        reward_batches = window_batches[self.warmup_skip :]
        arm_cost = self._mean_window_cost(reward_batches, batch_summary_history)
        if arm_cost is None:
            return self.current_arm
        self._update_arm_cost(self.current_arm, arm_cost)

        next_arm = self._select_next_arm()
        self.current_arm = next_arm
        self.active_batches = []
        return next_arm

    def boundaries_to_arm(self, boundaries):
        """Convert [0, p1, p2, total] to (p1, p2)."""
        if not self.enabled:
            return None
        if boundaries is None:
            return None
        return tuple(int(value) for value in boundaries[1:-1])

    def arm_to_boundaries(self, arm):
        """Convert (p1, p2) to [0, p1, p2, total_layers]."""
        if arm is None:
            return None
        return [0] + [int(value) for value in arm] + [self.total_layers]

    def _build_candidate_arms(self):
        """Generate a small local search space around the default split."""
        if not self.enabled:
            return []

        default_p1, default_p2 = self.current_arm
        offsets = (-4, -2, 0, 2, 4)
        arms = set()
        for p1_offset in offsets:
            for p2_offset in offsets:
                arm = (default_p1 + p1_offset, default_p2 + p2_offset)
                if self._valid_arm(arm):
                    arms.add(arm)

        return sorted(
            arms,
            key=lambda arm: (
                abs(arm[0] - default_p1) + abs(arm[1] - default_p2),
                abs(arm[0] - default_p1),
                abs(arm[1] - default_p2),
                arm[0],
                arm[1],
            ),
        )

    def _valid_arm(self, arm):
        if arm is None or len(arm) != 2:
            return False
        p1, p2 = (int(arm[0]), int(arm[1]))
        return 0 < p1 < p2 < self.total_layers

    def _ensure_arm(self, arm):
        if not self._valid_arm(arm):
            raise ValueError(f"Invalid 3-rank bandit arm: {arm}")
        if arm not in self.stats:
            self.arms.append(arm)
            self.stats[arm] = self._new_stats()

    def _mean_window_cost(self, batches, batch_summary_history):
        """Return the mean bottleneck-rank cost for one evaluation window."""
        costs = []
        for batch in batches:
            summary = batch_summary_history.get(int(batch))
            if not summary:
                continue
            rank_times = summary.get("rank_times", {})
            if len(rank_times) < self.world_size:
                continue
            costs.append(max(float(value) for value in rank_times.values()))
        if not costs:
            return None
        return sum(costs) / len(costs)

    def _update_arm_cost(self, arm, cost):
        stats = self.stats[arm]
        pulls = int(stats["pulls"])
        stats["mean_cost"] = (float(stats["mean_cost"]) * pulls + float(cost)) / (pulls + 1)
        stats["last_cost"] = float(cost)
        stats["pulls"] = pulls + 1
        self.total_pulls += 1
        self._refresh_normalized_rewards()

    def _refresh_normalized_rewards(self):
        observed = [
            float(stats["mean_cost"])
            for stats in self.stats.values()
            if int(stats["pulls"]) > 0
        ]
        if not observed:
            return

        min_cost = min(observed)
        max_cost = max(observed)
        if math.isclose(min_cost, max_cost):
            for stats in self.stats.values():
                if int(stats["pulls"]) > 0:
                    stats["reward"] = 0.5
            return

        denominator = max_cost - min_cost
        for stats in self.stats.values():
            if int(stats["pulls"]) == 0:
                stats["reward"] = 0.5
                continue
            stats["reward"] = 1.0 - ((float(stats["mean_cost"]) - min_cost) / denominator)

    def _select_next_arm(self):
        """Choose the next arm with UCB, testing unseen arms first."""
        for arm in self.arms:
            if int(self.stats[arm]["pulls"]) == 0:
                return arm

        log_total = math.log(max(self.total_pulls, 2))
        best_arm = self.arms[0]
        best_score = None
        for arm in self.arms:
            stats = self.stats[arm]
            pulls = int(stats["pulls"])
            score = float(stats["reward"]) + self.exploration_weight * math.sqrt(log_total / pulls)
            if best_score is None or score > best_score:
                best_arm = arm
                best_score = score
        return best_arm


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
        self.summary_path = self.path.with_name("scheduler_summary.csv")
        self.summary_fieldnames = [
            "batch",
            "prefill_mode",
            "layer_allocation",
            "rank",
            "time_label",
            "time_ms",
        ]
        self.allocations = {}
        # These fields keep online data in memory so adaptive policies do not
        # need to parse scheduler_summary.csv during the running experiment.
        self.rank_metrics = {rank: None for rank in range(self.world_size)}
        self.rank0_data = None
        self.rank1_data = None
        self.rank2_data = None
        self.latest_rank_metrics_batch = None
        self.environment_data = None
        self.environment_history = {}
        self.batch_summary_history = {}

        self._validate_boundaries(self.default_boundaries)
        self._load_existing_file()
        self.bandit = LayerBanditPolicy(
            total_layers=self.total_layers,
            default_boundaries=self.default_boundaries,
            world_size=self.world_size,
        )

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

    def update_environment_data(self, batch, environment_snapshot):
        """Store the active network environment used by a completed batch."""
        batch = int(batch)
        snapshot = dict(environment_snapshot)
        if "Bandwidth" in snapshot:
            snapshot["Bandwidth"] = list(snapshot["Bandwidth"])
        if "time_comm_delay" in snapshot:
            snapshot["time_comm_delay"] = list(snapshot["time_comm_delay"])
        self.environment_data = snapshot
        self.environment_history[batch] = snapshot

    def collect_batch_summary(self, batch, prefill_mode, boundaries, records):
        """Collect the minimal online data needed by future scheduling policy.

        This method intentionally mirrors the active value shown in
        "--- summary after batch ...". It stores only batch number, layer
        allocation, rank, and one mode-dependent timing value per rank.
        """
        batch = int(batch)
        boundaries = [int(value) for value in boundaries]
        layer_allocation = self._allocation_text(boundaries)
        time_label = self._time_label_for_mode(prefill_mode)
        rank_times = {}
        for record in records:
            if not isinstance(record, dict) or "rank" not in record:
                continue
            rank = int(record["rank"])
            rank_times[rank] = self._summary_time_for_record(prefill_mode, record)

        self.batch_summary_history[batch] = {
            "batch": batch,
            "prefill_mode": prefill_mode,
            "layer_allocation": layer_allocation,
            "boundaries": boundaries,
            "arm": tuple(boundaries[1:-1]) if self.world_size == 3 else None,
            "time_label": time_label,
            "rank_times": rank_times,
        }
        self.save_summary_history()

    def run_bandit_after_batch(self, batch):
        """Run the online bandit policy after one batch summary is collected."""
        return self.bandit.observe_and_select(
            batch=batch,
            batch_summary_history=self.batch_summary_history,
        )

    def reallocate_layer(self, batch=None, arm=None, rank_metrics=None):
        """Write the selected next-batch layer allocation to scheduler.csv.

        The bandit policy decides the arm before this method is called. This
        method is deliberately only the allocation writer: it converts the arm
        into boundaries, validates them, records the batch row, and returns the
        boundaries that were written.
        """
        _ = rank_metrics if rank_metrics is not None else self.rank_metrics
        if batch is None:
            if self.allocations:
                batch = max(self.allocations) + 1
            else:
                return list(self.default_boundaries)
        if arm is None:
            return list(self._latest_boundaries_before(int(batch)))

        boundaries = self._boundaries_from_arm(arm)
        self.record_allocation(batch, boundaries)
        return boundaries

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

    def save_summary_history(self):
        """Write minimal online batch summaries to scheduler_summary.csv."""
        self.summary_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.summary_path, "w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=self.summary_fieldnames)
            writer.writeheader()
            for batch in sorted(self.batch_summary_history):
                summary = self.batch_summary_history[batch]
                for rank in sorted(summary["rank_times"]):
                    writer.writerow(
                        {
                            "batch": summary["batch"],
                            "prefill_mode": summary["prefill_mode"],
                            "layer_allocation": summary["layer_allocation"],
                            "rank": rank,
                            "time_label": summary["time_label"],
                            "time_ms": f"{float(summary['rank_times'][rank]):.6f}",
                        }
                    )

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

    def _allocation_text(self, boundaries):
        return " ".join(
            f"rank{rank}={format_interval(boundaries[rank], boundaries[rank + 1])}"
            for rank in range(self.world_size)
        )

    def _boundaries_from_arm(self, arm):
        if self.world_size != 3:
            raise ValueError("multi-arm bandit layer reallocation currently requires WORLD_SIZE=3")
        if arm is None or len(arm) != 2:
            raise ValueError(f"expected arm=(p1, p2), got {arm}")
        boundaries = [0, int(arm[0]), int(arm[1]), self.total_layers]
        self._validate_boundaries(boundaries)
        return boundaries

    @staticmethod
    def _time_label_for_mode(prefill_mode):
        if prefill_mode == "distributed":
            return "T_comp + T_transfer + T_comm"
        if prefill_mode == "cloud-base":
            return "T_decode + T_transfer + T_comm"
        raise ValueError(f"Unknown prefill_mode: {prefill_mode}")

    @staticmethod
    def _summary_time_for_record(prefill_mode, record):
        if prefill_mode == "distributed":
            return (
                float(record["prefill_comp_time_ms"])
                + float(record["prefill_transfer_time_ms"])
                + float(record["decode_comp_time_ms"])
                + float(record["decode_transfer_time_ms"])
            )
        if prefill_mode == "cloud-base":
            return (
                float(record["cloud_prefill_rank2_time_ms"])
                + float(record["kv_cache_send_time_ms"])
                + float(record["kv_cache_recv_time_ms"])
                + float(record["decode_comp_time_ms"])
                + float(record["decode_transfer_time_ms"])
            )
        raise ValueError(f"Unknown prefill_mode: {prefill_mode}")

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
