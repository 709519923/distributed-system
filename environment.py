"""Network environment model for communication simulation.

Rank 0 owns this environment. In dynamic mode it applies the batch schedule at
the start of every batch, then broadcasts the active environment snapshot to the
other ranks. Scheduler stores the same snapshot after the batch finishes so a
future multi-arm-bandit policy can use communication state as input.
"""

import time


BYTES_PER_MB = 1024 * 1024

# Edit these values on Rank 0 to define the baseline network environment.
# Index convention:
# 0: 0->1, 1: 1->2, 2: 2->0, 3: 0->2, 4: 2->1
DEFAULT_BANDWIDTH = [None, None, None, None, None]
DEFAULT_TIME_COMM_DELAY = [10.0, 30.0, 40.0, 0.0, 0.0]

# Optional schedule. A key means "from this batch onward".
# Example:
DEFAULT_SCHEDULE = {}


class Environment:
    """Five-link network model used by the simulated communication wrappers.

    Link index convention:
    - 0: Rank 0 -> Rank 1
    - 1: Rank 1 -> Rank 2
    - 2: Rank 2 -> Rank 0
    - 3: Rank 0 -> Rank 2
    - 4: Rank 2 -> Rank 1

    Bandwidth uses MB/s. None means unlimited bandwidth on that link.
    time_comm_delay uses milliseconds and is added as fixed one-way delay.
    """

    LINK_ZERO_TO_ONE = 0
    LINK_ONE_TO_TWO = 1
    LINK_TWO_TO_ZERO = 2
    LINK_ZERO_TO_TWO = 3
    LINK_TWO_TO_ONE = 4
    LINK_COUNT = 5

    LINK_NAMES = (
        "0->1",
        "1->2",
        "2->0",
        "0->2",
        "2->1",
    )
    LINK_BY_RANK_PAIR = {
        (0, 1): LINK_ZERO_TO_ONE,
        (1, 2): LINK_ONE_TO_TWO,
        (2, 0): LINK_TWO_TO_ZERO,
        (0, 2): LINK_ZERO_TO_TWO,
        (2, 1): LINK_TWO_TO_ONE,
    }

    def __init__(self, Bandwidth=None, time_comm_delay=None, schedule=None):
        if Bandwidth is None:
            Bandwidth = DEFAULT_BANDWIDTH
        if time_comm_delay is None:
            time_comm_delay = DEFAULT_TIME_COMM_DELAY
        if schedule is None:
            schedule = DEFAULT_SCHEDULE
        self.default_Bandwidth = self._normalize_bandwidth(Bandwidth)
        self.default_time_comm_delay = self._normalize_delay(time_comm_delay)
        self.Bandwidth = list(self.default_Bandwidth)
        self.time_comm_delay = list(self.default_time_comm_delay)
        # Optional future hook. Example:
        # {
        #   10: {
        #       "Bandwidth": [100, 80, 120, 60, 70],
        #       "time_comm_delay": [1.0, 1.5, 2.0, 5.0, 4.0],
        #   }
        # }
        self.schedule = dict(schedule or {})
        self.active_batch = None

    @classmethod
    def from_broadcast_values(cls, values):
        """Build an Environment from a 10-float broadcast payload."""
        values = [float(value) for value in values]
        if len(values) != cls.LINK_COUNT * 2:
            raise ValueError(
                f"Environment broadcast expects {cls.LINK_COUNT * 2} values; got {len(values)}"
            )
        bandwidth_values = [
            None if value < 0 else value for value in values[: cls.LINK_COUNT]
        ]
        delay_values = values[cls.LINK_COUNT :]
        return cls(Bandwidth=bandwidth_values, time_comm_delay=delay_values, schedule={})

    def to_broadcast_values(self):
        """Return [bandwidths..., delays...] for Rank 0 -> workers broadcast."""
        bandwidth_values = [
            -1.0 if value is None else float(value) for value in self.Bandwidth
        ]
        delay_values = [float(value) for value in self.time_comm_delay]
        return bandwidth_values + delay_values

    def snapshot(self):
        """Return a plain dict that Scheduler can store after each batch."""
        return {
            "active_batch": self.active_batch,
            "Bandwidth": list(self.Bandwidth),
            "time_comm_delay": list(self.time_comm_delay),
            "link_names": list(self.LINK_NAMES),
        }

    def describe(self):
        """Human-readable one-line description for startup and batch logs."""
        parts = []
        for index, name in enumerate(self.LINK_NAMES):
            bandwidth = self.Bandwidth[index]
            bandwidth_text = "unlimited" if bandwidth is None else f"{bandwidth:g}MB/s"
            delay_text = f"{self.time_comm_delay[index]:g}ms"
            parts.append(f"{name}:bw={bandwidth_text},delay={delay_text}")
        return "; ".join(parts)

    def apply_batch(self, batch):
        """Apply every scheduled environment change whose key is <= batch."""
        batch = int(batch)
        bandwidth = list(self.default_Bandwidth)
        delay = list(self.default_time_comm_delay)
        for raw_schedule_batch in sorted(self.schedule, key=lambda value: int(value)):
            schedule_batch = int(raw_schedule_batch)
            if schedule_batch > batch:
                break
            entry = self.schedule[raw_schedule_batch]
            if "Bandwidth" in entry:
                bandwidth = self._merge_values(
                    bandwidth,
                    self._normalize_bandwidth(entry["Bandwidth"]),
                )
            if "time_comm_delay" in entry:
                delay = self._merge_values(
                    delay,
                    self._normalize_delay(entry["time_comm_delay"]),
                )
        self.Bandwidth = bandwidth
        self.time_comm_delay = delay
        self.active_batch = batch
        return self

    def link_index(self, src_rank, dst_rank):
        """Return the configured link index for a real source/destination pair."""
        key = (int(src_rank), int(dst_rank))
        if key not in self.LINK_BY_RANK_PAIR:
            raise ValueError(f"No simulated Environment link is defined for Rank {key[0]} -> Rank {key[1]}")
        return self.LINK_BY_RANK_PAIR[key]

    def has_effect(self, src_rank, dst_rank):
        """Return True when bandwidth or fixed delay should alter this link."""
        index = self.link_index(src_rank, dst_rank)
        return self.Bandwidth[index] is not None or self.time_comm_delay[index] > 0

    def target_seconds(self, src_rank, dst_rank, payload_bytes):
        """Return target wall time from bandwidth plus fixed delay."""
        index = self.link_index(src_rank, dst_rank)
        bandwidth = self.Bandwidth[index]
        bandwidth_seconds = 0.0
        if bandwidth is not None:
            bandwidth_seconds = float(payload_bytes) / (float(bandwidth) * BYTES_PER_MB)
        delay_seconds = float(self.time_comm_delay[index]) / 1000.0
        return bandwidth_seconds + delay_seconds

    def sleep_after_real_transfer(self, src_rank, dst_rank, payload_bytes, real_start_time):
        """Sleep for the part of target time not already spent in real NCCL."""
        target = self.target_seconds(src_rank, dst_rank, payload_bytes)
        if target <= 0:
            return 0.0
        real_elapsed = time.perf_counter() - real_start_time
        extra_sleep = max(0.0, target - real_elapsed)
        if extra_sleep > 0:
            time.sleep(extra_sleep)
        return extra_sleep

    def sleep_before_small_transfer(self, src_rank, dst_rank, payload_bytes):
        """Sleep before small sends that do not have an explicit done handshake."""
        target = self.target_seconds(src_rank, dst_rank, payload_bytes)
        if target > 0:
            time.sleep(target)
        return target

    @classmethod
    def _normalize_bandwidth(cls, values):
        if values is None:
            return [None] * cls.LINK_COUNT
        if len(values) != cls.LINK_COUNT:
            raise ValueError(f"Bandwidth must contain {cls.LINK_COUNT} values.")
        normalized = []
        for value in values:
            if value is None:
                normalized.append(None)
                continue
            value = float(value)
            if value <= 0:
                raise ValueError("Bandwidth values must be positive or None.")
            normalized.append(value)
        return normalized

    @classmethod
    def _normalize_delay(cls, values):
        if values is None:
            return [0.0] * cls.LINK_COUNT
        if len(values) != cls.LINK_COUNT:
            raise ValueError(f"time_comm_delay must contain {cls.LINK_COUNT} values.")
        normalized = []
        for value in values:
            value = float(value)
            if value < 0:
                raise ValueError("time_comm_delay values must be non-negative.")
            normalized.append(value)
        return normalized

    @staticmethod
    def _merge_values(base_values, override_values):
        """Apply scheduled values. None keeps unlimited bandwidth when present."""
        merged = list(base_values)
        for index, value in enumerate(override_values):
            merged[index] = value
        return merged
