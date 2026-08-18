"""Exhaustive arm runner and durable result writer for Lipschitz validation.

This module deliberately contains no bandit learning. Rank 0 gives every
logical prompt batch to every valid layer-allocation arm, records one raw row
as soon as an arm finishes, then writes a ranked block after all arms for that
logical batch have completed.
"""

import csv
from dataclasses import dataclass
from datetime import datetime
import math
import os
from pathlib import Path
import random


VALIDATION_ORDER_SEED = 20260818


@dataclass(frozen=True)
class ValidationTrial:
    """One physical distributed execution within a logical prompt batch."""

    logical_batch: int
    execution_batch: int
    start_index: int
    prompt_batch: tuple
    arm_index: int
    execution_order: int
    arm: tuple


class LipschitzValidationExperiment:
    """Build exhaustive trials and persist raw/ranked measurements."""

    result_fieldnames = [
        "timestamp",
        "prefill_mode",
        "logical_batch",
        "execution_batch",
        "arm_index",
        "execution_order",
        "p1",
        "p2",
        "layer_allocation",
        "prompt_count",
        "input_seq_len_max",
        "input_seq_len_avg",
        "decode_step_count",
        "rank0_time_ms",
        "rank1_time_ms",
        "rank2_time_ms",
        "bottleneck_rank",
        "bottleneck_time_ms",
        "cost_per_decode_step_ms",
        "measured_score",
        "arm_ranking",
        "is_best_arm",
    ]

    output_fieldnames = [
        "logical_batch",
        "execution_batch",
        "arm_index",
        "execution_order",
        "p1",
        "p2",
        "prompt",
        "generated_text",
        "full_text",
    ]

    def __init__(
        self,
        candidate_arms,
        total_layers,
        world_size,
        output_csv,
        result_directory="bandit_logs",
    ):
        self.total_layers = int(total_layers)
        self.world_size = int(world_size)
        self.candidate_arms = [tuple(int(value) for value in arm) for arm in candidate_arms]
        self._validate_candidate_arms()

        self.arm_to_index = {
            arm: index for index, arm in enumerate(self.candidate_arms, start=1)
        }
        self.execution_order = list(self.candidate_arms)
        random.Random(VALIDATION_ORDER_SEED).shuffle(self.execution_order)
        self.pending_rows = {}

        timestamp = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
        result_directory = Path(result_directory)
        result_directory.mkdir(parents=True, exist_ok=True)
        self.raw_result_path = (
            result_directory / f"lipschitz_validation_raw_{timestamp}.csv"
        )
        self.ranked_result_path = (
            result_directory / f"lipschitz_validation_ranked_{timestamp}.csv"
        )
        self.output_path = Path(output_csv)
        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        self._initialize_csv(self.raw_result_path, self.result_fieldnames)
        self._initialize_csv(self.ranked_result_path, self.result_fieldnames)
        self._initialize_csv(self.output_path, self.output_fieldnames)

    @property
    def arm_count(self):
        return len(self.candidate_arms)

    def iter_trials(self, prompt_batches):
        """Yield every arm for every logical batch with unique execution IDs."""
        execution_batch = 0
        for logical_batch, start_index, prompt_batch in prompt_batches:
            # Rotate the deterministic shuffle so chronological drift does not
            # always affect the same arm at the same position in every batch.
            rotation = (int(logical_batch) - 1) % self.arm_count
            ordered_arms = (
                self.execution_order[rotation:] + self.execution_order[:rotation]
            )
            for execution_order, arm in enumerate(ordered_arms, start=1):
                execution_batch += 1
                yield ValidationTrial(
                    logical_batch=int(logical_batch),
                    execution_batch=execution_batch,
                    start_index=int(start_index),
                    prompt_batch=tuple(prompt_batch),
                    arm_index=self.arm_to_index[arm],
                    execution_order=execution_order,
                    arm=arm,
                )

    def append_generated_rows(self, trial, rows):
        """Persist generated text immediately after one arm completes."""
        output_rows = []
        for row in rows:
            output_rows.append(
                {
                    "logical_batch": trial.logical_batch,
                    "execution_batch": trial.execution_batch,
                    "arm_index": trial.arm_index,
                    "execution_order": trial.execution_order,
                    "p1": trial.arm[0],
                    "p2": trial.arm[1],
                    "prompt": row.get("prompt", ""),
                    "generated_text": row.get("generated_text", ""),
                    "full_text": row.get("full_text", ""),
                }
            )
        self._append_rows(self.output_path, self.output_fieldnames, output_rows)

    def record_trial(self, trial, records, prefill_mode, boundaries):
        """Write one arm immediately and finalize ranking at group completion."""
        rank_times = {
            int(record["rank"]): self._rank_time_ms(prefill_mode, record)
            for record in records
        }
        missing_ranks = sorted(set(range(self.world_size)).difference(rank_times))
        if missing_ranks:
            raise ValueError(
                f"execution_batch={trial.execution_batch} is missing rank records: "
                f"{missing_ranks}"
            )

        bottleneck_rank = max(rank_times, key=rank_times.get)
        bottleneck_time_ms = float(rank_times[bottleneck_rank])
        decode_step_count = max(
            int(record.get("decode_step_count", 0)) for record in records
        )
        cost_per_step_ms = bottleneck_time_ms / max(1, decode_step_count)
        measured_score = 1.0 / (1.0 + cost_per_step_ms / 100.0)
        representative = min(records, key=lambda record: int(record["rank"]))

        row = {
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "prefill_mode": prefill_mode,
            "logical_batch": trial.logical_batch,
            "execution_batch": trial.execution_batch,
            "arm_index": trial.arm_index,
            "execution_order": trial.execution_order,
            "p1": trial.arm[0],
            "p2": trial.arm[1],
            "layer_allocation": self._allocation_text(boundaries),
            "prompt_count": max(int(record.get("prompt_count", 0)) for record in records),
            "input_seq_len_max": max(
                int(record.get("input_seq_len_max", 0)) for record in records
            ),
            "input_seq_len_avg": f"{float(representative.get('input_seq_len_avg', 0.0)):.6f}",
            "decode_step_count": decode_step_count,
            "rank0_time_ms": f"{rank_times[0]:.6f}",
            "rank1_time_ms": f"{rank_times[1]:.6f}",
            "rank2_time_ms": f"{rank_times[2]:.6f}",
            "bottleneck_rank": bottleneck_rank,
            "bottleneck_time_ms": f"{bottleneck_time_ms:.6f}",
            "cost_per_decode_step_ms": f"{cost_per_step_ms:.6f}",
            "measured_score": f"{measured_score:.9f}",
            "arm_ranking": "",
            "is_best_arm": "",
        }
        self._append_rows(self.raw_result_path, self.result_fieldnames, [row])

        pending = self.pending_rows.setdefault(trial.logical_batch, [])
        pending.append(row)
        if len(pending) > self.arm_count:
            raise RuntimeError(
                f"logical_batch={trial.logical_batch} received more than "
                f"{self.arm_count} arm results"
            )
        if len(pending) == self.arm_count:
            ranked_rows = sorted(
                pending,
                key=lambda item: (
                    float(item["cost_per_decode_step_ms"]),
                    int(item["arm_index"]),
                ),
            )
            for ranking, ranked_row in enumerate(ranked_rows, start=1):
                ranked_row["arm_ranking"] = ranking
                ranked_row["is_best_arm"] = int(ranking == 1)
            self._append_rows(
                self.ranked_result_path,
                self.result_fieldnames,
                ranked_rows,
            )
            del self.pending_rows[trial.logical_batch]
            return ranked_rows[0]
        return None

    def _validate_candidate_arms(self):
        if self.world_size != 3:
            raise ValueError("lipschitz_validation requires WORLD_SIZE=3")
        expected = [
            (p1, p2)
            for p1 in range(1, self.total_layers)
            for p2 in range(p1 + 1, self.total_layers)
        ]
        expected_count = math.comb(self.total_layers - 1, 2)
        if len(self.candidate_arms) != expected_count:
            raise ValueError(
                "lipschitz_validation candidate count mismatch: "
                f"expected {expected_count}, got {len(self.candidate_arms)}"
            )
        if len(set(self.candidate_arms)) != len(self.candidate_arms):
            raise ValueError("lipschitz_validation candidate arms contain duplicates")
        if set(self.candidate_arms) != set(expected):
            raise ValueError(
                "lipschitz_validation requires every valid arm satisfying "
                f"0 < p1 < p2 < {self.total_layers}"
            )

    @staticmethod
    def _rank_time_ms(prefill_mode, record):
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

    @staticmethod
    def _initialize_csv(path, fieldnames):
        with open(path, "w", encoding="utf-8", newline="") as file_handle:
            writer = csv.DictWriter(file_handle, fieldnames=fieldnames)
            writer.writeheader()
            file_handle.flush()
            os.fsync(file_handle.fileno())

    @staticmethod
    def _append_rows(path, fieldnames, rows):
        if not rows:
            return
        with open(path, "a", encoding="utf-8", newline="") as file_handle:
            writer = csv.DictWriter(file_handle, fieldnames=fieldnames)
            writer.writerows(rows)
            file_handle.flush()
            os.fsync(file_handle.fileno())

    @staticmethod
    def _allocation_text(boundaries):
        return " ".join(
            f"rank{rank}=[{int(boundaries[rank])},{int(boundaries[rank + 1])})"
            for rank in range(len(boundaries) - 1)
        )
