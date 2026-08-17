"""Ground-truth arm enumeration and result logging.

This module deliberately contains no model or NCCL code. Rank 0 uses it to
expand each logical dataset batch into one physical execution per configured
candidate arm, then rank the measured results after all arms finish.
"""

import csv
import os
from dataclasses import dataclass
from pathlib import Path


GROUND_TRUTH_SCENARIOS = {
    "A": {
        "request_type": "long_input_short_output",
        "input_min": 800,
        "input_max": 1500,
        "target_output_tokens": 90,
    },
    "B": {
        "request_type": "short_input_long_output",
        "input_min": 10,
        "input_max": 150,
        "target_output_tokens": 400,
    },
    "C": {
        "request_type": "medium_input_medium_output",
        "input_min": 200,
        "input_max": 600,
        "target_output_tokens": 256,
    },
}


@dataclass(frozen=True)
class GroundTruthTrial:
    """One physical execution of one arm for one logical dataset batch."""

    dataset_batch: int
    execution_batch: int
    arm_index: int
    execution_order: int
    arm: tuple
    prompt: str
    scenario: str
    request_type: str
    target_output_tokens: int
    source_row: int


def read_ground_truth_rows(csv_path, prompt_column):
    """Read and validate the labeled ground-truth experiment dataset."""
    rows = []
    with open(csv_path, "r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        if not reader.fieldnames:
            raise ValueError("ground_truth requires a CSV header.")

        prompt_key = prompt_column or reader.fieldnames[0]
        required = {
            prompt_key,
            "scenario",
            "request_type",
            "target_output_tokens",
        }
        missing = sorted(required.difference(reader.fieldnames))
        if missing:
            raise ValueError(
                "ground_truth CSV is missing columns: " + ", ".join(missing)
            )

        for row_number, row in enumerate(reader, start=2):
            prompt = (row.get(prompt_key) or "").strip()
            scenario = (row.get("scenario") or "").strip().upper()
            request_type = (row.get("request_type") or "").strip()
            if not prompt:
                raise ValueError(f"ground_truth CSV row {row_number} has an empty prompt.")
            spec = GROUND_TRUTH_SCENARIOS.get(scenario)
            if spec is None:
                raise ValueError(
                    f"ground_truth CSV row {row_number} has invalid scenario={scenario!r}."
                )
            if request_type != spec["request_type"]:
                raise ValueError(
                    f"ground_truth CSV row {row_number}: scenario {scenario} requires "
                    f"request_type={spec['request_type']!r}, got {request_type!r}."
                )
            try:
                target_output_tokens = int(row.get("target_output_tokens", ""))
            except ValueError as exc:
                raise ValueError(
                    f"ground_truth CSV row {row_number} has invalid target_output_tokens."
                ) from exc
            if target_output_tokens != int(spec["target_output_tokens"]):
                raise ValueError(
                    f"ground_truth CSV row {row_number}: scenario {scenario} requires "
                    f"target_output_tokens={spec['target_output_tokens']}, got "
                    f"{target_output_tokens}."
                )

            source_row_text = (row.get("source_row") or "").strip()
            source_row = int(source_row_text) if source_row_text else row_number - 1
            rows.append(
                {
                    "prompt": prompt,
                    "scenario": scenario,
                    "request_type": request_type,
                    "target_output_tokens": target_output_tokens,
                    "source_row": source_row,
                }
            )
    return rows


class GroundTruthExperiment:
    """Enumerate all configured arms and write per-request ground truth."""

    RESULT_FIELDS = [
        "dataset_batch",
        "execution_batch",
        "scenario",
        "request_type",
        "input_token_length",
        "target_output_tokens",
        "decode_step_count",
        "arm_index",
        "execution_order",
        "arm",
        "layer_allocation",
        "rank0_time_ms",
        "rank1_time_ms",
        "rank2_time_ms",
        "bottleneck_time_ms",
        "cost_per_token_ms",
        "measured_score",
        "arm_ranking",
        "is_best_arm",
    ]

    OUTPUT_FIELDS = [
        "dataset_batch",
        "execution_batch",
        "scenario",
        "request_type",
        "target_output_tokens",
        "arm",
        "prompt",
        "generated_text",
        "full_text",
    ]

    def __init__(
        self,
        dataset_rows,
        candidate_arms,
        total_layers,
        world_size,
        log_directory,
        run_timestamp,
    ):
        self.dataset_rows = [dict(row) for row in dataset_rows]
        self.arms = [tuple(int(value) for value in arm) for arm in candidate_arms]
        self.total_layers = int(total_layers)
        self.world_size = int(world_size)
        self.log_path = (
            Path(log_directory)
            / f"arm_details_ground_truth_{run_timestamp}.csv"
        )
        self.pending_results = {}

        if self.world_size != 3:
            raise ValueError("ground_truth currently requires WORLD_SIZE=3.")
        if not self.dataset_rows:
            raise ValueError("ground_truth dataset is empty.")
        if not self.arms:
            raise ValueError("ground_truth requires at least one candidate arm.")
        if len(set(self.arms)) != len(self.arms):
            raise ValueError("CANDIDATE_ARMS contains duplicate arms.")
        for arm in self.arms:
            if len(arm) != 2 or not (0 < arm[0] < arm[1] < self.total_layers):
                raise ValueError(
                    f"Invalid ground_truth arm={arm} for total_layers={self.total_layers}."
                )

    @property
    def arm_count(self):
        """Return the live arm-list length; no experiment constant is used."""
        return len(self.arms)

    @property
    def logical_batch_count(self):
        return len(self.dataset_rows)

    @property
    def execution_batch_count(self):
        return self.logical_batch_count * self.arm_count

    def iter_trials(self):
        """Yield one trial per logical-batch/arm pair.

        Arm execution order rotates by logical batch to reduce fixed ordering
        bias while still evaluating every configured arm exactly once.
        """
        execution_batch = 0
        for dataset_batch, row in enumerate(self.dataset_rows, start=1):
            rotation = (dataset_batch - 1) % self.arm_count
            ordered_arms = self.arms[rotation:] + self.arms[:rotation]
            for execution_order, arm in enumerate(ordered_arms, start=1):
                execution_batch += 1
                yield GroundTruthTrial(
                    dataset_batch=dataset_batch,
                    execution_batch=execution_batch,
                    arm_index=self.arms.index(arm) + 1,
                    execution_order=execution_order,
                    arm=arm,
                    prompt=row["prompt"],
                    scenario=row["scenario"],
                    request_type=row["request_type"],
                    target_output_tokens=int(row["target_output_tokens"]),
                    source_row=int(row["source_row"]),
                )

    def validate_input_length(self, trial, input_token_length):
        """Check the actual inference length against the scenario definition."""
        spec = GROUND_TRUTH_SCENARIOS[trial.scenario]
        input_token_length = int(input_token_length)
        if not int(spec["input_min"]) <= input_token_length <= int(spec["input_max"]):
            raise ValueError(
                f"Dataset batch {trial.dataset_batch} scenario {trial.scenario} has "
                f"input_token_length={input_token_length}; expected "
                f"[{spec['input_min']},{spec['input_max']}]."
            )

    def record_trial(self, trial, records, prefill_mode):
        """Store one measured arm result and flush a ranking after all arms finish."""
        records_by_rank = {int(record["rank"]): record for record in records}
        missing_ranks = sorted(set(range(self.world_size)).difference(records_by_rank))
        if missing_ranks:
            raise ValueError(
                f"Execution batch {trial.execution_batch} is missing metrics for "
                f"rank(s) {missing_ranks}."
            )

        rank_times = {
            rank: self._rank_time_ms(prefill_mode, records_by_rank[rank])
            for rank in range(self.world_size)
        }
        decode_step_count = max(
            int(record.get("decode_step_count", 0)) for record in records_by_rank.values()
        )
        input_token_length = max(
            int(record.get("input_seq_len_max", 0)) for record in records_by_rank.values()
        )
        bottleneck_time_ms = max(rank_times.values())
        cost_per_token_ms = bottleneck_time_ms / max(1, decode_step_count)
        measured_score = 1.0 / (1.0 + cost_per_token_ms / 100.0)
        boundaries = [0, int(trial.arm[0]), int(trial.arm[1]), self.total_layers]
        result = {
            "dataset_batch": trial.dataset_batch,
            "execution_batch": trial.execution_batch,
            "scenario": trial.scenario,
            "request_type": trial.request_type,
            "input_token_length": input_token_length,
            "target_output_tokens": trial.target_output_tokens,
            "decode_step_count": decode_step_count,
            "arm_index": trial.arm_index,
            "execution_order": trial.execution_order,
            "arm": self._format_arm(trial.arm),
            "layer_allocation": self._format_allocation(boundaries),
            "rank0_time_ms": rank_times[0],
            "rank1_time_ms": rank_times[1],
            "rank2_time_ms": rank_times[2],
            "bottleneck_time_ms": bottleneck_time_ms,
            "cost_per_token_ms": cost_per_token_ms,
            "measured_score": measured_score,
        }
        batch_results = self.pending_results.setdefault(trial.dataset_batch, [])
        batch_results.append(result)
        if len(batch_results) > self.arm_count:
            raise RuntimeError(
                f"Dataset batch {trial.dataset_batch} received more than "
                f"{self.arm_count} arm results."
            )
        if len(batch_results) == self.arm_count:
            self._rank_and_append(trial.dataset_batch, batch_results)
            del self.pending_results[trial.dataset_batch]
            return True
        return False

    def enrich_output_rows(self, trial, generated_rows):
        """Attach logical/physical experiment identity to generated text rows."""
        enriched = []
        for row in generated_rows:
            output = {
                "dataset_batch": trial.dataset_batch,
                "execution_batch": trial.execution_batch,
                "scenario": trial.scenario,
                "request_type": trial.request_type,
                "target_output_tokens": trial.target_output_tokens,
                "arm": self._format_arm(trial.arm),
            }
            output.update(row)
            enriched.append(output)
        return enriched

    def write_output_rows(self, output_csv, rows):
        """Write generated text with enough identity to trace every arm run."""
        output_path = Path(output_csv)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=self.OUTPUT_FIELDS)
            writer.writeheader()
            writer.writerows(rows)
        print(f"[Rank 0] Wrote {len(rows)} ground-truth rows to {output_path}")

    def _rank_and_append(self, dataset_batch, batch_results):
        """Rank a complete arm group and append it atomically to the audit CSV."""
        seen_arms = {row["arm"] for row in batch_results}
        expected_arms = {self._format_arm(arm) for arm in self.arms}
        if seen_arms != expected_arms:
            raise RuntimeError(
                f"Dataset batch {dataset_batch} arm set mismatch: "
                f"expected={sorted(expected_arms)}, got={sorted(seen_arms)}."
            )

        ranked = sorted(
            batch_results,
            key=lambda row: (float(row["cost_per_token_ms"]), int(row["arm_index"])),
        )
        for ranking, row in enumerate(ranked, start=1):
            row["arm_ranking"] = ranking
            row["is_best_arm"] = 1 if ranking == 1 else 0

        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        file_exists = self.log_path.exists()
        with open(self.log_path, "a", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=self.RESULT_FIELDS)
            if not file_exists:
                writer.writeheader()
            for row in sorted(batch_results, key=lambda item: int(item["arm_index"])):
                writer.writerow(self._format_result_row(row))
            f.flush()
            os.fsync(f.fileno())

        best = ranked[0]
        print(
            f"[Rank 0] Ground truth dataset batch {dataset_batch}: "
            f"best_arm={best['arm']}; cost_per_token_ms="
            f"{best['cost_per_token_ms']:.6f}; results={self.log_path}"
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
        raise ValueError(f"Unknown prefill_mode={prefill_mode!r}.")

    @staticmethod
    def _format_arm(arm):
        return f"({int(arm[0])},{int(arm[1])})"

    @staticmethod
    def _format_allocation(boundaries):
        return " ".join(
            f"rank{rank}=[{boundaries[rank]},{boundaries[rank + 1]})"
            for rank in range(len(boundaries) - 1)
        )

    @staticmethod
    def _format_result_row(row):
        output = dict(row)
        for field in (
            "rank0_time_ms",
            "rank1_time_ms",
            "rank2_time_ms",
            "bottleneck_time_ms",
            "cost_per_token_ms",
            "measured_score",
        ):
            output[field] = f"{float(output[field]):.6f}"
        return output
