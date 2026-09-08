"""Validate and combine the latest complete single-scenario bandit runs."""

import argparse
import csv
import math
from pathlib import Path


DEFAULT_POLICIES = (
    "ucb",
    "lipschitz",
    "epsilon_greedy",
    "thompson_sampling",
)


def read_rows(path):
    with open(path, encoding="utf-8-sig", newline="") as file_handle:
        return list(csv.DictReader(file_handle))


def latest_complete_run(log_dir, policy, scenario):
    pattern = f"batch_metrics_{policy}_{scenario}_*.csv"
    batch_prefix = f"batch_metrics_{policy}_{scenario}_"
    rejected = []
    for path in sorted(log_dir.glob(pattern), reverse=True):
        rows = read_rows(path)
        if not rows:
            rejected.append(f"{path.name}: empty batch log")
            continue
        batches = [int(row["batch"]) for row in rows]
        expected = list(range(1, len(rows) + 1))
        if batches != expected:
            rejected.append(f"{path.name}: {len(rows)} non-contiguous rows")
            continue
        if any(row.get("policy") != policy for row in rows):
            rejected.append(f"{path.name}: policy column mismatch")
            continue
        if any(str(row.get("scenario", "")).upper() != scenario for row in rows):
            rejected.append(f"{path.name}: scenario column mismatch")
            continue

        timestamp = path.stem[len(batch_prefix) :]
        summary_path = log_dir / f"run_summary_{policy}_{scenario}_{timestamp}.csv"
        if not summary_path.exists():
            rejected.append(f"{path.name}: matching run summary is missing")
            continue
        summary_rows = read_rows(summary_path)
        if len(summary_rows) != 1:
            rejected.append(f"{summary_path.name}: expected exactly one row")
            continue
        run_summary = summary_rows[0]
        if str(run_summary.get("status", "")).lower() != "complete":
            rejected.append(f"{summary_path.name}: run status is not complete")
            continue
        if int(run_summary.get("total_batches", -1)) != len(rows):
            rejected.append(f"{summary_path.name}: batch count mismatch")
            continue
        return path, rows

    detail = "; ".join(rejected) if rejected else "no matching files"
    raise FileNotFoundError(
        f"No complete run for {policy}/{scenario}: {detail}"
    )


def number(row, field, default=0.0):
    value = row.get(field, "")
    return float(value) if value not in (None, "") else float(default)


def percentile(values, fraction):
    if not values:
        return 0.0
    ordered = sorted(float(value) for value in values)
    index = max(0, math.ceil(len(ordered) * fraction) - 1)
    return ordered[index]


def format_value(value):
    return f"{value:.6f}" if isinstance(value, float) else value


def summarize(policy, source_path, rows):
    measured = [row for row in rows if int(row.get("pull_completed", 0)) == 1]
    algorithm_values = [number(row, "scheduler_algorithm_ms") for row in rows]
    inference_values = [number(row, "inference_wall_ms") for row in measured]
    total_algorithm_ms = sum(algorithm_values)
    total_inference_ms = sum(number(row, "inference_wall_ms") for row in rows)
    total_reward = number(rows[-1], "cumulative_reward") if rows else 0.0
    overhead_denominator = total_algorithm_ms + total_inference_ms
    first_row = rows[0]
    return {
        "policy": policy,
        "source_batch_metrics": str(source_path.resolve()),
        "scenario": first_row.get("scenario", ""),
        "arm_shuffle_seed": first_row.get("arm_shuffle_seed", ""),
        "exploration_weight": first_row.get("exploration_weight", ""),
        "decision_random_seed": first_row.get("decision_random_seed", ""),
        "epsilon": first_row.get("epsilon", ""),
        "thompson_prior_alpha": first_row.get("thompson_prior_alpha", ""),
        "thompson_prior_beta": first_row.get("thompson_prior_beta", ""),
        "time_comm_delay_0_to_1_ms": first_row.get(
            "time_comm_delay_0_to_1_ms", ""
        ),
        "time_comm_delay_1_to_2_ms": first_row.get(
            "time_comm_delay_1_to_2_ms", ""
        ),
        "time_comm_delay_2_to_0_ms": first_row.get(
            "time_comm_delay_2_to_0_ms", ""
        ),
        "total_batches": len(rows),
        "measured_batches": len(measured),
        "total_observed_reward": total_reward,
        "mean_observed_reward": total_reward / len(measured) if measured else 0.0,
        "mean_cost_ms_per_step": (
            sum(number(row, "cost_ms_per_step") for row in measured) / len(measured)
            if measured
            else 0.0
        ),
        "total_scheduler_algorithm_ms": total_algorithm_ms,
        "mean_scheduler_algorithm_ms": (
            total_algorithm_ms / len(algorithm_values) if algorithm_values else 0.0
        ),
        "p50_scheduler_algorithm_ms": percentile(algorithm_values, 0.50),
        "p95_scheduler_algorithm_ms": percentile(algorithm_values, 0.95),
        "max_scheduler_algorithm_ms": max(algorithm_values, default=0.0),
        "total_scheduler_audit_ms": sum(
            number(row, "scheduler_audit_ms") for row in rows
        ),
        "total_inference_wall_ms": total_inference_ms,
        "mean_inference_wall_ms": (
            sum(inference_values) / len(inference_values) if inference_values else 0.0
        ),
        "p50_inference_wall_ms": percentile(inference_values, 0.50),
        "p95_inference_wall_ms": percentile(inference_values, 0.95),
        "algorithm_overhead_percent": (
            total_algorithm_ms / overhead_denominator * 100.0
            if overhead_denominator > 0.0
            else 0.0
        ),
    }


def write_rows(path, fieldnames, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="") as file_handle:
        writer = csv.DictWriter(file_handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: format_value(row.get(field, "")) for field in fieldnames})


def parse_args():
    parser = argparse.ArgumentParser(
        description="Combine the latest complete UCB/Lipschitz/baseline batch logs."
    )
    parser.add_argument("--log-dir", type=Path, default=Path("bandit_logs"))
    parser.add_argument("--scenario", default="D", type=lambda value: value.upper())
    parser.add_argument("--policies", nargs="+", default=list(DEFAULT_POLICIES))
    return parser.parse_args()


def main():
    args = parse_args()

    combined_rows = []
    summary_rows = []
    fieldnames = None
    batch_counts = {}
    for policy in args.policies:
        source_path, rows = latest_complete_run(
            args.log_dir,
            policy,
            args.scenario,
        )
        batch_counts[policy] = len(rows)
        if fieldnames is None:
            fieldnames = ["source_batch_metrics"] + list(rows[0].keys())
        for row in rows:
            combined_rows.append(
                {"source_batch_metrics": str(source_path.resolve()), **row}
            )
        summary_rows.append(summarize(policy, source_path, rows))

    if len(set(batch_counts.values())) != 1:
        counts = ", ".join(
            f"{policy}={count}" for policy, count in batch_counts.items()
        )
        raise ValueError(
            "Policy runs must contain the same number of batches for comparison; "
            f"got {counts}."
        )

    batch_output = args.log_dir / f"comparison_batches_{args.scenario}.csv"
    summary_output = args.log_dir / f"comparison_summary_{args.scenario}.csv"
    write_rows(batch_output, fieldnames, combined_rows)
    write_rows(summary_output, list(summary_rows[0].keys()), summary_rows)
    print(f"Wrote {len(combined_rows)} batch rows to {batch_output}")
    print(f"Wrote {len(summary_rows)} policy rows to {summary_output}")


if __name__ == "__main__":
    main()
