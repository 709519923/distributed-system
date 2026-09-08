import csv
import tempfile
import unittest
from pathlib import Path

from extract_bandit_results import latest_complete_run, summarize


class ExtractBanditResultsTests(unittest.TestCase):
    def write_batch_file(self, path, batches):
        fieldnames = [
            "batch",
            "policy",
            "scenario",
            "pull_completed",
            "cumulative_reward",
            "cost_ms_per_step",
            "scheduler_algorithm_ms",
            "scheduler_audit_ms",
            "inference_wall_ms",
        ]
        with open(path, "w", encoding="utf-8", newline="") as file_handle:
            writer = csv.DictWriter(file_handle, fieldnames=fieldnames)
            writer.writeheader()
            for batch in batches:
                writer.writerow(
                    {
                        "batch": batch,
                        "policy": "ucb",
                        "scenario": "D",
                        "pull_completed": int(batch > 1),
                        "cumulative_reward": 0.5 if batch > 1 else 0.0,
                        "cost_ms_per_step": 100.0,
                        "scheduler_algorithm_ms": 0.25,
                        "scheduler_audit_ms": 0.0,
                        "inference_wall_ms": 1000.0,
                    }
                )

    def write_run_summary(self, path, status, total_batches):
        with open(path, "w", encoding="utf-8", newline="") as file_handle:
            writer = csv.DictWriter(
                file_handle,
                fieldnames=["status", "total_batches"],
            )
            writer.writeheader()
            writer.writerow({"status": status, "total_batches": total_batches})

    def test_latest_complete_run_skips_newer_incomplete_file(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            log_dir = Path(temporary_directory)
            complete = log_dir / "batch_metrics_ucb_D_2026-09-08.csv"
            incomplete = log_dir / "batch_metrics_ucb_D_2026-09-09.csv"
            self.write_batch_file(complete, (1, 2))
            self.write_batch_file(incomplete, (1,))
            self.write_run_summary(
                log_dir / "run_summary_ucb_D_2026-09-08.csv",
                "complete",
                2,
            )
            self.write_run_summary(
                log_dir / "run_summary_ucb_D_2026-09-09.csv",
                "interrupted",
                1,
            )

            selected_path, rows = latest_complete_run(log_dir, "ucb", "D")
            self.assertEqual(selected_path, complete)
            self.assertEqual(len(rows), 2)

            summary = summarize("ucb", selected_path, rows)
            self.assertEqual(summary["total_batches"], 2)
            self.assertEqual(summary["measured_batches"], 1)
            self.assertAlmostEqual(summary["total_scheduler_algorithm_ms"], 0.5)
            self.assertAlmostEqual(summary["algorithm_overhead_percent"], 0.02499375156210947)


if __name__ == "__main__":
    unittest.main()
