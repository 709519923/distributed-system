import csv
import tempfile
import unittest
from pathlib import Path

from scheduler import (
    ARM_SHUFFLE_SEED,
    EPSILON_GREEDY_EPSILON,
    THOMPSON_PRIOR_ALPHA,
    THOMPSON_PRIOR_BETA,
    EpsilonGreedyBanditPolicy,
    Scheduler,
    ThompsonSamplingBanditPolicy,
    create_bandit_policy,
)


DEFAULT_BOUNDARIES = [0, 5, 15, 22]


def batch_summary(arm, cost_ms_per_step):
    decode_steps = 31
    bottleneck_ms = float(cost_ms_per_step) * decode_steps
    return {
        "arm": tuple(arm),
        "boundaries": [0, int(arm[0]), int(arm[1]), 22],
        "decode_step_count": decode_steps,
        "rank_times": {
            0: bottleneck_ms,
            1: bottleneck_ms - 1.0,
            2: bottleneck_ms - 2.0,
        },
    }


class SchedulerPolicyTests(unittest.TestCase):
    def test_factory_registers_reproducible_new_policies(self):
        epsilon = create_bandit_policy(
            "epsilon_greedy", 22, DEFAULT_BOUNDARIES, 3, arm_shuffle_seed=42
        )
        thompson = create_bandit_policy(
            "thompson_sampling", 22, DEFAULT_BOUNDARIES, 3, arm_shuffle_seed=42
        )

        self.assertIsInstance(epsilon, EpsilonGreedyBanditPolicy)
        self.assertIsInstance(thompson, ThompsonSamplingBanditPolicy)
        self.assertEqual(len(epsilon.arms), 195)
        self.assertEqual(epsilon.arms, thompson.arms)
        self.assertEqual(epsilon.epsilon, EPSILON_GREEDY_EPSILON)
        self.assertEqual(epsilon.decision_random_seed, ARM_SHUFFLE_SEED)
        self.assertEqual(thompson.thompson_prior_alpha, THOMPSON_PRIOR_ALPHA)
        self.assertEqual(thompson.thompson_prior_beta, THOMPSON_PRIOR_BETA)

    def test_epsilon_greedy_observes_each_arm_then_exploits_best(self):
        policy = EpsilonGreedyBanditPolicy(
            total_layers=22,
            default_boundaries=DEFAULT_BOUNDARIES,
            world_size=3,
            epsilon=0.0,
            arm_shuffle_seed=42,
            decision_random_seed=42,
        )
        test_arms = [policy.current_arm] + [
            arm for arm in policy.arms if arm != policy.current_arm
        ][:2]
        policy.arms = test_arms
        policy.stats = {arm: policy.stats[arm] for arm in test_arms}

        history = {1: batch_summary(policy.current_arm, 50.0)}
        policy.update_after_batch(1, history)
        self.assertEqual(policy.total_pulls, 0)

        costs = {test_arms[0]: 30.0, test_arms[1]: 10.0, test_arms[2]: 20.0}
        for batch in range(2, 5):
            used_arm = policy.current_arm
            history[batch] = batch_summary(used_arm, costs[used_arm])
            policy.update_after_batch(batch, history)

        self.assertTrue(all(policy.stats[arm]["pulls"] == 1 for arm in test_arms))
        self.assertEqual(policy.current_arm, test_arms[1])
        self.assertEqual(policy.last_selection_mode, "greedy_exploit")

    def test_thompson_fractional_posterior_update_and_seed(self):
        first = ThompsonSamplingBanditPolicy(
            total_layers=22,
            default_boundaries=DEFAULT_BOUNDARIES,
            world_size=3,
            arm_shuffle_seed=42,
            decision_random_seed=42,
        )
        second = ThompsonSamplingBanditPolicy(
            total_layers=22,
            default_boundaries=DEFAULT_BOUNDARIES,
            world_size=3,
            arm_shuffle_seed=42,
            decision_random_seed=42,
        )
        observed_arm = first.current_arm
        history = {
            1: batch_summary(observed_arm, 100.0),
            2: batch_summary(observed_arm, 100.0),
        }
        first.update_after_batch(1, history)
        first.update_after_batch(2, history)

        stats = first.stats[observed_arm]
        self.assertAlmostEqual(stats["reward"], 0.5)
        self.assertAlmostEqual(stats["posterior_alpha"], 1.5)
        self.assertAlmostEqual(stats["posterior_beta"], 1.5)
        self.assertEqual(len(first.last_posterior_samples), 195)

        second.stats[observed_arm].update(
            pulls=1,
            mean_cost=100.0,
            last_cost=100.0,
            reward=0.5,
            posterior_alpha=1.5,
            posterior_beta=1.5,
        )
        second.total_pulls = 1
        selected = second._select_next_arm()
        self.assertEqual(first.current_arm, selected)
        self.assertEqual(first.last_posterior_samples, second.last_posterior_samples)

    def test_batch_and_run_csv_include_policy_timing_and_parameters(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            scheduler = Scheduler(
                allocation_csv=Path(temporary_directory) / "scheduler.csv",
                total_layers=22,
                default_boundaries=DEFAULT_BOUNDARIES,
                world_size=3,
                bandit_policy="epsilon_greedy",
                experiment_scenario="D",
            )

            for batch in (1, 2):
                selected_arm = scheduler.select_arm_before_batch(batch)
                scheduler.update_environment_data(
                    batch,
                    {
                        "Bandwidth": [None, None, None, None, None],
                        "time_comm_delay": [10.0, 30.0, 40.0, 0.0, 0.0],
                    },
                )
                scheduler.batch_summary_history[batch] = {
                    **batch_summary(selected_arm, 25.0),
                    "context": {
                        "scenario": "D",
                        "input_tokens_max": 1600,
                        "target_output_tokens": 32,
                    },
                }
                scheduler.update_policy_after_batch(batch)
                scheduler.append_batch_metrics(
                    batch=batch,
                    prompt_index=batch,
                    partition_transition_wall_ms=2.0,
                    rank0_layer_switch_ms=1.0,
                    inference_wall_ms=100.0,
                    batch_total_wall_ms=103.0,
                )

            scheduler.write_run_summary("complete")

            with open(scheduler.batch_metrics_path, encoding="utf-8", newline="") as f:
                batch_rows = list(csv.DictReader(f))
            self.assertEqual(len(batch_rows), 2)
            self.assertEqual(batch_rows[0]["policy"], "epsilon_greedy")
            self.assertEqual(batch_rows[0]["epsilon"], "0.050000")
            self.assertEqual(batch_rows[0]["decision_random_seed"], "42")
            self.assertEqual(batch_rows[0]["time_comm_delay_0_to_1_ms"], "10.000000")
            self.assertEqual(batch_rows[0]["time_comm_delay_1_to_2_ms"], "30.000000")
            self.assertEqual(batch_rows[0]["time_comm_delay_2_to_0_ms"], "40.000000")
            self.assertIn("scheduler_algorithm_ms", batch_rows[0])
            self.assertIn("scheduler_audit_ms", batch_rows[0])
            self.assertIn("next_selection_mode", batch_rows[0])

            with open(scheduler.run_summary_path, encoding="utf-8", newline="") as f:
                run_row = next(csv.DictReader(f))
            self.assertEqual(run_row["total_batches"], "2")
            self.assertEqual(run_row["measured_batches"], "1")
            self.assertIn("p95_scheduler_algorithm_ms", run_row)
            self.assertIn("algorithm_overhead_percent", run_row)

    def test_lipschitz_audit_time_is_separate_from_algorithm_time(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            scheduler = Scheduler(
                allocation_csv=Path(temporary_directory) / "scheduler.csv",
                total_layers=22,
                default_boundaries=DEFAULT_BOUNDARIES,
                world_size=3,
                bandit_policy="lipschitz",
                experiment_scenario="D",
            )
            for batch in (1, 2):
                selected_arm = scheduler.select_arm_before_batch(batch)
                scheduler.batch_summary_history[batch] = batch_summary(
                    selected_arm,
                    25.0,
                )
                scheduler.update_policy_after_batch(batch)

            timing = scheduler.algorithm_timing_history[2]
            self.assertGreater(timing["audit_ms"], 0.0)
            self.assertGreaterEqual(timing["update_ms"], 0.0)
            self.assertAlmostEqual(
                timing["algorithm_ms"],
                timing["select_ms"] + timing["update_ms"],
            )


if __name__ == "__main__":
    unittest.main()
