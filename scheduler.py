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
import time
from dataclasses import dataclass
from datetime import datetime
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


REQUEST_TYPE_SPECS = (
    {
        "name": "short_input_long_output",
        "input_min": 10.0,
        "input_max": 150.0,
        "output_estimate": 400.0,
    },
    {
        "name": "medium_input_medium_output",
        "input_min": 200.0,
        "input_max": 600.0,
        "output_estimate": 256.0,
    },
    {
        "name": "long_input_short_output",
        "input_min": 800.0,
        "input_max": 1500.0,
        "output_estimate": 90.0,
    },
)

CONTEXT_INPUT_TOKEN_SCALE = 1500.0
CONTEXT_OUTPUT_TOKEN_SCALE = 512.0
CONTEXT_BATCH_SIZE_SCALE = 128.0
CONTEXT_VECTOR_SIZE = 4
# Number of mandatory observations for every actual arm under each request type.
# Change this value to adjust contextual warmup without hard-coding an arm count.
CONTEXT_WARMUP_PULLS = 1
UCB_EXPLORATION_WEIGHT = 0.01

CONTROLLED_LEARNING_BATCHES = 600
CONTROLLED_TOTAL_BATCHES = 900
CONTROLLED_EXPECTED_ARM_COUNT = 20
CONTROLLED_CONTEXT_WARMUP_PULLS = 10
CONTROLLED_SCENARIO_SPECS = {
    "A": {
        "request_type": "long_input_short_output",
        "input_min": 800,
        "input_max": 1500,
        "output_tokens": 90,
    },
    "B": {
        "request_type": "short_input_long_output",
        "input_min": 10,
        "input_max": 150,
        "output_tokens": 400,
    },
    "C": {
        "request_type": "medium_input_medium_output",
        "input_min": 200,
        "input_max": 600,
        "output_tokens": 256,
    },
    "D": {
        "request_type": "extreme_prefill",
        "input_min": 1500,
        "input_max": 1800,
        "output_tokens": 32,
    },
    "E": {
        "request_type": "extreme_decode",
        "input_min": 10,
        "input_max": 64,
        "output_tokens": 768,
    },
    "F": {
        "request_type": "long_context_decode",
        "input_min": 800,
        "input_max": 1100,
        "output_tokens": 512,
    },
}
CONTROLLED_OUTPUT_BY_REQUEST_TYPE = {
    spec["request_type"]: int(spec["output_tokens"])
    for spec in CONTROLLED_SCENARIO_SPECS.values()
}

# Lipschitz distance uses two online-learned effective slopes. Initializing
# both to 0.5 preserves the old penalty 0.5 * (|dp1| + |dp2|) / total_layers.
LIPSCHITZ_INITIAL_Q1 = 0.5
LIPSCHITZ_INITIAL_Q2 = 0.5
LIPSCHITZ_LEARNING_RATE_UP = 0.05
LIPSCHITZ_LEARNING_RATE_DOWN = 0.005
LIPSCHITZ_MIN_SLOPE = 0.001
LIPSCHITZ_MAX_SLOPE = 10.0
LIPSCHITZ_SAFETY_FACTOR = 1.1
LIPSCHITZ_ELIMINATION_MARGIN = 0.0

# CANDIDATE_ARMS = [
#     (1, 5),
#     (1, 7),
#     (1, 9),
#     (1, 11),
#     (1, 13),
#     (1, 15),
#     (1, 17),
#     (3, 7),
#     (3, 11),
#     (3, 15),
#     (3, 19),
#     (5, 7),
#     (5, 11),
#     (5, 13),
#     (5, 15),
#     (5, 17),
#     (9, 12),
#     (9, 14),
#     (9, 16),
#     (9, 18),
# ]

CANDIDATE_ARMS = [
    # p1 = 1
    (1, 2),
    (1, 3),
    (1, 4),
    (1, 5),
    (1, 6),
    (1, 7),
    (1, 8),
    (1, 9),
    (1, 10),
    (1, 11),
    (1, 12),
    (1, 13),
    (1, 14),
    (1, 15),
    (1, 16),
    (1, 17),
    (1, 18),
    (1, 19),
    (1, 20),
    (1, 21),

    # p1 = 2
    (2, 3),
    (2, 4),
    (2, 5),
    (2, 6),
    (2, 7),
    (2, 8),
    (2, 9),
    (2, 10),
    (2, 11),
    (2, 12),
    (2, 13),
    (2, 14),
    (2, 15),
    (2, 16),
    (2, 17),
    (2, 18),
    (2, 19),
    (2, 20),
    (2, 21),

    # p1 = 3
    (3, 4),
    (3, 5),
    (3, 6),
    (3, 7),
    (3, 8),
    (3, 9),
    (3, 10),
    (3, 11),
    (3, 12),
    (3, 13),
    (3, 14),
    (3, 15),
    (3, 16),
    (3, 17),
    (3, 18),
    (3, 19),
    (3, 20),
    (3, 21),

    # p1 = 4
    (4, 5),
    (4, 6),
    (4, 7),
    (4, 8),
    (4, 9),
    (4, 10),
    (4, 11),
    (4, 12),
    (4, 13),
    (4, 14),
    (4, 15),
    (4, 16),
    (4, 17),
    (4, 18),
    (4, 19),
    (4, 20),
    (4, 21),

    # p1 = 5
    (5, 6),
    (5, 7),
    (5, 8),
    (5, 9),
    (5, 10),
    (5, 11),
    (5, 12),
    (5, 13),
    (5, 14),
    (5, 15),
    (5, 16),
    (5, 17),
    (5, 18),
    (5, 19),
    (5, 20),
    (5, 21),

    # p1 = 6
    (6, 7),
    (6, 8),
    (6, 9),
    (6, 10),
    (6, 11),
    (6, 12),
    (6, 13),
    (6, 14),
    (6, 15),
    (6, 16),
    (6, 17),
    (6, 18),
    (6, 19),
    (6, 20),
    (6, 21),

    # p1 = 7
    (7, 8),
    (7, 9),
    (7, 10),
    (7, 11),
    (7, 12),
    (7, 13),
    (7, 14),
    (7, 15),
    (7, 16),
    (7, 17),
    (7, 18),
    (7, 19),
    (7, 20),
    (7, 21),

    # p1 = 8
    (8, 9),
    (8, 10),
    (8, 11),
    (8, 12),
    (8, 13),
    (8, 14),
    (8, 15),
    (8, 16),
    (8, 17),
    (8, 18),
    (8, 19),
    (8, 20),
    (8, 21),

    # p1 = 9
    (9, 10),
    (9, 11),
    (9, 12),
    (9, 13),
    (9, 14),
    (9, 15),
    (9, 16),
    (9, 17),
    (9, 18),
    (9, 19),
    (9, 20),
    (9, 21),

    # p1 = 10
    (10, 11),
    (10, 12),
    (10, 13),
    (10, 14),
    (10, 15),
    (10, 16),
    (10, 17),
    (10, 18),
    (10, 19),
    (10, 20),
    (10, 21),

    # p1 = 11
    (11, 12),
    (11, 13),
    (11, 14),
    (11, 15),
    (11, 16),
    (11, 17),
    (11, 18),
    (11, 19),
    (11, 20),
    (11, 21),
]

def clamp01(value):
    """Clamp a numeric feature into [0, 1]."""
    return max(0.0, min(float(value), 1.0))


def classify_request_type(input_tokens):
    """Classify a prompt batch by input token length.

    The three ranges match the prepared dataset buckets. Values outside the
    ranges are assigned to the closest range, which keeps the scheduler usable
    when a dataset has slightly noisy token counts.
    """
    input_tokens = float(input_tokens)
    best_spec = None
    best_distance = None
    for spec in REQUEST_TYPE_SPECS:
        input_min = float(spec["input_min"])
        input_max = float(spec["input_max"])
        if input_min <= input_tokens <= input_max:
            return spec
        distance = min(abs(input_tokens - input_min), abs(input_tokens - input_max))
        if best_distance is None or distance < best_distance:
            best_spec = spec
            best_distance = distance
    return best_spec


def build_context_from_lengths(input_tokens, batch_size):
    """Build the contextual-bandit feature vector for one prompt batch."""
    request_spec = classify_request_type(input_tokens)
    output_estimate = float(request_spec["output_estimate"])
    batch_size = int(batch_size)
    input_tokens = float(input_tokens)
    features = [
        1.0,
        clamp01(input_tokens / CONTEXT_INPUT_TOKEN_SCALE),
        clamp01(output_estimate / CONTEXT_OUTPUT_TOKEN_SCALE),
        clamp01(batch_size / CONTEXT_BATCH_SIZE_SCALE),
    ]
    return {
        "request_type": request_spec["name"],
        "input_tokens": input_tokens,
        "estimated_output_tokens": output_estimate,
        "batch_size": batch_size,
        "features": features,
    }


def build_prompt_batch_contexts(prompts, tokenizer, batch_size, max_input_tokens):
    """Precompute context for every Rank 0 prompt batch.

    This is Rank 0 only bookkeeping. It does not change the input CSV format and
    does not add any worker-rank metric fields.
    """
    batch_size = int(batch_size)
    if batch_size <= 0:
        raise ValueError(f"batch_size must be positive, got {batch_size}")

    contexts = {}
    for start in range(0, len(prompts), batch_size):
        prompt_batch = prompts[start : start + batch_size]
        lengths = []
        for prompt in prompt_batch:
            encoded = tokenizer(
                prompt,
                add_special_tokens=False,
                truncation=True,
                max_length=max_input_tokens,
            )
            lengths.append(len(encoded["input_ids"]))
        input_tokens_avg = sum(lengths) / len(lengths) if lengths else 0.0
        input_tokens_max = max(lengths) if lengths else 0
        batch_number = start // batch_size + 1
        context = build_context_from_lengths(
            input_tokens=input_tokens_avg,
            batch_size=len(prompt_batch),
        )
        context["input_tokens_max"] = int(input_tokens_max)
        contexts[batch_number] = context
    return contexts


def build_single_scenario_context(batch, base_context, scenario):
    """Attach one explicit A-F scenario without exposing it to arm selection."""
    batch = int(batch)
    if not base_context:
        raise ValueError(f"Missing prompt context for single-scenario batch {batch}.")

    scenario = str(scenario or "").strip().upper()
    scenario_spec = CONTROLLED_SCENARIO_SPECS.get(scenario)
    if scenario_spec is None:
        raise ValueError(f"Batch {batch} has unsupported scenario={scenario!r}.")

    input_tokens = int(base_context.get("input_tokens_max", 0))
    input_min = int(scenario_spec["input_min"])
    input_max = int(scenario_spec["input_max"])
    if not input_min <= input_tokens <= input_max:
        raise ValueError(
            f"Batch {batch} scenario {scenario} has input_tokens={input_tokens}; "
            f"expected [{input_min}, {input_max}]."
        )

    context = dict(base_context)
    context.update(
        {
            "phase": "single_scenario",
            "label_used": 0,
            "scenario": scenario,
            "request_type": scenario_spec["request_type"],
            "inferred_request_type": scenario_spec["request_type"],
            "estimated_output_tokens": float(scenario_spec["output_tokens"]),
            "target_output_tokens": int(scenario_spec["output_tokens"]),
        }
    )
    return context


def build_contextual_controlled_context(batch, base_context, labeled_row):
    """Attach controlled-learning state without leaking labels into evaluation."""
    batch = int(batch)
    if not base_context:
        raise ValueError(f"Missing prompt context for controlled batch {batch}.")

    context = dict(base_context)
    inferred_request_type = str(context.get("request_type") or "")
    if inferred_request_type not in CONTROLLED_OUTPUT_BY_REQUEST_TYPE:
        raise ValueError(
            f"Batch {batch} inferred unsupported request_type={inferred_request_type!r}."
        )

    estimated_output_tokens = CONTROLLED_OUTPUT_BY_REQUEST_TYPE[inferred_request_type]
    features = list(context.get("features") or [])
    if len(features) != CONTEXT_VECTOR_SIZE:
        raise ValueError(f"Batch {batch} has invalid context features: {features}")
    features[2] = clamp01(estimated_output_tokens / CONTEXT_OUTPUT_TOKEN_SCALE)

    context["features"] = features
    context["estimated_output_tokens"] = float(estimated_output_tokens)
    context["inferred_request_type"] = inferred_request_type

    if batch <= CONTROLLED_LEARNING_BATCHES:
        expected_scenario = ("A", "B", "C")[(batch - 1) % 3]
        scenario = str(labeled_row.get("scenario") or "")
        labeled_request_type = str(labeled_row.get("request_type") or "")
        labeled_phase = str(labeled_row.get("phase") or "").lower()
        if labeled_phase != "warmup":
            raise ValueError(
                f"Batch {batch} must have phase='warmup'; got {labeled_phase!r}."
            )
        if scenario != expected_scenario:
            raise ValueError(
                f"Batch {batch} must follow A/B/C order; expected {expected_scenario}, "
                f"got {scenario!r}."
            )
        scenario_spec = CONTROLLED_SCENARIO_SPECS.get(scenario)
        if scenario_spec is None:
            raise ValueError(f"Batch {batch} has unsupported scenario={scenario!r}.")
        expected_request_type = scenario_spec["request_type"]
        if labeled_request_type != expected_request_type:
            raise ValueError(
                f"Batch {batch} label mismatch: scenario {scenario} requires "
                f"{expected_request_type!r}, got {labeled_request_type!r}."
            )
        if inferred_request_type != expected_request_type:
            raise ValueError(
                f"Batch {batch} prompt-length context inferred {inferred_request_type!r}, "
                f"but the learning label is {expected_request_type!r}."
            )

        context["phase"] = "learning"
        context["label_used"] = 1
        context["scenario"] = scenario
        context["target_output_tokens"] = int(scenario_spec["output_tokens"])
    else:
        # Evaluation deliberately ignores every label column. The phase switch
        # comes only from the fixed batch boundary, and context comes only from
        # tokenized prompt length.
        context["phase"] = "evaluation"
        context["label_used"] = 0
        context["scenario"] = ""
        context["target_output_tokens"] = None

    return context


def identity_matrix(size, scale=1.0):
    """Return scale * I as a list-of-lists matrix."""
    return [
        [float(scale) if row == col else 0.0 for col in range(size)]
        for row in range(size)
    ]


def solve_linear_system(matrix, vector):
    """Solve Ax=b for small dense systems using Gauss-Jordan elimination."""
    size = len(vector)
    augmented = [
        [float(matrix[row][col]) for col in range(size)] + [float(vector[row])]
        for row in range(size)
    ]

    for col in range(size):
        pivot_row = max(range(col, size), key=lambda row: abs(augmented[row][col]))
        pivot = augmented[pivot_row][col]
        if math.isclose(pivot, 0.0, abs_tol=1e-12):
            raise ValueError("linear system is singular")
        if pivot_row != col:
            augmented[col], augmented[pivot_row] = augmented[pivot_row], augmented[col]

        pivot = augmented[col][col]
        for item in range(col, size + 1):
            augmented[col][item] /= pivot

        for row in range(size):
            if row == col:
                continue
            factor = augmented[row][col]
            if math.isclose(factor, 0.0, abs_tol=1e-12):
                continue
            for item in range(col, size + 1):
                augmented[row][item] -= factor * augmented[col][item]

    return [augmented[row][size] for row in range(size)]


def dot(left, right):
    """Return dot(left, right) for short numeric vectors."""
    return sum(float(a) * float(b) for a, b in zip(left, right))


def add_outer_product_in_place(matrix, vector):
    """Update matrix += vector vector^T in place."""
    for row in range(len(vector)):
        for col in range(len(vector)):
            matrix[row][col] += float(vector[row]) * float(vector[col])


class LayerBanditPolicy:
    """Plain UCB policy for choosing 3-rank layer split arms.

    An arm is represented as (p1, p2), which maps to:

        rank0: [0, p1)
        rank1: [p1, p2)
        rank2: [p2, total_layers)

    Only the first completed batch of the whole run is treated as warmup. Every
    later batch updates the selected arm's cumulative mean reward, then selects
    the arm for the next batch with the classic UCB1 score.
    """

    def __init__(
        self,
        total_layers,
        default_boundaries,
        world_size,
        window_size=2,
        warmup_skip=1,
        exploration_weight=UCB_EXPLORATION_WEIGHT,
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
        self.reward_scale_ms = 100.0
        self.global_warmup_complete = False
        self.last_batch_was_warmup = False

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

    def select_arm(self, context=None, batch=None):
        """Select the arm used by the current batch.

        Plain UCB does not use context, but it follows the same before-batch
        interface as contextual policies.
        """
        _ = context
        _ = batch
        if not self.enabled:
            return None
        return self.current_arm

    def update_after_batch(self, batch, batch_summary_history):
        """Skip the first global batch, then update UCB after every batch."""
        if not self.enabled:
            return None

        batch = int(batch)
        summary = batch_summary_history.get(batch)
        if not summary:
            return None

        completed_arm = self._completed_arm_from_summary(summary)
        if completed_arm is None:
            return None
        self._ensure_arm(completed_arm)

        self.current_arm = completed_arm
        if not self.global_warmup_complete:
            self.global_warmup_complete = True
            self.last_batch_was_warmup = True
            return self.current_arm

        self.last_batch_was_warmup = False
        arm_cost = self._cost_per_token(summary)
        if arm_cost is None:
            return self.current_arm
        self._update_ucb_arm_reward(self.current_arm, arm_cost)

        self.current_arm = self._select_next_arm()
        return self.current_arm

    def _update_windowed_after_batch(self, batch, batch_summary_history):
        """Preserve the previous warmup-window update for structured policies."""
        if not self.enabled:
            return None

        batch = int(batch)
        summary = batch_summary_history.get(batch)
        if not summary:
            return None

        completed_arm = self._completed_arm_from_summary(summary)
        if completed_arm is None:
            return None
        self._ensure_arm(completed_arm)

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

        self.current_arm = self._select_next_arm()
        self.active_batches = []
        return self.current_arm

    def observe_and_select(self, batch, batch_summary_history):
        """Backward-compatible wrapper for the previous after-batch API."""
        return self.update_after_batch(batch, batch_summary_history)

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
        """Return the explicit candidate split arms for this experiment."""
        if not self.enabled:
            return []

        arms = []
        for arm in CANDIDATE_ARMS:
            normalized_arm = tuple(int(value) for value in arm)
            if self._valid_arm(normalized_arm):
                arms.append(normalized_arm)
        return arms

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

    def _completed_arm_from_summary(self, summary):
        completed_arm = summary.get("arm")
        if completed_arm is None:
            completed_arm = self.boundaries_to_arm(summary.get("boundaries"))
        if completed_arm is None:
            return None
        return tuple(int(value) for value in completed_arm)

    def _mean_window_cost(self, batches, batch_summary_history):
        """Return mean bottleneck-rank cost per decode step for a window."""
        costs = []
        for batch in batches:
            summary = batch_summary_history.get(int(batch))
            if not summary:
                continue
            cost = self._cost_per_token(summary)
            if cost is None:
                continue
            costs.append(cost)
        if not costs:
            return None
        return sum(costs) / len(costs)

    def _update_arm_cost(self, arm, cost):
        """Update cumulative mean cost and cumulative mean per-batch reward."""
        stats = self.stats[arm]
        pulls = int(stats["pulls"])
        cost = float(cost)
        latest_reward = self._reward_from_cost(cost)
        stats["mean_cost"] = (float(stats["mean_cost"]) * pulls + cost) / (pulls + 1)
        stats["last_cost"] = cost
        stats["reward"] = (
            float(stats["reward"]) * pulls + latest_reward
        ) / (pulls + 1)
        stats["pulls"] = pulls + 1
        self.total_pulls += 1

    def _update_ucb_arm_reward(self, arm, cost):
        """Update plain UCB with the cumulative mean of per-batch rewards."""
        stats = self.stats[arm]
        pulls = int(stats["pulls"])
        cost = float(cost)
        latest_reward = self._reward_from_cost(cost)

        # UCB1 estimates each arm's expected reward with its sample mean. The
        # initial reward=0.5 is ignored naturally when pulls is zero.
        stats["mean_cost"] = (
            float(stats["mean_cost"]) * pulls + cost
        ) / (pulls + 1)
        stats["last_cost"] = cost
        stats["reward"] = (
            float(stats["reward"]) * pulls + latest_reward
        ) / (pulls + 1)
        stats["pulls"] = pulls + 1
        self.total_pulls += 1

    def _cost_per_token(self, summary):
        rank_times = summary.get("rank_times", {})
        if len(rank_times) < self.world_size:
            return None
        bottleneck_ms = max(float(value) for value in rank_times.values())
        decode_steps = max(1, int(summary.get("decode_step_count", 0)))
        return bottleneck_ms / decode_steps

    def _reward_from_cost(self, cost_per_token_ms):
        return 1.0 / (1.0 + (float(cost_per_token_ms) / self.reward_scale_ms))

    def _ucb1_exploration_bonus(self, pulls):
        """Return c * sqrt(2 ln(N) / N_a) using the configured weight c."""
        pulls = int(pulls)
        if pulls <= 0:
            return float("inf")
        log_total = math.log(max(self.total_pulls, 2))
        return self.exploration_weight * math.sqrt(2.0 * log_total / pulls)

    def _select_next_arm(self):
        """Choose the next arm with UCB, testing unseen arms first."""
        for arm in self.arms:
            if int(self.stats[arm]["pulls"]) == 0:
                return arm

        best_arm = self.arms[0]
        best_score = None
        for arm in self.arms:
            stats = self.stats[arm]
            pulls = int(stats["pulls"])
            score = float(stats["reward"]) + self._ucb1_exploration_bonus(pulls)
            if best_score is None or score > best_score:
                best_arm = arm
                best_score = score
        return best_arm

    def arm_score_snapshot(self, selected_arm):
        """Return compact per-arm reward/score rows for logging."""
        selected_arm = tuple(selected_arm) if selected_arm is not None else None
        rows = []
        for arm in self.arms:
            stats = self.stats[arm]
            pulls = int(stats["pulls"])
            reward = float(stats["reward"])
            if self.last_batch_was_warmup:
                reward_text = ""
                score = ""
            elif pulls == 0:
                reward_text = f"{reward:.6f}"
                score = "untried"
            else:
                reward_text = f"{reward:.6f}"
                score_value = reward + self._ucb1_exploration_bonus(pulls)
                score = f"{score_value:.6f}"
            rows.append(
                {
                    "arm": self._format_arm(arm),
                    "reward": reward_text,
                    "score": score,
                    "selected": 1 if arm == selected_arm else 0,
                }
            )
        return rows

    @staticmethod
    def _format_arm(arm):
        return f"({int(arm[0])},{int(arm[1])})"


class ContextualBanditPolicy(LayerBanditPolicy):
    """Linear contextual-bandit policy for request-aware layer allocation.

    Each arm keeps a ridge-regression model. At batch start, the current
    context x_t selects the current arm. After the batch finishes, the observed
    reward updates only the selected arm's model.
    """

    policy_name = "contextual"

    def __init__(
        self,
        total_layers,
        default_boundaries,
        world_size,
        window_size=2,
        warmup_skip=1,
        exploration_weight=0.00,
        ridge_lambda=1.0,
        reward_scale_ms=100.0,
    ):
        self.ridge_lambda = float(ridge_lambda)
        self.reward_scale_ms = float(reward_scale_ms)
        self.context_warmup_pulls = int(CONTEXT_WARMUP_PULLS)
        if self.context_warmup_pulls < 1:
            raise ValueError("CONTEXT_WARMUP_PULLS must be at least 1.")
        self.last_context = None
        self.last_context_key = "unknown"
        self.last_features = [1.0, 0.0, 0.0, 0.0]
        super().__init__(
            total_layers=total_layers,
            default_boundaries=default_boundaries,
            world_size=world_size,
            window_size=window_size,
            warmup_skip=warmup_skip,
            exploration_weight=exploration_weight,
        )
        self.context_arm_pulls = {}

    def _new_stats(self):
        stats = super()._new_stats()
        stats.update(
            {
                "A": identity_matrix(CONTEXT_VECTOR_SIZE, self.ridge_lambda),
                "b": [0.0 for _ in range(CONTEXT_VECTOR_SIZE)],
                "mean_reward": 0.0,
                "last_reward": None,
            }
        )
        return stats

    def select_arm(self, context=None, batch=None):
        """Select the current batch arm with LinUCB score."""
        _ = batch
        if not self.enabled:
            return None

        features = self._features_from_context(context)
        context_key = self._context_key_from_context(context)
        self.last_context = context
        self.last_context_key = context_key
        self.last_features = features

        context_pulls = self._ensure_context_arm_pulls(context_key)
        for arm in self.arms:
            if int(context_pulls.get(arm, 0)) < self.context_warmup_pulls:
                self.current_arm = arm
                return arm

        best_arm = self.arms[0]
        best_score = None
        for arm in self.arms:
            score = self._score_arm(arm, features)
            if best_score is None or score > best_score:
                best_arm = arm
                best_score = score
        self.current_arm = best_arm
        return best_arm

    def update_after_batch(self, batch, batch_summary_history):
        """Update the selected arm's linear model from one completed batch."""
        if not self.enabled:
            return None

        summary = batch_summary_history.get(int(batch))
        if not summary:
            return None
        completed_arm = self._completed_arm_from_summary(summary)
        if completed_arm is None:
            return None
        self._ensure_arm(completed_arm)

        context = summary.get("context") or self.last_context
        context_key = self._context_key_from_context(context)
        features = self._features_from_context(context)
        cost = self._cost_per_token(summary)
        if cost is None:
            return completed_arm
        reward = self._reward_from_cost(cost)

        stats = self.stats[completed_arm]
        add_outer_product_in_place(stats["A"], features)
        for index, value in enumerate(features):
            stats["b"][index] += reward * float(value)

        pulls = int(stats["pulls"])
        stats["mean_cost"] = (float(stats["mean_cost"]) * pulls + cost) / (pulls + 1)
        stats["last_cost"] = cost
        stats["mean_reward"] = (
            float(stats["mean_reward"]) * pulls + reward
        ) / (pulls + 1)
        stats["last_reward"] = reward
        stats["reward"] = stats["mean_reward"]
        stats["pulls"] = pulls + 1
        self.total_pulls += 1
        context_pulls = self._ensure_context_arm_pulls(context_key)
        context_pulls[completed_arm] = int(context_pulls.get(completed_arm, 0)) + 1
        self.last_context_key = context_key
        self.current_arm = completed_arm
        return completed_arm

    def build_context(self, batch, batch_summary_history, environment_history=None):
        """Return stored context for compatibility with earlier policy hooks."""
        _ = environment_history
        summary = batch_summary_history.get(int(batch), {})
        return summary.get("context", {})

    def arm_score_snapshot(self, selected_arm):
        """Return per-arm reward/score rows for the current context."""
        selected_arm = tuple(selected_arm) if selected_arm is not None else None
        features = self.last_features
        rows = []
        for arm in self.arms:
            stats = self.stats[arm]
            score = self._score_arm(arm, features)
            rows.append(
                {
                    "arm": self._format_arm(arm),
                    "reward": f"{float(stats['reward']):.6f}",
                    "score": f"{score:.6f}",
                    "selected": 1 if arm == selected_arm else 0,
                }
            )
        return rows

    def _features_from_context(self, context):
        if not context:
            return [1.0, 0.0, 0.0, 0.0]
        features = context.get("features")
        if features is None:
            features = build_context_from_lengths(
                input_tokens=context.get("input_tokens", 0.0),
                batch_size=context.get("batch_size", 1),
            )["features"]
        features = [float(value) for value in features]
        if len(features) != CONTEXT_VECTOR_SIZE:
            raise ValueError(
                f"context feature size must be {CONTEXT_VECTOR_SIZE}; got {features}"
            )
        return features

    def _context_key_from_context(self, context):
        """Return the request-type key used for context-aware warmup."""
        if context and context.get("request_type"):
            return str(context["request_type"])
        if context:
            inferred = build_context_from_lengths(
                input_tokens=context.get("input_tokens", 0.0),
                batch_size=context.get("batch_size", 1),
            )
            return str(inferred["request_type"])
        return "unknown"

    def _ensure_context_arm_pulls(self, context_key):
        """Create the per-context arm-pull table used by contextual warmup."""
        context_key = str(context_key or "unknown")
        if context_key not in self.context_arm_pulls:
            self.context_arm_pulls[context_key] = {}
        context_pulls = self.context_arm_pulls[context_key]
        for arm in self.arms:
            context_pulls.setdefault(arm, 0)
        return context_pulls

    def _score_arm(self, arm, features):
        stats = self.stats[arm]
        theta = solve_linear_system(stats["A"], stats["b"])
        confidence_direction = solve_linear_system(stats["A"], features)
        prediction = dot(theta, features)
        uncertainty = math.sqrt(max(dot(features, confidence_direction), 0.0))
        return prediction + self.exploration_weight * uncertainty


class ContextualControlledBanditPolicy(ContextualBanditPolicy):
    """Contextual policy with labeled learning and label-free evaluation.

    Batches 1..600 use labels only to choose an exact output-token target. Arm
    selection still uses prompt-derived context. From batch 601 onward, labels
    are absent from the policy context and the learned linear models are frozen.
    """

    policy_name = "contextual_controlled"

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.context_warmup_pulls = int(CONTROLLED_CONTEXT_WARMUP_PULLS)
        self.last_model_updated = False
        self.learning_completion_checked = False
        unique_arms = list(dict.fromkeys(self.arms))
        if len(unique_arms) != len(self.arms):
            raise ValueError(
                "contextual_controlled requires distinct effective arms; "
                f"duplicates were found in {self.arms}."
            )
        if len(self.arms) != CONTROLLED_EXPECTED_ARM_COUNT:
            raise ValueError(
                "contextual_controlled requires exactly "
                f"{CONTROLLED_EXPECTED_ARM_COUNT} effective arms, but scheduler "
                f"constructed {len(self.arms)}: {self.arms}. Ensure the default "
                "split is already included in CANDIDATE_ARMS."
            )

    def select_arm(self, context=None, batch=None):
        """Use controlled exploration for learning and pure LinUCB for evaluation."""
        if not self.enabled:
            return None
        if not context:
            raise ValueError("contextual_controlled requires current-batch context.")

        phase = str(context.get("phase") or "")
        if phase == "learning":
            return super().select_arm(context=context, batch=batch)
        if phase != "evaluation":
            raise ValueError(f"Unknown contextual_controlled phase={phase!r}.")

        self._validate_learning_complete()
        features = self._features_from_context(context)
        self.last_context = context
        self.last_context_key = self._context_key_from_context(context)
        self.last_features = features

        best_arm = self.arms[0]
        best_score = self._score_arm(best_arm, features)
        for arm in self.arms[1:]:
            score = self._score_arm(arm, features)
            if score > best_score:
                best_arm = arm
                best_score = score
        self.current_arm = best_arm
        return best_arm

    def update_after_batch(self, batch, batch_summary_history):
        """Update during learning and freeze all linear models during evaluation."""
        summary = batch_summary_history.get(int(batch), {})
        context = summary.get("context") or self.last_context or {}
        phase = str(context.get("phase") or "")
        if phase == "learning":
            pulls_before = int(self.total_pulls)
            completed_arm = super().update_after_batch(batch, batch_summary_history)
            self.last_model_updated = int(self.total_pulls) > pulls_before
            return completed_arm
        if phase != "evaluation":
            raise ValueError(f"Unknown contextual_controlled phase={phase!r}.")

        completed_arm = self._completed_arm_from_summary(summary)
        if completed_arm is not None:
            self._ensure_arm(completed_arm)
            self.current_arm = completed_arm
        self.last_model_updated = False
        return completed_arm

    def _validate_learning_complete(self):
        """Verify every arm has ten observations under each request type."""
        if self.learning_completion_checked:
            return
        for request_spec in REQUEST_TYPE_SPECS:
            request_type = request_spec["name"]
            context_pulls = self._ensure_context_arm_pulls(request_type)
            incomplete = [
                arm
                for arm in self.arms
                if int(context_pulls.get(arm, 0)) < self.context_warmup_pulls
            ]
            if incomplete:
                raise RuntimeError(
                    "contextual_controlled reached evaluation before learning "
                    f"completed for request_type={request_type!r}; incomplete arms={incomplete}."
                )
        self.learning_completion_checked = True


class LipschitzBanditPolicy(LayerBanditPolicy):
    """Non-contextual Lipschitz policy with online split-point sensitivity.

    q1 and q2 are effective reward slopes for p1 and p2. They are learned from
    observed reward differences, so the policy does not assume that moving the
    two split points has the same performance impact on heterogeneous nodes.
    """

    policy_name = "lipschitz"

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.q1 = float(LIPSCHITZ_INITIAL_Q1)
        self.q2 = float(LIPSCHITZ_INITIAL_Q2)
        self.learning_rate_up = float(LIPSCHITZ_LEARNING_RATE_UP)
        self.learning_rate_down = float(LIPSCHITZ_LEARNING_RATE_DOWN)
        self.min_slope = float(LIPSCHITZ_MIN_SLOPE)
        self.max_slope = float(LIPSCHITZ_MAX_SLOPE)
        self.safety_factor = float(LIPSCHITZ_SAFETY_FACTOR)
        self.elimination_margin = float(LIPSCHITZ_ELIMINATION_MARGIN)
        self.active_arms = list(self.arms)
        self.last_pull_audit = None

    def update_after_batch(self, batch, batch_summary_history):
        """Use the same one-global-warmup, per-batch observation cadence as UCB1."""
        self.last_pull_audit = None
        if not self.enabled:
            return None

        batch = int(batch)
        summary = batch_summary_history.get(batch)
        if not summary:
            return None

        completed_arm = self._completed_arm_from_summary(summary)
        if completed_arm is None:
            return None
        self._ensure_arm(completed_arm)
        self.current_arm = completed_arm

        if not self.global_warmup_complete:
            self.global_warmup_complete = True
            self.last_batch_was_warmup = True
            return self.current_arm

        self.last_batch_was_warmup = False
        arm_cost = self._cost_per_token(summary)
        if arm_cost is None:
            return self.current_arm
        self._update_arm_cost(self.current_arm, arm_cost)
        self.current_arm = self._select_next_arm()
        return self.current_arm

    def _ensure_arm(self, arm):
        """Keep manually introduced valid arms visible to the active set."""
        already_known = arm in self.stats
        super()._ensure_arm(arm)
        if not already_known and hasattr(self, "active_arms"):
            self.active_arms.append(arm)

    def _update_arm_cost(self, arm, cost):
        """Update direct reward, online slopes, confidence bounds, and audit."""
        before_state = self._audit_state()
        q1_before = self.q1
        q2_before = self.q2

        super()._update_arm_cost(arm, cost)
        self._update_effective_slopes(arm)
        self._refresh_active_arms()

        after_state = self._audit_state()
        self.last_pull_audit = {
            "updated_arm": arm,
            "pull_index": self.total_pulls,
            "q1_before": q1_before,
            "q2_before": q2_before,
            "q1_after": self.q1,
            "q2_after": self.q2,
            "before": before_state,
            "after": after_state,
        }

    def _select_next_arm(self):
        """Choose the largest optimistic bound without mandatory arm sweeps."""
        candidates = self.active_arms or self.arms
        if not candidates:
            return self.current_arm

        best_arm = self.current_arm if self.current_arm in candidates else candidates[0]
        best_score = self._confidence_bounds(best_arm)[1]
        for arm in candidates:
            score = self._confidence_bounds(arm)[1]
            if score > best_score:
                best_arm = arm
                best_score = score
        return best_arm

    def arm_score_snapshot(self, selected_arm):
        """Return per-arm before/after changes and current Lipschitz state."""
        selected_arm = tuple(selected_arm) if selected_arm is not None else None
        audit = self.last_pull_audit
        if audit is None:
            current_state = self._audit_state()
            audit = {
                "updated_arm": None,
                "pull_index": self.total_pulls,
                "q1_before": self.q1,
                "q2_before": self.q2,
                "q1_after": self.q1,
                "q2_after": self.q2,
                "before": current_state,
                "after": current_state,
            }

        q1_after = float(audit["q1_after"])
        q2_after = float(audit["q2_after"])
        effective_lipschitz = q1_after + q2_after
        if effective_lipschitz > 0.0:
            w1 = q1_after / effective_lipschitz
            w2 = q2_after / effective_lipschitz
        else:
            w1 = 0.5
            w2 = 0.5
        active_arms_text = "|".join(self._format_arm(arm) for arm in self.active_arms)

        rows = []
        for arm in self.arms:
            before = audit["before"][arm]
            after = audit["after"][arm]
            rows.append(
                {
                    "arm": self._format_arm(arm),
                    "pull_completed": 1 if audit["updated_arm"] is not None else 0,
                    "pull_index": int(audit["pull_index"]),
                    "updated_arm": (
                        self._format_arm(audit["updated_arm"])
                        if audit["updated_arm"] is not None
                        else ""
                    ),
                    "pulls_before": int(before["pulls"]),
                    "pulls": int(after["pulls"]),
                    "pulls_delta": int(after["pulls"] - before["pulls"]),
                    "reward_before": f"{before['reward']:.6f}",
                    "reward": f"{after['reward']:.6f}",
                    "reward_delta": f"{after['reward'] - before['reward']:.6f}",
                    "score_before": f"{before['score']:.6f}",
                    "score": f"{after['score']:.6f}",
                    "score_delta": f"{after['score'] - before['score']:.6f}",
                    "confidence_lower": f"{after['lower']:.6f}",
                    "confidence_upper": f"{after['upper']:.6f}",
                    "confidence_width": f"{after['upper'] - after['lower']:.6f}",
                    "q1": f"{q1_after:.6f}",
                    "q2": f"{q2_after:.6f}",
                    "q1_delta": f"{q1_after - float(audit['q1_before']):.6f}",
                    "q2_delta": f"{q2_after - float(audit['q2_before']):.6f}",
                    "lipschitz_constant": f"{effective_lipschitz:.6f}",
                    "w1": f"{w1:.6f}",
                    "w2": f"{w2:.6f}",
                    "active": 1 if arm in self.active_arms else 0,
                    "active_arms": active_arms_text,
                    "selected": 1 if arm == selected_arm else 0,
                    "next_selected": 1 if arm == self.current_arm else 0,
                }
            )
        return rows

    def _audit_state(self):
        """Capture direct rewards and derived confidence state for every arm."""
        state = {}
        for arm in self.arms:
            lower, upper = self._confidence_bounds(arm)
            state[arm] = {
                "pulls": int(self.stats[arm]["pulls"]),
                "reward": float(self.stats[arm]["reward"]),
                "lower": lower,
                "upper": upper,
                "score": upper,
            }
        return state

    def _update_effective_slopes(self, updated_arm):
        """Learn q1/q2 from reward differences against observed peer arms."""
        updated_reward = float(self.stats[updated_arm]["reward"])
        for other_arm in self.arms:
            if other_arm == updated_arm or int(self.stats[other_arm]["pulls"]) == 0:
                continue
            x1, x2 = self._distance_components(updated_arm, other_arm)
            if x1 + x2 <= 0.0:
                continue

            other_reward = float(self.stats[other_arm]["reward"])
            target = self.safety_factor * abs(updated_reward - other_reward)
            predicted = self.q1 * x1 + self.q2 * x2
            underestimated = max(target - predicted, 0.0)
            overestimated = max(predicted - target, 0.0)

            self.q1 = self._clamp_slope(
                self.q1
                + self.learning_rate_up * underestimated * x1
                - self.learning_rate_down * overestimated * x1
            )
            self.q2 = self._clamp_slope(
                self.q2
                + self.learning_rate_up * underestimated * x2
                - self.learning_rate_down * overestimated * x2
            )

    def _clamp_slope(self, value):
        return max(self.min_slope, min(float(value), self.max_slope))

    def _observed_arms(self):
        return [arm for arm in self.arms if int(self.stats[arm]["pulls"]) > 0]

    def _confidence_radius(self, observed_arm):
        pulls = max(1, int(self.stats[observed_arm]["pulls"]))
        log_total = math.log(max(self.total_pulls, 2))
        return self.exploration_weight * math.sqrt(log_total / pulls)

    def _confidence_bounds(self, arm):
        """Infer a reward interval from all observed arms and weighted distance."""
        observed_arms = self._observed_arms()
        if not observed_arms:
            return 0.0, 1.0

        lower = 0.0
        upper = 1.0
        for observed_arm in observed_arms:
            reward = float(self.stats[observed_arm]["reward"])
            radius = self._confidence_radius(observed_arm)
            penalty = self._lipschitz_penalty(arm, observed_arm)
            lower = max(lower, reward - radius - penalty)
            upper = min(upper, reward + radius + penalty)

        lower = max(0.0, min(lower, 1.0))
        upper = max(0.0, min(upper, 1.0))
        if lower > upper:
            midpoint = max(0.0, min((lower + upper) / 2.0, 1.0))
            return midpoint, midpoint
        return lower, upper

    def _refresh_active_arms(self):
        """Recompute confidence-based candidates; removed arms may reactivate."""
        bounds = {arm: self._confidence_bounds(arm) for arm in self.arms}
        best_lower = max(lower for lower, _ in bounds.values())
        self.active_arms = [
            arm
            for arm in self.arms
            if bounds[arm][1] + self.elimination_margin >= best_lower
        ]
        if not self.active_arms and self.arms:
            self.active_arms = [max(self.arms, key=lambda arm: bounds[arm][1])]

    def _distance_components(self, left_arm, right_arm):
        if left_arm is None or right_arm is None:
            raise ValueError("arms must not be None")
        left = tuple(int(value) for value in left_arm)
        right = tuple(int(value) for value in right_arm)
        if len(left) != len(right):
            raise ValueError(f"arm sizes differ: {left_arm} vs {right_arm}")
        if len(left) != 2:
            raise ValueError(f"Lipschitz policy requires two split points: {left_arm}")
        scale = max(float(self.total_layers), 1.0)
        return abs(left[0] - right[0]) / scale, abs(left[1] - right[1]) / scale

    def _lipschitz_penalty(self, left_arm, right_arm):
        x1, x2 = self._distance_components(left_arm, right_arm)
        return self.q1 * x1 + self.q2 * x2

    def arm_distance(self, left_arm, right_arm):
        """Return normalized weighted distance using the learned w1/w2."""
        x1, x2 = self._distance_components(left_arm, right_arm)
        effective_lipschitz = self.q1 + self.q2
        if effective_lipschitz <= 0.0:
            return 0.5 * x1 + 0.5 * x2
        w1 = self.q1 / effective_lipschitz
        w2 = self.q2 / effective_lipschitz
        return w1 * x1 + w2 * x2


class ContextualLipschitzBanditPolicy(ContextualBanditPolicy):
    """Reserved policy hook combining contextual and Lipschitz ideas.

    It currently reuses ContextualBanditPolicy's LinUCB update and exposes an
    arm-distance hook for later reward smoothing across nearby allocations.
    """

    policy_name = "contextual_lipschitz"

    def build_context(self, batch, batch_summary_history, environment_history=None):
        """Return stored context for compatibility with earlier policy hooks."""
        _ = environment_history
        return super().build_context(batch, batch_summary_history)

    def arm_distance(self, left_arm, right_arm):
        """Return normalized L1 distance between two 3-rank split arms."""
        if left_arm is None or right_arm is None:
            return None
        left = tuple(int(value) for value in left_arm)
        right = tuple(int(value) for value in right_arm)
        if len(left) != len(right):
            raise ValueError(f"arm sizes differ: {left_arm} vs {right_arm}")
        distance = sum(abs(a - b) for a, b in zip(left, right))
        return distance / max(float(self.total_layers), 1.0)


BANDIT_POLICY_CLASSES = {
    "ucb": LayerBanditPolicy,
    "contextual": ContextualBanditPolicy,
    "contextual_controlled": ContextualControlledBanditPolicy,
    "lipschitz": LipschitzBanditPolicy,
    "contextual_lipschitz": ContextualLipschitzBanditPolicy,
}



def normalize_bandit_policy_name(name):
    """Normalize user/internal policy names to registry keys."""
    return (name or "ucb").strip().lower().replace("-", "_")


def create_bandit_policy(policy_name, total_layers, default_boundaries, world_size):
    """Create a bandit policy while keeping UCB as the default behavior."""
    normalized_name = normalize_bandit_policy_name(policy_name)
    policy_class = BANDIT_POLICY_CLASSES.get(normalized_name)
    if policy_class is None:
        valid_names = ", ".join(sorted(BANDIT_POLICY_CLASSES))
        raise ValueError(f"Unknown bandit_policy={policy_name!r}; valid values: {valid_names}")
    return policy_class(
        total_layers=total_layers,
        default_boundaries=default_boundaries,
        world_size=world_size,
    )


class Scheduler:
    """Maintain per-batch layer allocations in scheduler.csv.

    Rank 0 calls get_or_create() to choose a batch allocation, then broadcasts
    allocation.boundaries to the other ranks. Worker ranks do not need a local
    scheduler file; the Rank 0 scheduler file is the single source of truth.
    """

    def __init__(
        self,
        allocation_csv,
        total_layers,
        default_boundaries,
        world_size,
        bandit_policy="ucb",
        experiment_scenario=None,
    ):
        self.path = Path(allocation_csv)
        self.total_layers = int(total_layers)
        self.world_size = int(world_size)
        self.default_boundaries = [int(value) for value in default_boundaries]
        self.fieldnames = ["batch"] + [f"rank{rank}" for rank in range(self.world_size)]
        self.bandit_policy_name = normalize_bandit_policy_name(bandit_policy)
        self.experiment_scenario = str(experiment_scenario or "").strip().upper()
        self.run_timestamp = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
        self.run_id = self.run_timestamp
        self.bandit_log_dir = self.path.parent / "bandit_logs"
        self.summary_path = self.bandit_log_dir / f"scheduler_summary_{self.run_timestamp}.csv"
        self.arm_details_path = self.bandit_log_dir / f"arm_details_{self.run_timestamp}.csv"
        experiment_label = self.experiment_scenario or "mixed"
        self.batch_metrics_path = self.bandit_log_dir / (
            f"batch_metrics_{self.bandit_policy_name}_{experiment_label}_{self.run_timestamp}.csv"
        )
        self.run_summary_path = self.bandit_log_dir / (
            f"run_summary_{self.bandit_policy_name}_{experiment_label}_{self.run_timestamp}.csv"
        )
        self.summary_fieldnames = [
            "batch",
            "prefill_mode",
            "layer_allocation",
            "rank",
            "time_label",
            "time_ms",
        ]
        self.arm_details_fieldnames = [
            "batch",
            "policy",
            "phase",
            "label_used",
            "scenario",
            "inferred_request_type",
            "target_output_tokens",
            "model_updated",
            "pull_completed",
            "pull_index",
            "updated_arm",
            "arm",
            "pulls_before",
            "pulls",
            "pulls_delta",
            "reward_before",
            "reward",
            "reward_delta",
            "score_before",
            "score",
            "score_delta",
            "confidence_lower",
            "confidence_upper",
            "confidence_width",
            "q1",
            "q2",
            "q1_delta",
            "q2_delta",
            "lipschitz_constant",
            "w1",
            "w2",
            "active",
            "active_arms",
            "selected",
            "next_selected",
        ]
        self.batch_metrics_fieldnames = [
            "run_id",
            "policy",
            "scenario",
            "batch",
            "prompt_index",
            "input_tokens",
            "target_output_tokens",
            "decode_step_count",
            "used_arm",
            "next_arm",
            "pull_completed",
            "bottleneck_time_ms",
            "cost_ms_per_step",
            "observed_reward",
            "cumulative_reward",
            "scheduler_select_ms",
            "scheduler_update_ms",
            "scheduler_algorithm_ms",
            "scheduler_algorithm_cumulative_ms",
            "partition_transition_wall_ms",
            "rank0_layer_switch_ms",
            "inference_wall_ms",
            "inference_cumulative_ms",
            "batch_total_wall_ms",
        ]
        self.run_summary_fieldnames = [
            "run_id",
            "policy",
            "scenario",
            "status",
            "total_batches",
            "measured_batches",
            "total_target_output_tokens",
            "total_decode_steps",
            "total_observed_reward",
            "mean_observed_reward",
            "total_scheduler_select_ms",
            "total_scheduler_update_ms",
            "total_scheduler_algorithm_ms",
            "total_partition_transition_ms",
            "total_inference_wall_ms",
            "measured_inference_wall_ms",
            "total_batch_wall_ms",
            "mean_inference_wall_ms",
            "p50_inference_wall_ms",
            "p95_inference_wall_ms",
            "algorithm_overhead_percent",
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
        self.algorithm_timing_history = {}
        self.batch_metric_history = []
        self.cumulative_algorithm_ms = 0.0
        self.cumulative_inference_ms = 0.0
        self.cumulative_reward = 0.0

        self._validate_boundaries(self.default_boundaries)
        self._load_existing_file()
        self.bandit = create_bandit_policy(
            policy_name=self.bandit_policy_name,
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

    def collect_batch_summary(
        self,
        batch,
        prefill_mode,
        boundaries,
        records,
        context=None,
        selected_arm=None,
    ):
        """Collect the minimal online data needed by future scheduling policy.

        This method intentionally mirrors the active value shown in
        "--- summary after batch ...". It stores only batch number, layer
        allocation, rank, one mode-dependent timing value per rank, and the
        batch context used by contextual policies.
        """
        batch = int(batch)
        boundaries = [int(value) for value in boundaries]
        layer_allocation = self._allocation_text(boundaries)
        time_label = self._time_label_for_mode(prefill_mode)
        rank_times = {}
        decode_step_count = 0
        for record in records:
            if not isinstance(record, dict) or "rank" not in record:
                continue
            rank = int(record["rank"])
            rank_times[rank] = self._summary_time_for_record(prefill_mode, record)
            decode_step_count = max(decode_step_count, int(record.get("decode_step_count", 0)))

        summary_arm = selected_arm
        if summary_arm is None and self.world_size == 3:
            summary_arm = tuple(boundaries[1:-1])
        if summary_arm is not None:
            summary_arm = tuple(int(value) for value in summary_arm)
        self.batch_summary_history[batch] = {
            "batch": batch,
            "prefill_mode": prefill_mode,
            "layer_allocation": layer_allocation,
            "boundaries": boundaries,
            "arm": summary_arm,
            "context": dict(context) if context else None,
            "decode_step_count": decode_step_count,
            "time_label": time_label,
            "rank_times": rank_times,
        }

    def select_arm_before_batch(self, batch, context=None):
        """Select the layer-allocation arm for the current batch."""
        batch = int(batch)
        started_ns = time.perf_counter_ns()
        selected_arm = self.bandit.select_arm(context=context, batch=batch)
        elapsed_ms = (time.perf_counter_ns() - started_ns) / 1_000_000.0
        self.algorithm_timing_history.setdefault(batch, {})["select_ms"] = elapsed_ms
        return selected_arm

    def update_policy_after_batch(self, batch):
        """Update the active policy from the completed batch summary."""
        batch = int(batch)
        pulls_before = int(self.bandit.total_pulls)
        started_ns = time.perf_counter_ns()
        selected_arm = self.bandit.update_after_batch(
            batch=batch,
            batch_summary_history=self.batch_summary_history,
        )
        update_ms = (time.perf_counter_ns() - started_ns) / 1_000_000.0
        timing = self.algorithm_timing_history.setdefault(batch, {})
        timing["update_ms"] = update_ms
        timing["algorithm_ms"] = float(timing.get("select_ms", 0.0)) + update_ms
        timing["pull_completed"] = int(self.bandit.total_pulls) > pulls_before
        self.cumulative_algorithm_ms += timing["algorithm_ms"]
        timing["algorithm_cumulative_ms"] = self.cumulative_algorithm_ms
        return selected_arm

    def run_bandit_after_batch(self, batch):
        """Backward-compatible wrapper for the previous after-batch hook."""
        selected_arm = self.update_policy_after_batch(batch)
        self.write_batch_audit(batch)
        return selected_arm

    def write_batch_audit(self, batch):
        """Persist existing scheduler summaries and per-arm audit rows."""
        summary = self.batch_summary_history.get(int(batch), {})
        used_arm = summary.get("arm", self.bandit.current_arm)
        self.append_arm_details(batch, used_arm)
        self.save_summary_history()

    def reallocate_layer(self, batch=None, arm=None, rank_metrics=None):
        """Write the selected current-batch layer allocation to scheduler.csv.

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

    def append_batch_metrics(
        self,
        batch,
        prompt_index,
        partition_transition_wall_ms,
        rank0_layer_switch_ms,
        inference_wall_ms,
        batch_total_wall_ms,
    ):
        """Append one plot-ready timing and reward row for a completed batch."""
        batch = int(batch)
        summary = self.batch_summary_history.get(batch, {})
        context = summary.get("context") or {}
        timing = self.algorithm_timing_history.get(batch, {})
        rank_times = summary.get("rank_times") or {}
        bottleneck_time_ms = max(rank_times.values()) if rank_times else None
        cost_ms_per_step = self.bandit._cost_per_token(summary)
        observed_reward = (
            self.bandit._reward_from_cost(cost_ms_per_step)
            if cost_ms_per_step is not None
            else None
        )
        pull_completed = bool(timing.get("pull_completed", False))
        if pull_completed and observed_reward is not None:
            self.cumulative_reward += observed_reward
        self.cumulative_inference_ms += float(inference_wall_ms)

        used_arm = summary.get("arm")
        next_arm = self.bandit.current_arm
        raw_row = {
            "run_id": self.run_id,
            "policy": self.bandit_policy_name,
            "scenario": context.get("scenario", self.experiment_scenario),
            "batch": batch,
            "prompt_index": int(prompt_index),
            "input_tokens": int(context.get("input_tokens_max", 0)),
            "target_output_tokens": context.get("target_output_tokens", ""),
            "decode_step_count": int(summary.get("decode_step_count", 0)),
            "used_arm": self.bandit._format_arm(used_arm) if used_arm is not None else "",
            "next_arm": self.bandit._format_arm(next_arm) if next_arm is not None else "",
            "pull_completed": int(pull_completed),
            "bottleneck_time_ms": bottleneck_time_ms,
            "cost_ms_per_step": cost_ms_per_step,
            "observed_reward": observed_reward,
            "cumulative_reward": self.cumulative_reward,
            "scheduler_select_ms": float(timing.get("select_ms", 0.0)),
            "scheduler_update_ms": float(timing.get("update_ms", 0.0)),
            "scheduler_algorithm_ms": float(timing.get("algorithm_ms", 0.0)),
            "scheduler_algorithm_cumulative_ms": float(
                timing.get("algorithm_cumulative_ms", self.cumulative_algorithm_ms)
            ),
            "partition_transition_wall_ms": float(partition_transition_wall_ms),
            "rank0_layer_switch_ms": float(rank0_layer_switch_ms),
            "inference_wall_ms": float(inference_wall_ms),
            "inference_cumulative_ms": self.cumulative_inference_ms,
            "batch_total_wall_ms": float(batch_total_wall_ms),
        }
        self.batch_metric_history.append(raw_row)

        self.batch_metrics_path.parent.mkdir(parents=True, exist_ok=True)
        file_exists = self.batch_metrics_path.exists()
        output = {}
        for field in self.batch_metrics_fieldnames:
            value = raw_row.get(field, "")
            output[field] = f"{value:.6f}" if isinstance(value, float) else value
        with open(self.batch_metrics_path, "a", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=self.batch_metrics_fieldnames)
            if not file_exists:
                writer.writeheader()
            writer.writerow(output)

    def write_run_summary(self, status):
        """Write one aggregate row for the current experiment run."""
        rows = list(self.batch_metric_history)
        measured_rows = [row for row in rows if int(row["pull_completed"]) == 1]
        inference_values = sorted(float(row["inference_wall_ms"]) for row in measured_rows)

        def percentile(values, fraction):
            if not values:
                return 0.0
            index = max(0, math.ceil(len(values) * fraction) - 1)
            return float(values[index])

        total_algorithm_ms = sum(float(row["scheduler_algorithm_ms"]) for row in rows)
        total_inference_ms = sum(float(row["inference_wall_ms"]) for row in rows)
        measured_inference_ms = sum(
            float(row["inference_wall_ms"]) for row in measured_rows
        )
        overhead_denominator = total_algorithm_ms + total_inference_ms
        summary = {
            "run_id": self.run_id,
            "policy": self.bandit_policy_name,
            "scenario": self.experiment_scenario,
            "status": status,
            "total_batches": len(rows),
            "measured_batches": len(measured_rows),
            "total_target_output_tokens": sum(
                int(row["target_output_tokens"])
                for row in rows
                if row["target_output_tokens"] != ""
            ),
            "total_decode_steps": sum(int(row["decode_step_count"]) for row in rows),
            "total_observed_reward": self.cumulative_reward,
            "mean_observed_reward": (
                self.cumulative_reward / len(measured_rows) if measured_rows else 0.0
            ),
            "total_scheduler_select_ms": sum(
                float(row["scheduler_select_ms"]) for row in rows
            ),
            "total_scheduler_update_ms": sum(
                float(row["scheduler_update_ms"]) for row in rows
            ),
            "total_scheduler_algorithm_ms": total_algorithm_ms,
            "total_partition_transition_ms": sum(
                float(row["partition_transition_wall_ms"]) for row in rows
            ),
            "total_inference_wall_ms": total_inference_ms,
            "measured_inference_wall_ms": measured_inference_ms,
            "total_batch_wall_ms": sum(float(row["batch_total_wall_ms"]) for row in rows),
            "mean_inference_wall_ms": (
                measured_inference_ms / len(measured_rows) if measured_rows else 0.0
            ),
            "p50_inference_wall_ms": percentile(inference_values, 0.50),
            "p95_inference_wall_ms": percentile(inference_values, 0.95),
            "algorithm_overhead_percent": (
                total_algorithm_ms / overhead_denominator * 100.0
                if overhead_denominator > 0.0
                else 0.0
            ),
        }

        self.run_summary_path.parent.mkdir(parents=True, exist_ok=True)
        output = {}
        for field in self.run_summary_fieldnames:
            value = summary.get(field, "")
            output[field] = f"{value:.6f}" if isinstance(value, float) else value
        with open(self.run_summary_path, "w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=self.run_summary_fieldnames)
            writer.writeheader()
            writer.writerow(output)

    def append_arm_details(self, batch, selected_arm):
        """Append one policy-state snapshot for every candidate arm."""
        self.arm_details_path.parent.mkdir(parents=True, exist_ok=True)
        file_exists = self.arm_details_path.exists()
        summary = self.batch_summary_history.get(int(batch), {})
        context = summary.get("context") or {}
        controlled_policy = self.bandit_policy_name == "contextual_controlled"
        with open(self.arm_details_path, "a", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=self.arm_details_fieldnames)
            if not file_exists:
                writer.writeheader()
            for row in self.bandit.arm_score_snapshot(selected_arm):
                output = {field: row.get(field, "") for field in self.arm_details_fieldnames}
                output["batch"] = int(batch)
                output["policy"] = self.bandit_policy_name
                if context:
                    output["phase"] = context.get("phase", "")
                    output["label_used"] = context.get("label_used", "")
                    output["scenario"] = context.get("scenario", "")
                    output["inferred_request_type"] = context.get(
                        "inferred_request_type",
                        context.get("request_type", ""),
                    )
                    target_output_tokens = context.get("target_output_tokens")
                    output["target_output_tokens"] = (
                        target_output_tokens
                        if target_output_tokens is not None
                        else f"natural(max={context.get('max_new_tokens', 512)})"
                    )
                if controlled_policy:
                    output["model_updated"] = int(
                        bool(getattr(self.bandit, "last_model_updated", False))
                    )
                writer.writerow(output)

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
