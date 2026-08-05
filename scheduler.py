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
        "output_estimate": 384.0,
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
        "output_estimate": 72.0,
    },
)

CONTEXT_INPUT_TOKEN_SCALE = 1500.0
CONTEXT_OUTPUT_TOKEN_SCALE = 512.0
CONTEXT_BATCH_SIZE_SCALE = 128.0
CONTEXT_VECTOR_SIZE = 4
# Number of mandatory observations for every actual arm under each request type.
# Change this value to adjust contextual warmup without hard-coding an arm count.
CONTEXT_WARMUP_PULLS = 1

CANDIDATE_ARMS = [
    (1, 11),
    (1, 19),
    (1, 7),
    (3, 11),
    (9, 19),
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
        window_size=2,
        warmup_skip=1,
        exploration_weight=0.01,
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
        """Observe one completed batch and update the policy state."""
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

        # If a manual row or a different policy changes the split, treat it as
        # the active arm and start a fresh measurement window.
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
        stats = self.stats[arm]
        pulls = int(stats["pulls"])
        stats["mean_cost"] = (float(stats["mean_cost"]) * pulls + float(cost)) / (pulls + 1)
        stats["last_cost"] = float(cost)
        stats["reward"] = self._reward_from_cost(float(stats["mean_cost"]))
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

    def arm_score_snapshot(self, selected_arm):
        """Return compact per-arm reward/score rows for logging."""
        selected_arm = tuple(selected_arm) if selected_arm is not None else None
        rows = []
        log_total = math.log(max(self.total_pulls, 2))
        for arm in self.arms:
            stats = self.stats[arm]
            pulls = int(stats["pulls"])
            reward = float(stats["reward"])
            if pulls == 0:
                score = "untried"
            else:
                score_value = reward + self.exploration_weight * math.sqrt(log_total / pulls)
                score = f"{score_value:.6f}"
            rows.append(
                {
                    "arm": self._format_arm(arm),
                    "reward": f"{reward:.6f}",
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


class LipschitzBanditPolicy(LayerBanditPolicy):
    """Simple Lipschitz-UCB policy for nearby layer-split sharing.

    Reward updates reuse LayerBanditPolicy's unified per-token absolute reward.
    Arm selection replaces the raw arm reward with a Lipschitz-smoothed reward
    estimate from already observed nearby arms.
    """

    policy_name = "lipschitz"
    lipschitz_constant = 0.5

    def _select_next_arm(self):
        """Choose the next arm with a Lipschitz-smoothed UCB score."""
        for arm in self.arms:
            if int(self.stats[arm]["pulls"]) == 0:
                return arm

        log_total = math.log(max(self.total_pulls, 2))
        best_arm = self.arms[0]
        best_score = None
        for arm in self.arms:
            pulls = int(self.stats[arm]["pulls"])
            smooth_reward = self._lipschitz_reward_estimate(arm)
            exploration = self.exploration_weight * math.sqrt(log_total / pulls)
            score = smooth_reward + exploration
            if best_score is None or score > best_score:
                best_arm = arm
                best_score = score
        return best_arm

    def arm_score_snapshot(self, selected_arm):
        """Return scores that expose the active Lipschitz-UCB selection value."""
        selected_arm = tuple(selected_arm) if selected_arm is not None else None
        rows = []
        log_total = math.log(max(self.total_pulls, 2))
        for arm in self.arms:
            stats = self.stats[arm]
            pulls = int(stats["pulls"])
            reward = float(stats["reward"])
            if pulls == 0:
                score = "untried"
            else:
                smooth_reward = self._lipschitz_reward_estimate(arm)
                score_value = smooth_reward + self.exploration_weight * math.sqrt(
                    log_total / pulls
                )
                score = f"{score_value:.6f}"
            rows.append(
                {
                    "arm": self._format_arm(arm),
                    "reward": f"{reward:.6f}",
                    "score": score,
                    "selected": 1 if arm == selected_arm else 0,
                }
            )
        return rows

    def _lipschitz_reward_estimate(self, arm):
        """Estimate an arm reward from observed nearby arms."""
        observed_arms = [
            observed_arm
            for observed_arm in self.arms
            if int(self.stats[observed_arm]["pulls"]) > 0
        ]
        if not observed_arms:
            return 0.5
        return max(
            float(self.stats[observed_arm]["reward"])
            - self.lipschitz_constant * self.arm_distance(arm, observed_arm)
            for observed_arm in observed_arms
        )

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
    ):
        self.path = Path(allocation_csv)
        self.total_layers = int(total_layers)
        self.world_size = int(world_size)
        self.default_boundaries = [int(value) for value in default_boundaries]
        self.fieldnames = ["batch"] + [f"rank{rank}" for rank in range(self.world_size)]
        self.run_timestamp = datetime.now().strftime("%Y-%m-%d-%H-%M")
        self.bandit_log_dir = self.path.parent / "bandit_logs"
        self.summary_path = self.bandit_log_dir / f"scheduler_summary_{self.run_timestamp}.csv"
        self.arm_details_path = self.bandit_log_dir / f"arm_details_{self.run_timestamp}.csv"
        self.summary_fieldnames = [
            "batch",
            "prefill_mode",
            "layer_allocation",
            "rank",
            "time_label",
            "time_ms",
        ]
        self.arm_details_fieldnames = ["batch", "arm", "reward", "score", "selected"]
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
        self.bandit_policy_name = normalize_bandit_policy_name(bandit_policy)
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
        self.save_summary_history()

    def select_arm_before_batch(self, batch, context=None):
        """Select the layer-allocation arm for the current batch."""
        _ = batch
        return self.bandit.select_arm(context=context, batch=batch)

    def update_policy_after_batch(self, batch):
        """Update the active policy from the completed batch summary."""
        selected_arm = self.bandit.update_after_batch(
            batch=batch,
            batch_summary_history=self.batch_summary_history,
        )
        summary = self.batch_summary_history.get(int(batch), {})
        used_arm = summary.get("arm", selected_arm)
        self.append_arm_details(batch, used_arm)
        return selected_arm

    def run_bandit_after_batch(self, batch):
        """Backward-compatible wrapper for the previous after-batch hook."""
        return self.update_policy_after_batch(batch)

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

    def append_arm_details(self, batch, selected_arm):
        """Append one compact reward/score snapshot for every candidate arm."""
        self.arm_details_path.parent.mkdir(parents=True, exist_ok=True)
        file_exists = self.arm_details_path.exists()
        with open(self.arm_details_path, "a", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=self.arm_details_fieldnames)
            if not file_exists:
                writer.writeheader()
            for row in self.bandit.arm_score_snapshot(selected_arm):
                writer.writerow(
                    {
                        "batch": int(batch),
                        "arm": row["arm"],
                        "reward": row["reward"],
                        "score": row["score"],
                        "selected": row["selected"],
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
