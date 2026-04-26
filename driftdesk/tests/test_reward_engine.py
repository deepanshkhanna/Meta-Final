"""
Tests for RewardEngine:
  - _is_grounded_edit() behavior
  - loop_penalty capped at 0.30
  - total reward capped at 1.0
  - no_spurious_rewrite logic (only credited when transient errors seen)
  - drift vs. clean episode reward weights
"""
import math

import pytest

from driftdesk.server.reward_engine import EpisodeRecord, RewardEngine


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_engine() -> RewardEngine:
    return RewardEngine(format_anneal_steps=50)


def complete_record(tasks=None, completed=None, steps=3, min_steps=3, max_steps=10):
    tasks     = tasks or ["airline_rebook", "bank_dispute", "insurance_claim"]
    completed = completed or list(tasks)
    r = EpisodeRecord()
    r.tasks = list(tasks)
    r.completed_tasks = list(completed)
    r.task_priorities = {t: i for i, t in enumerate(tasks)}
    r.steps_taken = steps
    r.min_steps_needed = min_steps
    r.max_steps = max_steps
    r.valid_json_actions = steps
    r.total_actions = steps
    return r


# ---------------------------------------------------------------------------
# _is_grounded_edit
# ---------------------------------------------------------------------------

class TestIsGroundedEdit:
    def test_adds_only_changed_field(self):
        old = {"passenger_name": "Alice", "flight_id": "UA123", "new_date": "2025-01-01"}
        new = {"passenger_name": "Alice", "flight_id": "UA123", "new_date": "2025-01-01", "seat_class": "business"}
        assert RewardEngine._is_grounded_edit(new, old, ["seat_class"]) is True

    def test_mutates_only_changed_field(self):
        old = {"passenger_name": "Alice", "flight_id": "UA123", "new_date": "2025-01-01"}
        new = {"passenger_name": "Alice", "flight_id": "UA456", "new_date": "2025-01-01"}
        assert RewardEngine._is_grounded_edit(new, old, ["flight_id"]) is True

    def test_mutates_wrong_field(self):
        old = {"passenger_name": "Alice", "flight_id": "UA123"}
        new = {"passenger_name": "Bob",   "flight_id": "UA123"}
        assert RewardEngine._is_grounded_edit(new, old, ["flight_id"]) is False

    def test_no_changes_returns_false(self):
        old = {"a": 1}
        new = {"a": 1}
        assert RewardEngine._is_grounded_edit(new, old, ["a"]) is False

    def test_changes_superset_of_changed_fields(self):
        old = {"a": 1, "b": 2}
        new = {"a": 9, "b": 9}
        # Changed both a and b, but only b was in the drift error → not grounded
        assert RewardEngine._is_grounded_edit(new, old, ["b"]) is False

    def test_adds_and_mutates_all_in_changed_fields(self):
        old = {"a": 1}
        new = {"a": 2, "c": 3}
        assert RewardEngine._is_grounded_edit(new, old, ["a", "c"]) is True


# ---------------------------------------------------------------------------
# loop_penalty is capped at 0.30
# ---------------------------------------------------------------------------

class TestLoopPenaltyCap:
    def test_cap_at_0_30(self):
        eng = make_engine()
        r = complete_record()
        r.repeated_failed_calls = 100  # 100 * 0.05 = 5.0 → should cap at 0.30
        _, comps = eng.compute_episode_reward(r, has_drift=False)
        assert comps["loop_penalty"] == pytest.approx(-0.30, abs=1e-6)

    def test_small_loops_not_capped(self):
        eng = make_engine()
        r = complete_record()
        r.repeated_failed_calls = 2   # 2 * 0.05 = 0.10
        _, comps = eng.compute_episode_reward(r, has_drift=False)
        assert comps["loop_penalty"] == pytest.approx(-0.10, abs=1e-6)

    def test_zero_loops(self):
        eng = make_engine()
        r = complete_record()
        r.repeated_failed_calls = 0
        _, comps = eng.compute_episode_reward(r, has_drift=False)
        assert comps["loop_penalty"] == pytest.approx(0.0, abs=1e-6)


# ---------------------------------------------------------------------------
# Total reward capped at 1.0
# ---------------------------------------------------------------------------

class TestRewardCap:
    def test_reward_never_exceeds_one(self):
        eng = make_engine()
        r = complete_record()
        r.drift_events_fired = 0
        r.valid_json_actions = r.total_actions
        r.repeated_failed_calls = 0
        r.policy_violations = 0
        total, _ = eng.compute_episode_reward(r, has_drift=False)
        assert total <= 1.0 + 1e-9

    def test_reward_can_be_negative(self):
        eng = make_engine()
        r = EpisodeRecord()
        r.tasks = ["airline_rebook"]
        r.completed_tasks = []
        r.task_priorities = {"airline_rebook": 0}
        r.steps_taken = 10
        r.min_steps_needed = 3
        r.max_steps = 10
        r.repeated_failed_calls = 10
        r.valid_json_actions = 0
        r.total_actions = 10
        total, _ = eng.compute_episode_reward(r, has_drift=False)
        assert total <= 0.0


# ---------------------------------------------------------------------------
# no_spurious_rewrite — only credited when transient errors were seen
# ---------------------------------------------------------------------------

class TestNoSpuriousRewrite:
    def test_nan_when_no_transient_seen(self):
        eng = make_engine()
        r = complete_record()
        r.transient_errors_seen = 0
        r.spurious_rewrites = 0
        _, comps = eng.compute_episode_reward(r, has_drift=True)
        assert math.isnan(comps["no_spurious_rewrite"]), (
            "no_spurious_rewrite should be NaN when no transient errors observed"
        )

    def test_one_when_no_spurious_rewrites(self):
        eng = make_engine()
        r = complete_record()
        r.transient_errors_seen = 3
        r.spurious_rewrites = 0
        _, comps = eng.compute_episode_reward(r, has_drift=True)
        assert comps["no_spurious_rewrite"] == pytest.approx(1.0, abs=1e-6)

    def test_decays_with_spurious_rewrites(self):
        eng = make_engine()
        r = complete_record()
        r.transient_errors_seen = 2
        r.spurious_rewrites = 1  # 1 - 0.5*1 = 0.5
        _, comps = eng.compute_episode_reward(r, has_drift=True)
        assert comps["no_spurious_rewrite"] == pytest.approx(0.5, abs=1e-6)

    def test_clamped_at_zero(self):
        eng = make_engine()
        r = complete_record()
        r.transient_errors_seen = 1
        r.spurious_rewrites = 100  # should clamp at 0.0
        _, comps = eng.compute_episode_reward(r, has_drift=True)
        assert comps["no_spurious_rewrite"] >= 0.0


# ---------------------------------------------------------------------------
# Episode reward with / without drift — weight check
# ---------------------------------------------------------------------------

class TestEpisodeRewardWeights:
    def test_drift_episode_uses_drift_recovery_weight(self):
        eng = make_engine()

        # Episode with perfect drift recovery
        r = complete_record(steps=4)
        r.drift_events_fired = 1
        r.error_grounded_edits = 1
        r.first_retry_successes = 1
        r.transient_errors_seen = 1
        r.spurious_rewrites = 0
        r.policy_violations = 0
        r.repeated_failed_calls = 0
        total_drift, comps = eng.compute_episode_reward(r, has_drift=True)

        # Should get significant drift recovery credit (weight 0.45)
        assert comps["drift_recovery"] > 0.5

        # Same record but no drift — drift_recovery not weighted
        total_clean, comps_clean = eng.compute_episode_reward(r, has_drift=False)
        # Clean episode task_completion has weight 0.5 — should dominate
        assert comps_clean["task_completion"] == pytest.approx(1.0, abs=1e-6)
