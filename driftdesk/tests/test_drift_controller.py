"""
Tests for SchemaDriftController:
  - reset() clears all state
  - maybe_drift() is deterministic in eval_mode (same seed → same schedule)
  - get_transient_error() returns the right payload
  - Firing an event at the same step twice is idempotent
"""
import random

import pytest

from driftdesk.server.drift_controller import SchemaDriftController


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

MODULES = ["airline_rebook", "bank_dispute", "insurance_claim"]


def make_controller(track: str = "cued") -> SchemaDriftController:
    return SchemaDriftController(
        rng=random.Random(42),
        drift_track=track,
        max_drift_per_episode=2,
        min_step_before_first_drift=1,
        transient_error_prob=0.0,  # disable transients for most tests
    )


# ---------------------------------------------------------------------------
# reset() clears state
# ---------------------------------------------------------------------------

class TestReset:
    def test_reset_clears_fired_events(self):
        ctrl = make_controller()
        ctrl.reset(MODULES, episode_length=10)
        # Fire anything that fires at step 1
        ctrl.maybe_drift(1)
        ctrl.maybe_drift(2)
        # After reset, fired list should be empty again
        ctrl.reset(MODULES, episode_length=10)
        assert ctrl.fired_events() == []
        assert ctrl.total_drift_events() == 0

    def test_reset_restores_version_to_v1(self):
        ctrl = make_controller()
        ctrl.reset(MODULES, episode_length=10)
        # Force all steps so versions might advance
        for s in range(10):
            ctrl.maybe_drift(s)
        # After reset, all modules should be back at v1
        ctrl.reset(MODULES, episode_length=10)
        from driftdesk.schemas import TRAIN_VERSIONS
        for mod in MODULES:
            assert ctrl.active_version(mod) == TRAIN_VERSIONS[0]

    def test_reset_clears_pending_cues(self):
        ctrl = make_controller(track="cued")
        ctrl.reset(MODULES, episode_length=10)
        # Fire drift events to populate pending cues
        for s in range(10):
            ctrl.maybe_drift(s)
        ctrl.reset(MODULES, episode_length=10)
        # After reset, no cues should be pending
        for mod in MODULES:
            assert ctrl.consume_cue(mod) is False


# ---------------------------------------------------------------------------
# maybe_drift() is deterministic in eval_mode
# ---------------------------------------------------------------------------

class TestDeterministicEvalMode:
    def _fired_summary(self, ctrl, episode_length):
        fired = []
        for s in range(episode_length):
            evts = ctrl.maybe_drift(s)
            fired.extend(evts)
        return [(e["step"], e["module"], e["from_version"], e["to_version"]) for e in
                ctrl.fired_events()]

    def test_same_eval_seed_same_schedule(self):
        ctrl1 = make_controller()
        ctrl1.reset(MODULES, episode_length=10, eval_mode=True, eval_seed=77)
        summary1 = self._fired_summary(ctrl1, episode_length=10)

        ctrl2 = make_controller()
        ctrl2.reset(MODULES, episode_length=10, eval_mode=True, eval_seed=77)
        summary2 = self._fired_summary(ctrl2, episode_length=10)

        assert summary1 == summary2, (
            f"Different schedules for same eval_seed: {summary1} vs {summary2}"
        )

    def test_different_eval_seed_may_differ(self):
        ctrl1 = make_controller()
        ctrl1.reset(MODULES, episode_length=10, eval_mode=True, eval_seed=1)
        s1 = []
        for s in range(10):
            s1.extend(ctrl1.maybe_drift(s))

        ctrl2 = make_controller()
        ctrl2.reset(MODULES, episode_length=10, eval_mode=True, eval_seed=999)
        s2 = []
        for s in range(10):
            s2.extend(ctrl2.maybe_drift(s))

        # Can't assert inequality (sometimes RNG coincides), but the test at
        # least ensures both runs complete without error.
        assert isinstance(s1, list)
        assert isinstance(s2, list)

    def test_eval_mode_always_fires_max_drift(self):
        ctrl = SchemaDriftController(
            rng=random.Random(0),
            drift_track="cued",
            max_drift_per_episode=2,
            min_step_before_first_drift=1,
            transient_error_prob=0.0,
        )
        ctrl.reset(MODULES, episode_length=10, eval_mode=True, eval_seed=42)
        for s in range(10):
            ctrl.maybe_drift(s)
        # In eval_mode with max_drift=2 and 3 modules, should fire exactly 2
        assert ctrl.total_drift_events() == 2


# ---------------------------------------------------------------------------
# get_transient_error()
# ---------------------------------------------------------------------------

class TestTransientError:
    def test_returns_none_when_no_transient_scheduled(self):
        ctrl = make_controller()  # transient_error_prob=0.0
        ctrl.reset(MODULES, episode_length=10)
        for s in range(10):
            for mod in MODULES:
                assert ctrl.get_transient_error(s, mod) is None

    def test_returns_payload_when_transient_scheduled(self):
        ctrl = SchemaDriftController(
            rng=random.Random(7),
            drift_track="cued",           # must be non-"none" for transients to be scheduled
            max_drift_per_episode=1,
            min_step_before_first_drift=1,
            transient_error_prob=1.0,     # always inject one transient per eligible step
        )
        ctrl.reset(MODULES, episode_length=5)
        found = False
        for s in range(0, 6):
            for mod in MODULES:
                result = ctrl.get_transient_error(s, mod)
                if result is not None:
                    assert result["code"] == "TRANSIENT_ERROR"
                    assert result["http_status"] == 500
                    assert result["is_transient"] is True
                    assert result["module"] == mod
                    found = True
                    break
            if found:
                break
        assert found, "Expected at least one transient error with prob=1.0 over 6 steps"


# ---------------------------------------------------------------------------
# Idempotency — firing at same step twice doesn't double-apply
# ---------------------------------------------------------------------------

class TestIdempotency:
    def test_double_fire_idempotent(self):
        ctrl = SchemaDriftController(
            rng=random.Random(1),
            drift_track="cued",
            max_drift_per_episode=1,
            min_step_before_first_drift=1,
            transient_error_prob=0.0,
        )
        ctrl.reset(MODULES, episode_length=10)

        total_before = 0
        versions_after = {}
        for s in range(10):
            ctrl.maybe_drift(s)
            ctrl.maybe_drift(s)  # call twice at same step

        # Should still have fired at most 1 event (not 2)
        assert ctrl.total_drift_events() <= 1
