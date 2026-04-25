"""
RewardEngine — 7-component decomposed reward for DriftDesk.

Components (episode-type-conditioned):

DRIFT EPISODES (at least one drift event fired):
  task_completion       weight 0.25
  drift_recovery        weight 0.45  (3 sub-signals averaged)
  policy_grounding      weight 0.10
  priority_score        weight 0.10
  efficiency            weight 0.10
  format_valid          annealed  (starts 0.05, decays to 0 by step 50)
  loop_penalty          capped at -0.30 (applied inside sum)

CLEAN EPISODES (no drift events):
  task_completion       weight 0.50
  policy_grounding      weight 0.10
  priority_score        weight 0.20
  efficiency            weight 0.20
  format_valid          same annealing
  loop_penalty          same cap

drift_recovery sub-signals:
  1. error_grounded_edit   ∈ {0,1}:
       Agent's next action after drift-error edits *exactly* the changed_fields named in error.
  2. first_retry_success   ∈ {0,1}:
       Agent succeeds within 1 retry after drift (vs brute-force multi-retry).
  3. no_spurious_rewrite   ∈ {0,1}:
       On TRANSIENT_ERROR (not a schema change), agent does NOT mutate its payload.

Anti-gaming:
  - loop_penalty: capped linear penalty per repeated-identical-failure.
  - no_spurious_rewrite: penalises "always switch template on any error" exploit.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple


@dataclass
class EpisodeRecord:
    """Tracks per-episode signals needed for reward computation."""

    # Task completion
    tasks: List[str] = field(default_factory=list)               # module names
    completed_tasks: List[str] = field(default_factory=list)
    task_priorities: Dict[str, int] = field(default_factory=dict) # module -> priority (0=highest)

    # Drift recovery tracking
    drift_events_fired: int = 0
    successful_recoveries: int = 0    # agent adapted correctly after a drift error
    spurious_rewrites: int = 0        # agent mutated payload after a transient error
    first_retry_successes: int = 0    # succeeded on first retry after drift

    # Pending drift state (per module)
    last_drift_error_module: Optional[str] = None
    last_drift_error_fields: List[str] = field(default_factory=list)
    post_drift_attempt: Dict[str, bool] = field(default_factory=dict)  # module -> tried_after_drift
    post_drift_success: Dict[str, bool] = field(default_factory=dict)  # module -> succeeded

    # Error grounding
    error_grounded_edits: int = 0    # agent mentioned changed_fields in next action

    # Steps and efficiency
    max_steps: int = 10
    steps_taken: int = 0
    min_steps_needed: int = 3        # theoretical minimum

    # Loop penalty
    repeated_failed_calls: int = 0
    last_failed_call: Optional[Tuple[str, str]] = None  # (module, error_code)

    # Format validity
    valid_json_actions: int = 0
    total_actions: int = 0

    # Transient errors observed (needed to know whether no_spurious_rewrite is meaningful)
    transient_errors_seen: int = 0

    # Policy grounding
    policy_violations: int = 0       # agent sent field present in policy as required but missing


class RewardEngine:
    """Computes step-level and episode-level rewards."""

    def __init__(self, format_anneal_steps: int = 50) -> None:
        self._format_anneal_steps = format_anneal_steps

    # ------------------------------------------------------------------
    # Step-level signals (called after each env.step)
    # ------------------------------------------------------------------

    def record_action(
        self,
        record: EpisodeRecord,
        module: str,
        payload: Dict[str, Any],
        success: bool,
        result: Dict[str, Any],
        is_drift_error: bool,
        is_transient_error: bool,
        drift_changed_fields: Optional[List[str]],
        is_valid_json: bool,
        pre_drift_payload: Optional[Dict[str, Any]],
    ) -> float:
        """Update episode record with the result of one agent action.

        Returns a step-level partial reward (format + loop signals only;
        main reward is computed at episode end).
        """
        record.total_actions += 1
        record.steps_taken += 1

        if is_valid_json:
            record.valid_json_actions += 1

        if is_transient_error:
            record.transient_errors_seen += 1

        # Drift-recovery tracking
        if is_drift_error and drift_changed_fields is not None:
            record.drift_events_fired = max(record.drift_events_fired, 1)  # at least counted
            record.last_drift_error_module = module
            record.last_drift_error_fields = drift_changed_fields
            record.post_drift_attempt[module] = False
            record.post_drift_success[module] = False

        elif record.post_drift_attempt.get(module) is False:
            # This is the first action after a drift error for this module
            record.post_drift_attempt[module] = True
            if success:
                record.post_drift_success[module] = True
                record.successful_recoveries += 1
                record.first_retry_successes += 1

                # Check error-grounded edit: did agent change exactly the drifted fields?
                if (
                    pre_drift_payload is not None
                    and drift_changed_fields
                    and self._is_grounded_edit(payload, pre_drift_payload, drift_changed_fields)
                ):
                    record.error_grounded_edits += 1
            # else: recovery failed, may retry (next step)

        elif success and record.last_drift_error_module == module:
            # Subsequent successful retry (not first)
            if not record.post_drift_success.get(module, False):
                record.post_drift_success[module] = True
                record.successful_recoveries += 1
                # no first_retry credit

        # Spurious rewrite detection
        if is_transient_error and pre_drift_payload is not None:
            # Agent received a transient error — check if next action mutates payload
            pass  # We track this on the *next* action; see record_spurious_rewrite()

        # Loop penalty
        current_fail = (module, result.get("code", "")) if not success else None
        if current_fail and current_fail == record.last_failed_call:
            record.repeated_failed_calls += 1
        record.last_failed_call = current_fail if not success else None

        if success and module not in record.completed_tasks:
            record.completed_tasks.append(module)

        # Step-level partial reward (format only)
        return self._format_reward(record)

    def record_spurious_rewrite(
        self,
        record: EpisodeRecord,
        module: str,
        new_payload: Dict[str, Any],
        prev_payload: Dict[str, Any],
    ) -> None:
        """Call this when an action follows a TRANSIENT_ERROR for the same module.

        If the agent mutated its payload after a transient error, penalise.
        """
        if new_payload != prev_payload:
            record.spurious_rewrites += 1

    # ------------------------------------------------------------------
    # Episode-level reward (called at episode end or done=True)
    # ------------------------------------------------------------------

    def compute_episode_reward(
        self,
        record: EpisodeRecord,
        has_drift: bool,
    ) -> Tuple[float, Dict[str, float]]:
        """Compute total episode reward and per-component breakdown.

        Returns:
            (total_reward, component_dict)
        """
        components: Dict[str, float] = {}

        # 1. Task completion
        tc = len(record.completed_tasks) / max(len(record.tasks), 1)
        components["task_completion"] = tc

        # 2. Drift recovery (only meaningful when drift occurred).
        # `no_spurious_rewrite` is only credited when the episode actually exposed
        # transient errors; otherwise it is excluded from the average to avoid
        # the previous reward-inflation path (audit issue C3).
        dr = self._drift_recovery(record)
        components["drift_recovery"] = dr
        components["drift_recovery_error_grounded"] = (
            record.error_grounded_edits / max(record.drift_events_fired, 1)
        )
        components["drift_recovery_first_retry"] = (
            record.first_retry_successes / max(record.drift_events_fired, 1)
        )
        if record.transient_errors_seen > 0:
            components["no_spurious_rewrite"] = max(
                0.0, 1.0 - 0.5 * record.spurious_rewrites
            )
        else:
            # Not measured this episode — report neutral marker, do NOT inflate dr.
            components["no_spurious_rewrite"] = float("nan")

        # 3. Policy grounding
        pg = max(0.0, 1.0 - 0.25 * record.policy_violations)
        components["policy_grounding"] = pg

        # 4. Priority score
        ps = self._priority_score(record)
        components["priority_score"] = ps

        # 5. Efficiency
        eff = self._efficiency(record)
        components["efficiency"] = eff

        # 6. Format validity (annealed — at episode end use final annealing weight)
        fmt_w = self._format_anneal_weight(record.steps_taken)
        fmt = record.valid_json_actions / max(record.total_actions, 1)
        components["format_valid"] = fmt

        # 7. Loop penalty (capped)
        lp = min(0.30, 0.05 * record.repeated_failed_calls)
        components["loop_penalty"] = -lp

        # --- Weighted sum (conditioned on drift vs. clean) ---
        if has_drift:
            total = (
                0.25 * tc
                + 0.45 * dr
                + 0.10 * pg
                + 0.10 * ps
                + 0.10 * eff
                + fmt_w * fmt
                - lp
            )
        else:
            total = (
                0.50 * tc
                + 0.10 * pg
                + 0.20 * ps
                + 0.20 * eff
                + fmt_w * fmt
                - lp
            )

        # Clamp at +1.0 (component upper bound) but allow negative values so
        # that the loop_penalty / spurious-rewrite signals provide gradient
        # near the failure regime (audit issue C7).
        total = min(1.0, total)
        components["total"] = total
        return total, components

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _drift_recovery(self, record: EpisodeRecord) -> float:
        if record.drift_events_fired == 0 and record.transient_errors_seen == 0:
            return 0.0

        signals: List[float] = []
        if record.drift_events_fired > 0:
            n = record.drift_events_fired
            signals.append(record.error_grounded_edits / n)
            signals.append(record.first_retry_successes / n)
        if record.transient_errors_seen > 0:
            signals.append(
                max(0.0, 1.0 - 0.5 * record.spurious_rewrites)
            )
        if not signals:
            return 0.0
        return sum(signals) / len(signals)

    def _priority_score(self, record: EpisodeRecord) -> float:
        if not record.completed_tasks or not record.task_priorities:
            return 0.0
        # Check that each completed task was done before any lower-priority (higher number) ones
        completion_order = record.completed_tasks  # list in completion order
        violations = 0
        for i, mod in enumerate(completion_order):
            pri = record.task_priorities.get(mod, 99)
            # All tasks completed before this one should have <= priority number
            for earlier in completion_order[:i]:
                if record.task_priorities.get(earlier, 99) > pri:
                    violations += 1
        max_possible_violations = len(completion_order) * (len(completion_order) - 1) / 2
        if max_possible_violations == 0:
            return 1.0
        return max(0.0, 1.0 - violations / max_possible_violations)

    def _efficiency(self, record: EpisodeRecord) -> float:
        extra = record.steps_taken - record.min_steps_needed
        budget = record.max_steps - record.min_steps_needed
        if budget <= 0:
            return 1.0
        return max(0.0, 1.0 - extra / budget)

    def _format_anneal_weight(self, step: int) -> float:
        if self._format_anneal_steps <= 0:
            return 0.0
        progress = min(1.0, step / self._format_anneal_steps)
        return 0.05 * (1.0 - progress)

    def _format_reward(self, record: EpisodeRecord) -> float:
        w = self._format_anneal_weight(record.steps_taken)
        if record.total_actions == 0:
            return 0.0
        fmt = record.valid_json_actions / record.total_actions
        return w * fmt

    @staticmethod
    def _is_grounded_edit(
        new_payload: Dict[str, Any],
        old_payload: Dict[str, Any],
        changed_fields: List[str],
    ) -> bool:
        """True if agent changed *only* the fields listed in the drift error."""
        added = set(new_payload) - set(old_payload)
        removed = set(old_payload) - set(new_payload)
        mutated = {k for k in new_payload if k in old_payload and new_payload[k] != old_payload[k]}
        all_changes = added | removed | mutated
        if not all_changes:
            return False
        return all_changes.issubset(set(changed_fields))
