"""
DriftDeskEnvironment — main OpenEnv Environment subclass.

Implements: reset(), step(), state property.

Episode flow:
  1. reset(seed) → SchemaDriftController builds drift schedule, PolicyInjector generates
     policy doc, all task modules start at v1. Returns DriftDeskObservation.
  2. step(action) → DriftController fires pending drifts, TaskModule.execute() validates
     payload against current schema, RewardEngine updates record.
  3. Episode ends when all tasks completed OR max_steps reached.
  4. On done=True, episode reward is computed and returned in the final observation.
"""
from __future__ import annotations

import random
import os
from typing import Any, Dict, List, Optional
from uuid import uuid4

from openenv.core.env_server.interfaces import Environment
from openenv.core.env_server.types import State

from driftdesk.models import DriftDeskAction, DriftDeskObservation, DriftDeskState
from driftdesk.schemas import REGISTRY, TRAIN_VERSIONS
from driftdesk.server.drift_controller import SchemaDriftController
from driftdesk.server.policy_injector import PolicyDocumentInjector
from driftdesk.server.reward_engine import EpisodeRecord, RewardEngine
from driftdesk.server.task_modules.airline import AirlineRebookModule
from driftdesk.server.task_modules.bank import BankDisputeModule
from driftdesk.server.task_modules.insurance import InsuranceClaimModule


# ------------------------------------------------------------------
# Constants
# ------------------------------------------------------------------

MAX_STEPS = 10
MODULES = ["airline_rebook", "bank_dispute", "insurance_claim"]

# Curriculum stages: (drift_track, min_step_before_first_drift)
CURRICULUM = [
    ("none", 99),    # stage 0: no drift
    ("cued", 1),     # stage 1: cued drift (easy)
    ("silent", 1),   # stage 2: silent drift (hard)
]


class DriftDeskEnvironment(Environment):
    """DriftDesk RL environment — personal assistant with schema/policy drift."""

    SUPPORTS_CONCURRENT_SESSIONS: bool = True

    def __init__(
        self,
        curriculum_stage: int = 1,
        eval_mode: bool = False,
        eval_seed: Optional[int] = None,
    ) -> None:
        super().__init__()

        self._curriculum_stage = min(curriculum_stage, len(CURRICULUM) - 1)
        self._eval_mode = eval_mode
        self._eval_seed = eval_seed

        drift_track, min_step = CURRICULUM[self._curriculum_stage]

        self._rng = random.Random()
        self._drift_ctrl = SchemaDriftController(
            rng=self._rng,
            drift_track=drift_track,
            max_drift_per_episode=2,
            min_step_before_first_drift=min_step,
            transient_error_prob=0.08,
        )
        self._policy_injector = PolicyDocumentInjector()
        self._reward_engine = RewardEngine(format_anneal_steps=50)

        # Task module instances (stateful: hold pre-auth tokens)
        self._task_modules = {
            "airline_rebook": AirlineRebookModule(),
            "bank_dispute": BankDisputeModule(),
            "insurance_claim": InsuranceClaimModule(),
        }

        # Episode state
        self._state = DriftDeskState(episode_id=str(uuid4()), step_count=0)
        self._record = EpisodeRecord()
        self._policy_doc: str = ""
        self._task_list: List[Dict[str, Any]] = []
        self._has_drift: bool = False

        # Track last payload per module for spurious-rewrite detection
        self._last_payload: Dict[str, Any] = {}
        self._last_result: Dict[str, Any] = {}
        self._last_was_transient: Dict[str, bool] = {}

    # ------------------------------------------------------------------
    # reset
    # ------------------------------------------------------------------

    def reset(
        self,
        seed: Optional[int] = None,
        episode_id: Optional[str] = None,
        curriculum_stage: Optional[int] = None,
        **kwargs: Any,
    ) -> DriftDeskObservation:
        if seed is not None:
            self._rng.seed(seed)

        if curriculum_stage is not None:
            stage = min(int(curriculum_stage), len(CURRICULUM) - 1)
            self._curriculum_stage = stage
            drift_track, min_step = CURRICULUM[stage]
            self._drift_ctrl._drift_track = drift_track
            self._drift_ctrl._min_step = min_step

        eid = episode_id or str(uuid4())

        # Reset task modules
        for mod in self._task_modules.values():
            mod.reset()

        # Build drift schedule
        self._drift_ctrl.reset(
            modules=MODULES,
            episode_length=MAX_STEPS,
            eval_mode=self._eval_mode,
            eval_seed=self._eval_seed if self._eval_mode else seed,
        )

        self._has_drift = self._drift_ctrl._drift_track != "none"

        # Generate policy document (v1 schemas only)
        self._policy_doc = self._policy_injector.generate(MODULES)

        # Build task list with random priorities
        priorities = list(range(len(MODULES)))
        self._rng.shuffle(priorities)
        self._task_list = [
            {
                "module": mod,
                "description": self._task_description(mod),
                "priority": pri,
                "status": "pending",
                "completed": False,
            }
            for mod, pri in zip(MODULES, priorities)
        ]

        # Reset episode record
        self._record = EpisodeRecord(
            tasks=list(MODULES),
            task_priorities={mod: pri for mod, pri in zip(MODULES, priorities)},
            max_steps=MAX_STEPS,
            min_steps_needed=len(MODULES),
        )

        # Reset tracking
        self._last_payload = {}
        self._last_result = {}
        self._last_was_transient = {}

        # Reset state
        self._state = DriftDeskState(
            episode_id=eid,
            step_count=0,
            active_schema_versions=self._drift_ctrl.all_active_versions(),
            task_statuses={t["module"]: "pending" for t in self._task_list},
        )

        return DriftDeskObservation(
            policy_doc=self._policy_doc,
            tasks=self._task_list,
            last_result={"status": "ready", "message": "Episode started. Complete all tasks."},
            step_count=0,
            episode_id=eid,
            done=False,
            reward=None,
        )

    # ------------------------------------------------------------------
    # step
    # ------------------------------------------------------------------

    def step(
        self,
        action: DriftDeskAction,
        timeout_s: Optional[float] = None,
        **kwargs: Any,
    ) -> DriftDeskObservation:
        self._state.step_count += 1
        step = self._state.step_count
        # Format validity: callers (e.g. the GRPO trainer) may pass the result of
        # parsing the raw model completion via kwargs["is_valid_json"]. If absent,
        # we fall back to True since the action already round-tripped through
        # Pydantic at the FastAPI layer (see audit issue C4).
        is_valid_json = bool(kwargs.get("is_valid_json", True))

        module_key = action.module
        is_pre_auth = ":pre_auth" in module_key
        module_name = module_key.replace(":pre_auth", "")

        # Guard: unknown module
        if module_name not in self._task_modules:
            result = {
                "code": "UNKNOWN_MODULE",
                "message": f"Unknown module: {module_name!r}. Valid: {list(self._task_modules)}",
                "http_status": 400,
            }
            return self._build_obs(result, step, done=False, reward=None)

        # Detect spurious rewrite BEFORE we overwrite last-payload tracking:
        # if the previous step on this module returned a transient error, the
        # agent should resend the *same* payload. Any change is spurious.
        if (
            self._last_was_transient.get(module_name)
            and not is_pre_auth
            and module_name in self._last_payload
        ):
            self._reward_engine.record_spurious_rewrite(
                self._record,
                module_name,
                action.payload,
                self._last_payload[module_name],
            )

        # Fire pending drift events for this step
        fired = self._drift_ctrl.maybe_drift(step)
        for evt in fired:
            self._state.drift_events_fired.append(evt.__class__.__name__ if hasattr(evt, "__class__") else str(evt))
            self._state.total_drift_events += 1
            self._record.drift_events_fired += 1

        # Check for transient error injection (before real execution)
        transient = self._drift_ctrl.get_transient_error(step, module_name)
        if transient and not is_pre_auth:
            # Record transient observation in the reward engine so that
            # `no_spurious_rewrite` becomes meaningful for this episode.
            self._record.transient_errors_seen += 1
            self._last_payload[module_name] = action.payload
            self._last_was_transient[module_name] = True
            self._last_result[module_name] = transient
            return self._build_obs(transient, step, done=False, reward=None)

        # Clear transient flag since this step has real execution
        self._last_was_transient[module_name] = False

        # Execute action
        task_mod = self._task_modules[module_name]
        if is_pre_auth:
            success, result = task_mod.execute_pre_auth()
            if success:
                self._state.pre_auth_tokens[module_name] = result["pre_auth_token"]
        else:
            current_version = self._drift_ctrl.active_version(module_name)
            hint = self._drift_ctrl.consume_cue(module_name)
            success, result = task_mod.execute(action.payload, current_version, hint=hint)

        # Compute drift-error metadata for reward engine
        is_drift_error = (not success) and (result.get("code") == "DRIFT")
        is_transient_error = (not success) and (result.get("code") == "TRANSIENT_ERROR")
        drift_changed_fields = result.get("changed_fields") if is_drift_error else None

        pre_drift_payload = self._last_payload.get(module_name)
        self._last_payload[module_name] = action.payload

        # Cue version header in result if drift was cued
        # (The hint is already inside the result payload via to_error_payload_with_hint)

        # Update reward engine record
        step_reward = self._reward_engine.record_action(
            record=self._record,
            module=module_name,
            payload=action.payload,
            success=success,
            result=result,
            is_drift_error=is_drift_error,
            is_transient_error=is_transient_error,
            drift_changed_fields=drift_changed_fields,
            is_valid_json=is_valid_json,
            pre_drift_payload=pre_drift_payload,
        )

        # Update task status
        if success and not is_pre_auth:
            for task in self._task_list:
                if task["module"] == module_name:
                    task["status"] = "completed"
                    task["completed"] = True
            self._state.task_statuses[module_name] = "completed"

        # Update state
        self._state.active_schema_versions = self._drift_ctrl.all_active_versions()
        self._state.drift_events_fired = self._drift_ctrl.fired_events()
        self._state.successful_recoveries = self._record.successful_recoveries
        self._state.repeated_failed_calls = self._record.repeated_failed_calls

        # Check done
        all_done = all(t["completed"] for t in self._task_list)
        timed_out = step >= MAX_STEPS
        done = all_done or timed_out

        episode_reward: Optional[float] = None
        components: Optional[Dict[str, float]] = None
        if done:
            episode_reward, components = self._reward_engine.compute_episode_reward(
                self._record, self._has_drift
            )
            self._state.reward_components = components

        return self._build_obs(result, step, done=done, reward=episode_reward or step_reward,
                               reward_components=components)

    # ------------------------------------------------------------------
    # state property
    # ------------------------------------------------------------------

    @property
    def state(self) -> DriftDeskState:
        return self._state

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _build_obs(
        self,
        result: Dict[str, Any],
        step: int,
        done: bool,
        reward: Optional[float],
        reward_components: Optional[Dict[str, float]] = None,
    ) -> DriftDeskObservation:
        return DriftDeskObservation(
            policy_doc=self._policy_doc,
            tasks=list(self._task_list),
            last_result=result,
            step_count=step,
            episode_id=self._state.episode_id,
            done=done,
            reward=reward,
            reward_components=reward_components or {},
        )

    @staticmethod
    def _task_description(module_name: str) -> str:
        descs = {
            "airline_rebook": (
                "Rebook flight AI-202 for passenger 'Jordan Lee' to 2026-05-10. "
                "Reason: WORK conference rescheduled."
            ),
            "bank_dispute": (
                "File a dispute for account ACC-9921, amount $450.00, merchant 'FastCharge Inc', "
                "description 'Unauthorized charge', type FRAUD."
            ),
            "insurance_claim": (
                "Submit insurance claim for claimant CLM-5501, incident date 2026-04-15, "
                "amount $1200.00, description 'Laptop damaged in flood'."
            ),
        }
        return descs.get(module_name, f"Complete task for module {module_name}.")
