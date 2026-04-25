"""
DummyDriftEnv — a stub environment with the same OpenEnv interface as DriftDeskEnvironment.

Purpose: Lets Member C wire up TRL + GRPO training code against this stub while
Members A/B build the real environment in parallel. Returns random rewards so
the training loop can be validated end-to-end before the real env is ready.

Usage:
    from dummy_env import DummyDriftEnv
    env = DummyDriftEnv()
    obs = env.reset(seed=42)
    obs = env.step(action)
"""
from __future__ import annotations

import random
import sys
import os
from typing import Any, Optional
from uuid import uuid4

sys.path.insert(0, os.path.dirname(__file__))

from openenv.core.env_server.interfaces import Environment
from openenv.core.env_server.types import State

from models import DriftDeskAction, DriftDeskObservation


_DUMMY_POLICY = (
    "# DriftDesk Policy (Dummy)\n"
    "Complete all three tasks: airline_rebook, bank_dispute, insurance_claim.\n"
    "Respond with JSON: {\"module\": \"<name>\", \"payload\": {...}}\n"
)

_DUMMY_TASKS = [
    {"module": "airline_rebook", "description": "Rebook flight AI-202.", "priority": 0, "status": "pending", "completed": False},
    {"module": "bank_dispute",   "description": "File bank dispute.",     "priority": 1, "status": "pending", "completed": False},
    {"module": "insurance_claim","description": "Submit insurance claim.", "priority": 2, "status": "pending", "completed": False},
]


class DummyDriftEnv(Environment):
    """Stub environment. Returns random rewards; never crashes."""

    SUPPORTS_CONCURRENT_SESSIONS: bool = True

    def __init__(self) -> None:
        super().__init__()
        self._rng = random.Random()
        self._state = State(episode_id=str(uuid4()), step_count=0)
        self._tasks = [dict(t) for t in _DUMMY_TASKS]
        self._step_count = 0

    def reset(
        self,
        seed: Optional[int] = None,
        episode_id: Optional[str] = None,
        **kwargs: Any,
    ) -> DriftDeskObservation:
        if seed is not None:
            self._rng.seed(seed)
        eid = episode_id or str(uuid4())
        self._state = State(episode_id=eid, step_count=0)
        self._tasks = [dict(t) for t in _DUMMY_TASKS]
        self._step_count = 0
        return DriftDeskObservation(
            policy_doc=_DUMMY_POLICY,
            tasks=list(self._tasks),
            last_result={"status": "ready"},
            step_count=0,
            episode_id=eid,
            done=False,
            reward=None,
        )

    def step(
        self,
        action: DriftDeskAction,
        timeout_s: Optional[float] = None,
        **kwargs: Any,
    ) -> DriftDeskObservation:
        self._step_count += 1
        self._state.step_count = self._step_count

        module = action.module.replace(":pre_auth", "")
        success = self._rng.random() > 0.4   # ~60% random success

        result = (
            {"status": "ok", "module": module}
            if success
            else {
                "code": "DRIFT",
                "missing_fields": ["reason_code"],
                "changed_fields": ["reason_code"],
                "http_status": 422,
                "module": module,
            }
        )

        if success:
            for t in self._tasks:
                if t["module"] == module:
                    t["status"] = "completed"
                    t["completed"] = True

        all_done = all(t["completed"] for t in self._tasks)
        timed_out = self._step_count >= 10
        done = all_done or timed_out

        reward = self._rng.uniform(0.0, 0.3) if not success else self._rng.uniform(0.3, 1.0)
        if not done:
            reward = None

        return DriftDeskObservation(
            policy_doc=_DUMMY_POLICY,
            tasks=list(self._tasks),
            last_result=result,
            step_count=self._step_count,
            episode_id=self._state.episode_id,
            done=done,
            reward=reward,
        )

    @property
    def state(self) -> State:
        return self._state
