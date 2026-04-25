"""
Data models for DriftDesk environment.

Action: agent's tool call (module + payload).
Observation: what the agent sees each step (task list, last result, policy doc, etc.).
State: internal bookkeeping returned to training infrastructure.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from openenv.core.env_server.types import Action, Observation, State as _OpenEnvState
from pydantic import Field


# ---------------------------------------------------------------------------
# Action
# ---------------------------------------------------------------------------

class DriftDeskAction(Action):
    """A single tool call from the agent.

    The agent selects a module and provides a JSON payload.
    The environment validates the payload against the current (possibly drifted)
    schema and returns the result.

    Special pre_auth actions:
        Set module = "<module>:pre_auth" to execute the pre-authorisation step
        required by FLOW_RESTRUCTURE schemas.
    """

    module: str = Field(
        ...,
        description=(
            "Target task module. One of: 'airline_rebook', 'bank_dispute', "
            "'insurance_claim', or '<module>:pre_auth' for pre-auth step."
        ),
    )
    payload: Dict[str, Any] = Field(
        default_factory=dict,
        description="Key-value fields for the API call. Must match current schema.",
    )


# ---------------------------------------------------------------------------
# Observation
# ---------------------------------------------------------------------------

class TaskStatus(str):
    PENDING = "pending"
    COMPLETED = "completed"
    FAILED = "failed"


class DriftDeskObservation(Observation):
    """Full observation returned to the agent after reset or step.

    Fields the agent uses to decide its next action:
    - policy_doc: natural-language description of current policies/schema hints.
    - tasks: list of task descriptors (module, description, priority, status).
    - last_result: structured result of the last action (success or error body).
    - step_count: how many steps have been taken this episode.
    - episode_id: unique episode identifier.
    """

    policy_doc: str = Field(
        default="",
        description="Natural-language policy document injected at episode start.",
    )
    tasks: List[Dict[str, Any]] = Field(
        default_factory=list,
        description=(
            "List of task descriptors: {module, description, priority, status, "
            "completed}. Priority 0 is highest."
        ),
    )
    last_result: Dict[str, Any] = Field(
        default_factory=dict,
        description=(
            "Result of the last env.step() call. On error contains 'code', "
            "'missing_fields', 'changed_fields', 'hint' etc."
        ),
    )
    step_count: int = Field(default=0, description="Steps taken so far in this episode.")
    episode_id: str = Field(default="", description="Unique episode identifier.")


# ---------------------------------------------------------------------------
# State (internal — returned to training infra via /state)
# ---------------------------------------------------------------------------

class DriftDeskState(_OpenEnvState):
    """Internal environment state exposed via GET /state."""

    active_schema_versions: Dict[str, int] = Field(
        default_factory=dict,
        description="Current schema version per module, e.g. {'airline_rebook': 2}.",
    )
    drift_events_fired: List[Dict[str, Any]] = Field(
        default_factory=list,
        description="Log of drift events that occurred this episode.",
    )
    task_statuses: Dict[str, str] = Field(
        default_factory=dict,
        description="Per-module task status: pending | completed | failed.",
    )
    successful_recoveries: int = Field(
        default=0,
        description="Number of times agent recovered correctly after a drift error.",
    )
    total_drift_events: int = Field(
        default=0,
        description="Total drift events fired in this episode.",
    )
    repeated_failed_calls: int = Field(
        default=0,
        description="Number of repeated failed calls (same module, same error).",
    )
    pre_auth_tokens: Dict[str, str] = Field(
        default_factory=dict,
        description="Issued pre-auth tokens per module (for FLOW_RESTRUCTURE schemas).",
    )
    reward_components: Dict[str, float] = Field(
        default_factory=dict,
        description="Latest per-component reward breakdown.",
    )
