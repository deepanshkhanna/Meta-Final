"""
SchemaDriftController

Manages drift event scheduling and activation per episode.

Design:
- Each episode begins with a drift schedule (list of (step, module, new_version)).
- The controller exposes `maybe_drift(step)` which fires pending events.
- Supports 3 tracks:
    - "none"   : no drift (early curriculum)
    - "cued"   : drift fires with a version-header cue in responses (learnable)
    - "silent" : drift fires with no cue (hard track)
- Transient errors are injected periodically to prevent "always switch on error" hacking.
"""
from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

from schemas import REGISTRY, TRAIN_VERSIONS, DriftType


@dataclass
class DriftEvent:
    step: int
    module_name: str
    from_version: int
    to_version: int
    drift_types: List[DriftType]
    cued: bool = False              # whether a hint header is included in next response


@dataclass
class TransientError:
    """Fake error injected to prevent template-swap hacking on every 4xx."""
    step: int
    module_name: str


class SchemaDriftController:
    """Schedules and fires drift events within a single episode.

    Args:
        rng: Random state (seeded for reproducibility).
        drift_track: "none" | "cued" | "silent".
        max_drift_per_episode: Maximum number of drift events (default 2).
        min_step_before_first_drift: Agent gets at least N steps before any drift.
        transient_error_prob: Per-step probability of injecting a transient 500.
    """

    def __init__(
        self,
        rng: random.Random,
        drift_track: str = "cued",
        max_drift_per_episode: int = 2,
        min_step_before_first_drift: int = 1,
        transient_error_prob: float = 0.08,
    ) -> None:
        self._rng = rng
        self._drift_track = drift_track
        self._max_drift = max_drift_per_episode
        self._min_step = min_step_before_first_drift
        self._transient_prob = transient_error_prob

        self._active_versions: Dict[str, int] = {}
        self._schedule: List[DriftEvent] = []
        self._transients: List[TransientError] = []
        self._fired: List[DriftEvent] = []
        self._pending_cue_modules: List[str] = []   # modules whose next response gets a cue

    # ------------------------------------------------------------------
    # Episode initialisation
    # ------------------------------------------------------------------

    def reset(
        self,
        modules: List[str],
        episode_length: int,
        eval_mode: bool = False,
        eval_seed: Optional[int] = None,
    ) -> None:
        """Set up drift schedule for a new episode.

        Args:
            modules: Active task module names.
            episode_length: Max steps this episode.
            eval_mode: If True, uses fixed seed schedule (deterministic eval).
            eval_seed: Seed for eval-mode schedule.
        """
        rng = random.Random(eval_seed) if eval_mode and eval_seed is not None else self._rng

        # Reset state — ALL modules start at v1
        self._active_versions = {m: TRAIN_VERSIONS[0] for m in modules}
        self._schedule = []
        self._transients = []
        self._fired = []
        self._pending_cue_modules = []

        if self._drift_track == "none":
            return

        # Build drift schedule using a temporary version tracker
        # (does NOT mutate _active_versions — those only update when events fire)
        # In eval mode we always fire max_drift events; in train mode we fire
        # at least one (so curriculum stages with drift always exercise the
        # drift-recovery signal — see audit issue C3).
        if eval_mode:
            num_events = self._max_drift
        else:
            num_events = rng.randint(1, self._max_drift) if self._max_drift > 0 else 0
        candidate_modules = list(modules)
        rng.shuffle(candidate_modules)

        tmp_versions: Dict[str, int] = dict(self._active_versions)  # temporary for scheduling

        for i in range(min(num_events, len(candidate_modules))):
            mod = candidate_modules[i]
            current_v = tmp_versions[mod]
            # Move to v2 if at v1; v2 is the max training version
            next_v = TRAIN_VERSIONS[-1] if current_v < TRAIN_VERSIONS[-1] else current_v
            if next_v == current_v:
                continue

            # Schedule drift early so the agent always encounters it before
            # completing all tasks (3 modules → min 3 successful steps to win).
            # First drift in steps [min_step, min_step+1]; subsequent drifts can
            # spread out but must still land within the first 60% of the episode.
            if i == 0:
                earliest = self._min_step
                latest = max(earliest, self._min_step + 1)
            else:
                earliest = self._min_step + i
                latest = max(earliest, min(episode_length - 2, self._min_step + 2 + i))
            step = rng.randint(earliest, latest)

            schema = REGISTRY.get(mod, next_v)
            event = DriftEvent(
                step=step,
                module_name=mod,
                from_version=current_v,
                to_version=next_v,
                drift_types=schema.drift_from_previous,
                cued=(self._drift_track == "cued"),
            )
            self._schedule.append(event)
            tmp_versions[mod] = next_v  # update temp tracker for multi-event chains

        # Schedule transient errors (fake 500s) across random steps
        for step in range(self._min_step, episode_length):
            if rng.random() < self._transient_prob and candidate_modules:
                mod = rng.choice(candidate_modules)
                self._transients.append(TransientError(step=step, module_name=mod))

    # ------------------------------------------------------------------
    # Per-step queries
    # ------------------------------------------------------------------

    def maybe_drift(self, step: int) -> List[DriftEvent]:
        """Fire any scheduled drift events at this step.

        Returns list of events fired (usually 0 or 1).
        Side-effect: updates _active_versions to the new schema version.
        """
        fired: List[DriftEvent] = []
        for evt in self._schedule:
            if evt.step == step and evt not in self._fired:
                # Update the active version NOW (drift fires at this step)
                self._active_versions[evt.module_name] = evt.to_version
                self._fired.append(evt)
                fired.append(evt)
                if evt.cued:
                    self._pending_cue_modules.append(evt.module_name)
        return fired

    def get_transient_error(self, step: int, module_name: str) -> Optional[Dict[str, Any]]:
        """Return a transient error payload if one is scheduled for this step+module, else None."""
        for t in self._transients:
            if t.step == step and t.module_name == module_name:
                return {
                    "code": "TRANSIENT_ERROR",
                    "http_status": 500,
                    "message": "Upstream service temporarily unavailable. Retry unchanged.",
                    "module": module_name,
                    "is_transient": True,
                }
        return None

    def active_version(self, module_name: str) -> int:
        return self._active_versions.get(module_name, TRAIN_VERSIONS[0])

    def consume_cue(self, module_name: str) -> bool:
        """Returns True (and removes) if a cue is pending for this module."""
        if module_name in self._pending_cue_modules:
            self._pending_cue_modules.remove(module_name)
            return True
        return False

    def all_active_versions(self) -> Dict[str, int]:
        return dict(self._active_versions)

    def fired_events(self) -> List[Dict[str, Any]]:
        return [
            {
                "step": e.step,
                "module": e.module_name,
                "from_version": e.from_version,
                "to_version": e.to_version,
                "drift_types": [d.value for d in e.drift_types],
                "cued": e.cued,
            }
            for e in self._fired
        ]

    def total_drift_events(self) -> int:
        return len(self._fired)
