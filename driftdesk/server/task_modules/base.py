"""Task module base class."""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Dict, Optional, Tuple


class TaskModule(ABC):
    """Each task module encapsulates one real-world task type.

    It validates an agent action payload against the *currently active*
    schema version and returns (success, result_payload).
    """

    module_name: str

    def __init__(self) -> None:
        self._pre_auth_token: Optional[str] = None

    @abstractmethod
    def execute(
        self,
        payload: Dict[str, Any],
        schema_version: int,
        hint: bool = False,
    ) -> Tuple[bool, Dict[str, Any]]:
        """Validate and execute the action.

        Args:
            payload: Agent-supplied key-value pairs.
            schema_version: Currently active schema version (1, 2, or 99).
            hint: If True, error payloads include the 'hint' field.

        Returns:
            (success, result_payload)
            On success: result_payload contains {"status": "ok", ...task_data}.
            On failure: result_payload is the structured drift error body.
        """

    def execute_pre_auth(self) -> Tuple[bool, Dict[str, Any]]:
        """Execute the pre-authorisation step for FLOW_RESTRUCTURE schemas.

        Returns:
            (True, {"pre_auth_token": <token>}) always — pre-auth cannot fail.
        """
        import uuid
        token = str(uuid.uuid4())
        self._pre_auth_token = token
        return True, {"status": "ok", "pre_auth_token": token}

    def reset(self) -> None:
        self._pre_auth_token = None
