"""
Bank dispute task module.

Schema v1: {account_id, amount, merchant, description}
Schema v2: adds dispute_type (FIELD_ADD)
Schema 99 (held-out): removes fields, renames, moves endpoint
"""
from __future__ import annotations

from typing import Any, Dict, Tuple

from driftdesk.schemas import REGISTRY
from driftdesk.server.task_modules.base import TaskModule


class BankDisputeModule(TaskModule):
    module_name = "bank_dispute"

    def execute(
        self,
        payload: Dict[str, Any],
        schema_version: int,
        hint: bool = False,
    ) -> Tuple[bool, Dict[str, Any]]:
        schema = REGISTRY.get(self.module_name, schema_version)
        required = set(schema.required_field_names())
        provided = set(k for k in payload if not k.startswith("_"))

        missing = sorted(required - provided)
        unexpected = sorted(provided - {f.name for f in schema.active_fields()})

        if missing or unexpected:
            err = (
                schema.to_error_payload_with_hint(missing, unexpected)
                if hint
                else schema.to_error_payload(missing, unexpected)
            )
            return False, {**err, "http_status": 422, "module": self.module_name}

        for f in schema.active_fields():
            if f.allowed_values and payload.get(f.name) not in f.allowed_values:
                return False, {
                    "code": "VALIDATION_ERROR",
                    "field": f.name,
                    "allowed": f.allowed_values,
                    "got": payload.get(f.name),
                    "http_status": 400,
                    "module": self.module_name,
                }

        return True, {
            "status": "ok",
            "module": self.module_name,
            "case_id": f"DISP-{abs(hash(str(payload))) % 100000:05d}",
        }
