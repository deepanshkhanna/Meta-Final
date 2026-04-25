"""
Insurance claim task module.

Schema v1: {claimant_id, incident_date, amount, description}
Schema v2: restructures to {claimant_id, incident_id, amount, line_items} (FIELD_RENAME + ADD + REMOVE)
Schema 99 (held-out): FLOW_RESTRUCTURE — requires pre-auth token
"""
from __future__ import annotations

from typing import Any, Dict, Tuple

from schemas import REGISTRY
from server.task_modules.base import TaskModule


class InsuranceClaimModule(TaskModule):
    module_name = "insurance_claim"

    def execute(
        self,
        payload: Dict[str, Any],
        schema_version: int,
        hint: bool = False,
    ) -> Tuple[bool, Dict[str, Any]]:
        schema = REGISTRY.get(self.module_name, schema_version)

        # FLOW_RESTRUCTURE: check pre-auth token first
        if schema.pre_step_required:
            provided_token = payload.get("pre_auth_token")
            if not provided_token or provided_token != self._pre_auth_token:
                err: Dict[str, Any] = {
                    "code": "DRIFT",
                    "missing_fields": ["pre_auth_token"],
                    "changed_fields": ["pre_auth_token"],
                    "http_status": 403,
                    "module": self.module_name,
                }
                if hint:
                    err["hint"] = (
                        "A pre-authorisation step is now required. "
                        "Call GET /insurance/pre-auth first to obtain a token."
                    )
                return False, err

        required = set(schema.required_field_names())
        provided = set(k for k in payload if not k.startswith("_"))

        missing = sorted(required - provided)
        unexpected = sorted(provided - {f.name for f in schema.active_fields()})

        if missing or unexpected:
            err2 = (
                schema.to_error_payload_with_hint(missing, unexpected)
                if hint
                else schema.to_error_payload(missing, unexpected)
            )
            return False, {**err2, "http_status": 422, "module": self.module_name}

        # Validate line_items structure for v2+
        if "line_items" in payload:
            items = payload["line_items"]
            if not isinstance(items, list) or not items:
                return False, {
                    "code": "VALIDATION_ERROR",
                    "field": "line_items",
                    "message": "line_items must be a non-empty list of {code, cost} objects",
                    "http_status": 400,
                    "module": self.module_name,
                }

        return True, {
            "status": "ok",
            "module": self.module_name,
            "claim_id": f"CLM-{abs(hash(str(payload))) % 100000:05d}",
        }
