"""
PolicyDocumentInjector

Generates a natural-language policy document at the start of each episode.
The document encodes the *initial* schema rules (v1 for all modules).
Mid-episode drift is NOT reflected in the policy doc — the agent must infer it
from error signals (cued or uncued).

Policy doc is capped at ~400 tokens to avoid blowing context.
"""
from __future__ import annotations

from typing import Dict, List

from schemas import REGISTRY, TRAIN_VERSIONS


_POLICY_TEMPLATE = """\
# DriftDesk Executive Assistant — Policy Document
*Episode policy effective at session start. Rules may be updated; always verify with live API.*

## Priority Rules
- Priority 0 tasks MUST be completed before Priority 1 tasks.
- If a task cannot be completed due to an error, document the error and proceed to the next task.
- Do not abandon a task after a single failure — attempt recovery before moving on.

## Airline Rebooking (module: airline_rebook)
Endpoint: POST /airline/rebook
Required fields: {airline_fields}
Notes: Use exact date format YYYY-MM-DD. Confirm rebooking with the returned confirmation_code.

## Bank Dispute (module: bank_dispute)
Endpoint: POST /bank/dispute
Required fields: {bank_fields}
Notes: amount must be a number (float). Record the returned case_id.

## Insurance Claim (module: insurance_claim)
Endpoint: POST /insurance/claims
Required fields: {insurance_fields}
Notes: description should be concise (max 200 characters). Record the returned claim_id.

## Error Handling Protocol
- HTTP 422 with code=DRIFT: A schema field has changed. Read the error body carefully.
  The 'changed_fields' key lists affected fields. Adapt your payload accordingly.
- HTTP 422 with code=VALIDATION_ERROR: Invalid field value. Check 'allowed' values in error body.
- HTTP 500 with code=TRANSIENT_ERROR: Temporary outage. Retry the same payload unchanged.
- Do NOT mutate your payload on a TRANSIENT_ERROR. Only mutate on DRIFT errors.

## Response Format
Always respond with a JSON action in this format:
{{"module": "<module_name>", "payload": {{...fields...}}}}
For pre-auth step: {{"module": "<module_name>:pre_auth", "payload": {{}}}}
"""


class PolicyDocumentInjector:
    """Generates episode-start policy documents from schema v1 field specs."""

    def generate(self, active_modules: List[str]) -> str:
        """Generate a policy doc for the given active modules using their v1 schemas."""
        field_strs: Dict[str, str] = {}
        for mod in active_modules:
            schema = REGISTRY.get(mod, TRAIN_VERSIONS[0])
            fields = ", ".join(
                f"{f.name} ({f.type}, {'required' if f.required else 'optional'})"
                for f in schema.active_fields()
            )
            field_strs[mod] = fields

        return _POLICY_TEMPLATE.format(
            airline_fields=field_strs.get("airline_rebook", "N/A"),
            bank_fields=field_strs.get("bank_dispute", "N/A"),
            insurance_fields=field_strs.get("insurance_claim", "N/A"),
        ).strip()
