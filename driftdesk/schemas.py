"""
Schema DSL for DriftDesk.

Defines typed field specs and schema versions. All task modules are built
from these primitives, enabling programmatic held-out schema generation
and fine-grained drift-type tagging per field.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional


# ---------------------------------------------------------------------------
# Drift taxonomy (6 canonical types)
# ---------------------------------------------------------------------------

class DriftType(str, Enum):
    FIELD_ADD = "FIELD_ADD"
    FIELD_RENAME = "FIELD_RENAME"
    FIELD_REMOVE = "FIELD_REMOVE"
    TYPE_CHANGE = "TYPE_CHANGE"
    ENDPOINT_MOVE = "ENDPOINT_MOVE"
    FLOW_RESTRUCTURE = "FLOW_RESTRUCTURE"


# ---------------------------------------------------------------------------
# Field-level spec
# ---------------------------------------------------------------------------

@dataclass
class FieldSpec:
    name: str
    type: str                       # "str" | "int" | "float" | "bool" | "list"
    required: bool = True
    version_introduced: int = 1
    version_deprecated: Optional[int] = None
    description: str = ""
    allowed_values: Optional[List[Any]] = None  # enum constraint

    def active_in(self, version: int) -> bool:
        deprecated = self.version_deprecated
        return (
            self.version_introduced <= version
            and (deprecated is None or version < deprecated)
        )


# ---------------------------------------------------------------------------
# Schema version
# ---------------------------------------------------------------------------

@dataclass
class SchemaVersion:
    module_name: str
    version: int                        # 1, 2, or 99 for held-out
    endpoint: str                       # e.g. "/rebook"
    method: str                         # "POST" | "DELETE" | "GET"
    fields: List[FieldSpec]
    drift_from_previous: List[DriftType] = field(default_factory=list)
    pre_step_required: bool = False     # True = multi-step flow (FLOW_RESTRUCTURE)
    pre_step_endpoint: Optional[str] = None

    def active_fields(self) -> List[FieldSpec]:
        return [f for f in self.fields if f.active_in(self.version)]

    def required_field_names(self) -> List[str]:
        return [f.name for f in self.active_fields() if f.required]

    def to_error_payload(self, missing: List[str], unexpected: List[str]) -> Dict[str, Any]:
        return {
            "code": "DRIFT",
            "missing_fields": missing,
            "unexpected_fields": unexpected,
            "changed_fields": missing + unexpected,
            "hint": None,               # hard track by default
        }

    def to_error_payload_with_hint(self, missing: List[str], unexpected: List[str]) -> Dict[str, Any]:
        payload = self.to_error_payload(missing, unexpected)
        payload["hint"] = (
            f"Schema updated. Required fields are now: {self.required_field_names()}"
        )
        return payload


# ---------------------------------------------------------------------------
# Schema Registry — single source of truth
# ---------------------------------------------------------------------------

class SchemaRegistry:
    """Holds all module schemas.  Schemas are defined once; versions materialise lazily."""

    def __init__(self) -> None:
        self._registry: Dict[str, Dict[int, SchemaVersion]] = {}

    def register(self, schema: SchemaVersion) -> None:
        self._registry.setdefault(schema.module_name, {})[schema.version] = schema

    def get(self, module_name: str, version: int) -> SchemaVersion:
        try:
            return self._registry[module_name][version]
        except KeyError:
            raise ValueError(
                f"Schema not found: module={module_name!r} version={version}"
            )

    def versions(self, module_name: str) -> List[int]:
        return sorted(self._registry.get(module_name, {}).keys())

    def all_modules(self) -> List[str]:
        return list(self._registry.keys())


# ---------------------------------------------------------------------------
# Build global registry
# ---------------------------------------------------------------------------

def _build_registry() -> SchemaRegistry:
    reg = SchemaRegistry()

    # ------------------------------------------------------------------ #
    # MODULE: airline_rebook                                               #
    # ------------------------------------------------------------------ #
    reg.register(SchemaVersion(
        module_name="airline_rebook",
        version=1,
        endpoint="/airline/rebook",
        method="POST",
        fields=[
            FieldSpec("flight_id", "str", required=True, version_introduced=1,
                      description="Flight identifier (e.g. AI-202)"),
            FieldSpec("passenger_name", "str", required=True, version_introduced=1,
                      description="Full passenger name"),
            FieldSpec("new_date", "str", required=True, version_introduced=1,
                      description="New travel date in YYYY-MM-DD format"),
        ],
    ))

    reg.register(SchemaVersion(
        module_name="airline_rebook",
        version=2,
        endpoint="/airline/rebook",
        method="POST",
        fields=[
            FieldSpec("flight_id", "str", required=True, version_introduced=1),
            FieldSpec("passenger_name", "str", required=True, version_introduced=1),
            FieldSpec("new_date", "str", required=True, version_introduced=1),
            FieldSpec("reason_code", "str", required=True, version_introduced=2,
                      description="Mandatory reason for rebooking",
                      allowed_values=["MEDICAL", "WORK", "WEATHER", "OTHER"]),
        ],
        drift_from_previous=[DriftType.FIELD_ADD],
    ))

    # Held-out (version 99): endpoint moves + field rename
    reg.register(SchemaVersion(
        module_name="airline_rebook",
        version=99,
        endpoint="/airline/rebook/v2",
        method="POST",
        fields=[
            FieldSpec("pnr", "str", required=True, version_introduced=99,
                      description="Booking reference (replaces flight_id)"),
            FieldSpec("traveler_id", "str", required=True, version_introduced=99,
                      description="Traveler ID (replaces passenger_name)"),
            FieldSpec("departure_date", "str", required=True, version_introduced=99,
                      description="Departure date (replaces new_date)"),
            FieldSpec("reason_code", "str", required=True, version_introduced=2),
        ],
        drift_from_previous=[DriftType.FIELD_RENAME, DriftType.ENDPOINT_MOVE],
    ))

    # ------------------------------------------------------------------ #
    # MODULE: bank_dispute                                                 #
    # ------------------------------------------------------------------ #
    reg.register(SchemaVersion(
        module_name="bank_dispute",
        version=1,
        endpoint="/bank/dispute",
        method="POST",
        fields=[
            FieldSpec("account_id", "str", required=True, version_introduced=1,
                      description="Bank account identifier"),
            FieldSpec("amount", "float", required=True, version_introduced=1,
                      description="Disputed amount in USD"),
            FieldSpec("merchant", "str", required=True, version_introduced=1,
                      description="Merchant name"),
            FieldSpec("description", "str", required=True, version_introduced=1,
                      description="Description of dispute"),
        ],
    ))

    reg.register(SchemaVersion(
        module_name="bank_dispute",
        version=2,
        endpoint="/bank/dispute",
        method="POST",
        fields=[
            FieldSpec("account_id", "str", required=True, version_introduced=1),
            FieldSpec("amount", "float", required=True, version_introduced=1),
            FieldSpec("merchant", "str", required=True, version_introduced=1),
            FieldSpec("description", "str", required=True, version_introduced=1),
            FieldSpec("dispute_type", "str", required=True, version_introduced=2,
                      description="Category of dispute",
                      allowed_values=["FRAUD", "DUPLICATE", "NOT_RECEIVED", "QUALITY"]),
        ],
        drift_from_previous=[DriftType.FIELD_ADD],
    ))

    # Held-out: removes fields, renames, moves endpoint
    reg.register(SchemaVersion(
        module_name="bank_dispute",
        version=99,
        endpoint="/bank/disputes/new",
        method="POST",
        fields=[
            FieldSpec("transaction_id", "str", required=True, version_introduced=99,
                      description="Transaction ID (replaces account_id + merchant combo)"),
            FieldSpec("dispute_type", "str", required=True, version_introduced=2),
            FieldSpec("evidence", "str", required=False, version_introduced=99,
                      description="Optional evidence URL"),
        ],
        drift_from_previous=[DriftType.FIELD_REMOVE, DriftType.FIELD_RENAME,
                              DriftType.ENDPOINT_MOVE],
    ))

    # ------------------------------------------------------------------ #
    # MODULE: insurance_claim                                              #
    # ------------------------------------------------------------------ #
    reg.register(SchemaVersion(
        module_name="insurance_claim",
        version=1,
        endpoint="/insurance/claims",
        method="POST",
        fields=[
            FieldSpec("claimant_id", "str", required=True, version_introduced=1,
                      description="Claimant identifier"),
            FieldSpec("incident_date", "str", required=True, version_introduced=1,
                      description="Date of incident YYYY-MM-DD"),
            FieldSpec("amount", "float", required=True, version_introduced=1,
                      description="Claimed amount in USD"),
            FieldSpec("description", "str", required=True, version_introduced=1,
                      description="Description of incident"),
        ],
    ))

    reg.register(SchemaVersion(
        module_name="insurance_claim",
        version=2,
        endpoint="/insurance/claims",
        method="POST",
        fields=[
            FieldSpec("claimant_id", "str", required=True, version_introduced=1),
            FieldSpec("incident_id", "str", required=True, version_introduced=2,
                      description="Incident reference (replaces incident_date + description)"),
            FieldSpec("amount", "float", required=True, version_introduced=1),
            FieldSpec("line_items", "list", required=True, version_introduced=2,
                      description='List of {"code": str, "cost": float} items'),
        ],
        drift_from_previous=[DriftType.FIELD_RENAME, DriftType.FIELD_ADD,
                              DriftType.FIELD_REMOVE],
    ))

    # Held-out: flow restructure (pre-auth step required)
    reg.register(SchemaVersion(
        module_name="insurance_claim",
        version=99,
        endpoint="/insurance/claims/submit",
        method="POST",
        fields=[
            FieldSpec("pre_auth_token", "str", required=True, version_introduced=99,
                      description="Token from GET /insurance/pre-auth"),
            FieldSpec("claimant_id", "str", required=True, version_introduced=1),
            FieldSpec("incident_id", "str", required=True, version_introduced=2),
            FieldSpec("amount", "float", required=True, version_introduced=1),
        ],
        drift_from_previous=[DriftType.FLOW_RESTRUCTURE],
        pre_step_required=True,
        pre_step_endpoint="/insurance/pre-auth",
    ))

    return reg


# Module-level singleton
REGISTRY: SchemaRegistry = _build_registry()

# Training-time schema versions (v1, v2 only; 99 is held-out for eval)
TRAIN_VERSIONS = [1, 2]
HELD_OUT_VERSION = 99
