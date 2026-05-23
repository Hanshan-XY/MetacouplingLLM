"""JSON schemas for the LLM-assisted helpers (PR #36).

Three schemas used with the strict-output dispatch in
``indicators.llm._call_with_strict_json``:

- ``STUDY_CONFIG_SCHEMA`` for ``define_study()``
- ``INPUT_CHECK_SCHEMA`` for ``check_inputs()``
- ``EDGE_CLASSIFICATIONS_SCHEMA`` for ``classify_ambiguous_edges()``

Each schema is also reused as the Anthropic submit-tool input
schema (with strict=True at the tool level) so the same JSON
shape works across all four adapters.
"""

from __future__ import annotations


# ---------------------------------------------------------------------------
# define_study() -- structured study configuration
# ---------------------------------------------------------------------------

STUDY_CONFIG_SCHEMA: dict[str, object] = {
    "type": "object",
    "additionalProperties": False,
    "properties": {
        "focal_system": {
            "type": "string",
            "description": "Primary focal system ID (e.g., 'Brazil').",
        },
        "flow_type": {
            "type": "string",
            "description": (
                "What flows (e.g., 'soybean trade', 'water', 'migration')."
            ),
        },
        "flow_unit": {
            "type": "string",
            "description": "Unit of the flow values (e.g., 'tons').",
        },
        "intracoupling_rule": {"type": "string"},
        "pericoupling_rule": {"type": "string"},
        "telecoupling_rule": {"type": "string"},
        "recommended_input_level": {
            "type": "string",
            "description": (
                "Either 'directed weighted edge list' or 'aggregated totals'."
            ),
        },
        "required_columns": {
            "type": "array",
            "items": {"type": "string"},
        },
        "warnings": {
            "type": "array",
            "items": {"type": "string"},
        },
    },
    "required": [
        "focal_system",
        "flow_type",
        "flow_unit",
        "intracoupling_rule",
        "pericoupling_rule",
        "telecoupling_rule",
        "recommended_input_level",
        "required_columns",
        "warnings",
    ],
}


# ---------------------------------------------------------------------------
# check_inputs() -- data-sufficiency check
# ---------------------------------------------------------------------------

INPUT_CHECK_SCHEMA: dict[str, object] = {
    "type": "object",
    "additionalProperties": False,
    "properties": {
        "can_compute_flow_shares": {"type": "boolean"},
        "can_compute_mfe": {"type": "boolean"},
        "can_compute_mfci": {"type": "boolean"},
        "missing_information": {
            "type": "array",
            "items": {"type": "string"},
        },
        "warnings": {
            "type": "array",
            "items": {"type": "string"},
        },
    },
    "required": [
        "can_compute_flow_shares",
        "can_compute_mfe",
        "can_compute_mfci",
        "missing_information",
        "warnings",
    ],
}


# ---------------------------------------------------------------------------
# classify_ambiguous_edges() -- per-edge classification array
# ---------------------------------------------------------------------------

EDGE_CLASSIFICATIONS_SCHEMA: dict[str, object] = {
    "type": "object",
    "additionalProperties": False,
    "properties": {
        "classifications": {
            "type": "array",
            "items": {
                "type": "object",
                "additionalProperties": False,
                "properties": {
                    "origin": {"type": "string"},
                    "destination": {"type": "string"},
                    "suggested_coupling_type": {
                        "type": "string",
                        "enum": ["I", "P", "T", "unknown"],
                    },
                    "confidence": {
                        "type": "string",
                        "enum": ["low", "medium", "high"],
                    },
                    "reason": {"type": "string"},
                    "needs_user_confirmation": {"type": "boolean"},
                },
                "required": [
                    "origin",
                    "destination",
                    "suggested_coupling_type",
                    "confidence",
                    "reason",
                    "needs_user_confirmation",
                ],
            },
        }
    },
    "required": ["classifications"],
}
