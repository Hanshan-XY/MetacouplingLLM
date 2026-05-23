"""Optional LLM-assisted helpers for the indicators submodule (PR #36).

Five helpers + one shared adapter-dispatch routine + one
``LLMTrace`` dataclass for reproducibility.  Each helper takes any
``LLMClient`` (OpenAI / Anthropic / Gemini / Grok adapters from
``metacouplingllm.llm.client``) and returns a
``(result, trace)`` tuple per spec §15.4 reproducibility rule.

Strict-output dispatch reuses the pattern from
``websearch.py::extract_web_map_signals`` (PR #28 + PR #30):

- OpenAI:    ``response_format = {"type": "json_schema", ...}``
- Anthropic: ``tools = [submit_tool]`` + ``tool_choice``
- Gemini:    ``response_schema`` + ``response_mime_type``
- Grok:      ``response_format = {"type": "json_schema", ...}``

All five helpers refuse to invent numerical flow values per the
``_GUARDRAILS`` block in ``_prompts.py``.  The two prose-output
helpers (``interpret_results``, ``write_methods``) explicitly
forbid recalculating numeric values.

Pandas is imported lazily; users without
``pip install metacouplingllm[indicators]`` get the same clean
ImportError as the rest of the submodule.
"""

from __future__ import annotations

import datetime
import json
import warnings
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from metacouplingllm.indicators._prompts import (
    CHECK_INPUTS_PROMPT_VERSION,
    CHECK_INPUTS_SYSTEM,
    CHECK_INPUTS_USER,
    CLASSIFY_EDGES_PROMPT_VERSION,
    CLASSIFY_EDGES_SYSTEM,
    CLASSIFY_EDGES_USER,
    DEFINE_STUDY_PROMPT_VERSION,
    DEFINE_STUDY_SYSTEM,
    DEFINE_STUDY_USER,
    INTERPRET_RESULTS_PROMPT_VERSION,
    INTERPRET_RESULTS_SYSTEM,
    INTERPRET_RESULTS_USER,
    WRITE_METHODS_PROMPT_VERSION,
    WRITE_METHODS_SYSTEM,
    WRITE_METHODS_USER,
    _INTERPRET_AUDIENCES,
)
from metacouplingllm.indicators._schemas import (
    EDGE_CLASSIFICATIONS_SCHEMA,
    INPUT_CHECK_SCHEMA,
    STUDY_CONFIG_SCHEMA,
)
from metacouplingllm.llm.client import LLMClient, Message

if TYPE_CHECKING:  # pragma: no cover - type-only
    import pandas as pd


# ---------------------------------------------------------------------------
# Reproducibility: LLMTrace
# ---------------------------------------------------------------------------


@dataclass
class LLMTrace:
    """Per spec §15.4 item 7: everything needed to reproduce an
    LLM-assisted indicator call.  Returned alongside the result
    from every helper so users can save it however they like.

    Attributes
    ----------
    timestamp_utc:
        ISO 8601 timestamp of the call (e.g., ``"2026-05-22T16:30:00Z"``).
    model:
        Model identifier from the adapter, or ``""`` when the
        adapter doesn't expose a ``.model`` attribute.
    prompt_version:
        Constant per helper (e.g., ``"define_study_v1"``).  Bump
        when changing prompt wording so old traces remain
        attributable to the prompt they ran against.
    system_prompt, user_prompt:
        The exact strings sent to the LLM.
    raw_response:
        The full ``response.content`` string returned by the LLM
        (before JSON parsing).
    usage:
        Token-usage dict from the adapter when available
        (e.g., ``{"prompt_tokens": ..., "completion_tokens": ...}``).
        ``None`` when the adapter doesn't surface usage.
    """

    timestamp_utc: str
    model: str
    prompt_version: str
    system_prompt: str
    user_prompt: str
    raw_response: str
    usage: dict[str, int] | None = None


def _now_utc_iso() -> str:
    """ISO 8601 timestamp with Z suffix."""
    return (
        datetime.datetime.now(datetime.timezone.utc)
        .replace(microsecond=0)
        .isoformat()
        .replace("+00:00", "Z")
    )


def _model_of(llm_client: Any) -> str:
    """Best-effort extraction of the model name from an adapter."""
    model = getattr(llm_client, "model", None) or getattr(
        llm_client, "_model", None,
    )
    return str(model) if model else ""


# ---------------------------------------------------------------------------
# Strict-output dispatch (adapter-aware)
# ---------------------------------------------------------------------------


def _build_submit_tool(name: str, schema: dict, description: str) -> dict:
    """Anthropic submit-tool wrapper matching the
    `_ANTHROPIC_WEB_MAP_SIGNALS_SUBMIT_TOOL` shape from
    websearch.py:2420.  Used by `_call_with_strict_json` when the
    adapter is AnthropicAdapter."""
    return {
        "name": name,
        "description": description,
        "input_schema": schema,
        "strict": True,
    }


def _call_with_strict_json(
    llm_client: LLMClient,
    *,
    system_prompt: str,
    user_prompt: str,
    schema: dict,
    schema_name: str,
    prompt_version: str,
    model: str | None = None,
) -> tuple[dict, LLMTrace]:
    """Adapter-dispatched strict-JSON call.

    Returns ``(parsed_dict, LLMTrace)``.  Falls back to
    ``_extract_json_object`` parsing when the adapter-specific
    strict path returns something that doesn't deserialize.
    Raises ``RuntimeError`` only when both strict and fallback
    paths fail to produce a dict.
    """
    # Lazy import to avoid the indicators submodule pulling
    # websearch.py / fastembed at load time.
    from metacouplingllm.knowledge.websearch import _extract_json_object
    from metacouplingllm.llm.client import (
        AnthropicAdapter,
        GeminiAdapter,
        GrokAdapter,
        OpenAIAdapter,
    )

    messages = [
        Message(role="system", content=system_prompt),
        Message(role="user", content=user_prompt),
    ]
    chat_kwargs: dict[str, Any] = {}
    used_tool_path = False
    tool_name = f"submit_{schema_name}"
    submit_tool = _build_submit_tool(
        tool_name, schema,
        description=(
            f"Submit the structured {schema_name} object.  Call this "
            "tool exactly once as the final action of your turn.  All "
            "fields are required."
        ),
    )

    # Dispatch identical to websearch.py::extract_web_map_signals
    # (lines 2622-2680).  Each adapter has a different mechanism
    # for enforcing strict structured JSON output.
    if isinstance(llm_client, OpenAIAdapter):
        chat_kwargs["response_format"] = {
            "type": "json_schema",
            "json_schema": {
                "name": schema_name,
                "strict": True,
                "schema": schema,
            },
        }
    elif isinstance(llm_client, AnthropicAdapter):
        chat_kwargs["tools"] = [submit_tool]
        chat_kwargs["tool_choice"] = {
            "type": "tool", "name": tool_name,
        }
        used_tool_path = True
    elif isinstance(llm_client, GeminiAdapter):
        chat_kwargs["response_schema"] = schema
        chat_kwargs["response_mime_type"] = "application/json"
    elif isinstance(llm_client, GrokAdapter):
        chat_kwargs["response_format"] = {
            "type": "json_schema",
            "json_schema": {
                "name": schema_name,
                "strict": True,
                "schema": schema,
            },
        }
    # else: unknown adapter -- send plain chat + parse from prose.

    if model is not None:
        # Best-effort model override: only supported by adapters
        # that expose a `model` kwarg on chat().  For now, attach
        # to the kwargs and trust the adapter to ignore unknowns
        # (most do).  Users who need strict model override should
        # construct a new adapter instance.
        chat_kwargs.setdefault("model", model)

    response = llm_client.chat(messages=messages, **chat_kwargs)

    raw_response = response.content or ""
    parsed: dict | None = None

    # Anthropic returns the structured payload in a tool_use block.
    if used_tool_path:
        tool_uses = getattr(response, "tool_uses", None) or []
        for tu in tool_uses:
            if tu.get("name") == tool_name:
                tu_input = tu.get("input")
                if isinstance(tu_input, dict):
                    parsed = tu_input
                    break

    # Fallback: parse JSON from response text.
    if parsed is None and raw_response:
        try:
            parsed = json.loads(raw_response)
        except (ValueError, TypeError):
            parsed = _extract_json_object(raw_response)

    if not isinstance(parsed, dict):
        raise RuntimeError(
            f"LLM helper for {schema_name!r} did not return a "
            f"parseable JSON object.  Raw response (first 200 "
            f"chars): {raw_response[:200]!r}"
        )

    trace = LLMTrace(
        timestamp_utc=_now_utc_iso(),
        model=_model_of(llm_client),
        prompt_version=prompt_version,
        system_prompt=system_prompt,
        user_prompt=user_prompt,
        raw_response=raw_response,
        usage=dict(response.usage) if response.usage else None,
    )
    return parsed, trace


def _call_with_prose(
    llm_client: LLMClient,
    *,
    system_prompt: str,
    user_prompt: str,
    prompt_version: str,
    model: str | None = None,
) -> tuple[str, LLMTrace]:
    """Plain chat call for the prose-output helpers
    (``interpret_results``, ``write_methods``).  No strict-output
    dispatch needed -- we just want the text back."""
    messages = [
        Message(role="system", content=system_prompt),
        Message(role="user", content=user_prompt),
    ]
    chat_kwargs: dict[str, Any] = {}
    if model is not None:
        chat_kwargs.setdefault("model", model)
    response = llm_client.chat(messages=messages, **chat_kwargs)
    text = (response.content or "").strip()
    trace = LLMTrace(
        timestamp_utc=_now_utc_iso(),
        model=_model_of(llm_client),
        prompt_version=prompt_version,
        system_prompt=system_prompt,
        user_prompt=user_prompt,
        raw_response=response.content or "",
        usage=dict(response.usage) if response.usage else None,
    )
    return text, trace


# ---------------------------------------------------------------------------
# Public helpers
# ---------------------------------------------------------------------------


def define_study(
    description: str,
    *,
    llm_client: LLMClient,
    model: str | None = None,
) -> tuple[dict, LLMTrace]:
    """Convert a natural-language study description into a
    structured configuration dict.

    Output keys (per ``STUDY_CONFIG_SCHEMA``):
    ``focal_system``, ``flow_type``, ``flow_unit``,
    ``intracoupling_rule``, ``pericoupling_rule``,
    ``telecoupling_rule``, ``recommended_input_level``,
    ``required_columns``, ``warnings``.
    """
    if not description or not description.strip():
        raise ValueError("define_study: description must not be empty.")
    return _call_with_strict_json(
        llm_client,
        system_prompt=DEFINE_STUDY_SYSTEM,
        user_prompt=DEFINE_STUDY_USER.format(description=description),
        schema=STUDY_CONFIG_SCHEMA,
        schema_name="study_config",
        prompt_version=DEFINE_STUDY_PROMPT_VERSION,
        model=model,
    )


def check_inputs(
    data_summary: dict,
    sample_rows: list[dict] | None = None,
    *,
    llm_client: LLMClient,
    model: str | None = None,
) -> tuple[dict, LLMTrace]:
    """Validate whether the user's tabular data can support the
    three indicator families.

    Parameters
    ----------
    data_summary:
        Dict with at minimum ``columns`` (list of column names) and
        ``row_count`` (int).  Optionally ``has_coupling_col`` (bool).
    sample_rows:
        Optional small sample (≤5 rows recommended) as a list of
        dicts.  When provided, the LLM uses them to spot unit
        mixing, single-self-loop intracoupling, etc.
    """
    if not isinstance(data_summary, dict):
        raise TypeError(
            "check_inputs: data_summary must be a dict with at "
            "least 'columns' and 'row_count' keys."
        )
    columns = data_summary.get("columns") or []
    row_count = data_summary.get("row_count", 0)
    has_coupling_col = bool(data_summary.get("has_coupling_col", False))
    sample_n = len(sample_rows or [])
    sample_block = (
        json.dumps(sample_rows, indent=2) if sample_rows else "(none provided)"
    )
    user_prompt = CHECK_INPUTS_USER.format(
        columns=", ".join(map(str, columns)),
        row_count=row_count,
        has_coupling_col=has_coupling_col,
        sample_n=sample_n,
        sample_rows=sample_block,
    )
    return _call_with_strict_json(
        llm_client,
        system_prompt=CHECK_INPUTS_SYSTEM,
        user_prompt=user_prompt,
        schema=INPUT_CHECK_SCHEMA,
        schema_name="input_check",
        prompt_version=CHECK_INPUTS_PROMPT_VERSION,
        model=model,
    )


def classify_ambiguous_edges(
    edges: "pd.DataFrame",
    study_config: dict,
    *,
    llm_client: LLMClient,
    model: str | None = None,
    origin_col: str = "origin_id",
    destination_col: str = "destination_id",
) -> tuple["pd.DataFrame", LLMTrace]:
    """Classify edges that the deterministic ``classify_coupling``
    pass couldn't resolve.

    Returns a DataFrame that mirrors the input's index plus columns
    ``suggested_coupling_type`` (``"I"`` / ``"P"`` / ``"T"`` /
    ``"unknown"``), ``confidence`` (``"low"`` / ``"medium"`` /
    ``"high"``), ``reason``, and ``needs_user_confirmation``.

    The LLM is instructed to return ``"unknown"`` when information
    is insufficient -- per spec §16 item 3, the package never lets
    the LLM invent adjacency facts.
    """
    try:
        import pandas as pd
    except ImportError as exc:  # pragma: no cover - exercised in tests
        raise ImportError(
            "metacouplingllm.indicators requires pandas. Install with "
            "`pip install metacouplingllm[indicators]`."
        ) from exc

    if not isinstance(edges, pd.DataFrame):
        raise TypeError(
            "classify_ambiguous_edges: edges must be a pandas DataFrame."
        )
    if origin_col not in edges.columns or destination_col not in edges.columns:
        raise KeyError(
            f"edges DataFrame must contain {origin_col!r} and "
            f"{destination_col!r} columns.  Available: "
            f"{list(edges.columns)}."
        )
    if edges.empty:
        # Nothing to classify; return an empty result with the
        # expected columns so callers don't have to special-case it.
        empty = edges.iloc[0:0].copy()
        for col in (
            "suggested_coupling_type", "confidence", "reason",
            "needs_user_confirmation",
        ):
            empty[col] = pd.Series(dtype="object")
        empty_trace = LLMTrace(
            timestamp_utc=_now_utc_iso(),
            model=_model_of(llm_client),
            prompt_version=CLASSIFY_EDGES_PROMPT_VERSION,
            system_prompt=CLASSIFY_EDGES_SYSTEM,
            user_prompt="(skipped: no edges to classify)",
            raw_response="",
            usage=None,
        )
        return empty, empty_trace

    edges_list = "\n".join(
        f"{i+1}. origin={row[origin_col]!r}, destination={row[destination_col]!r}"
        for i, (_, row) in enumerate(edges.iterrows())
    )
    user_prompt = CLASSIFY_EDGES_USER.format(
        study_config_json=json.dumps(study_config, indent=2),
        edges_list=edges_list,
    )
    parsed, trace = _call_with_strict_json(
        llm_client,
        system_prompt=CLASSIFY_EDGES_SYSTEM,
        user_prompt=user_prompt,
        schema=EDGE_CLASSIFICATIONS_SCHEMA,
        schema_name="edge_classifications",
        prompt_version=CLASSIFY_EDGES_PROMPT_VERSION,
        model=model,
    )
    classifications = parsed.get("classifications") or []
    if len(classifications) != len(edges):
        warnings.warn(
            f"classify_ambiguous_edges: LLM returned "
            f"{len(classifications)} classifications for {len(edges)} "
            "edges.  Mismatched rows will be filled with "
            "'unknown' / low confidence.",
            UserWarning,
            stacklevel=2,
        )

    out = edges.copy()
    # Default-fill columns so a short LLM response doesn't break us.
    out["suggested_coupling_type"] = "unknown"
    out["confidence"] = "low"
    out["reason"] = ""
    out["needs_user_confirmation"] = True

    for i, (idx, _) in enumerate(edges.iterrows()):
        if i >= len(classifications):
            break
        entry = classifications[i]
        if not isinstance(entry, dict):
            continue
        out.at[idx, "suggested_coupling_type"] = str(
            entry.get("suggested_coupling_type", "unknown")
        )
        out.at[idx, "confidence"] = str(entry.get("confidence", "low"))
        out.at[idx, "reason"] = str(entry.get("reason", ""))
        out.at[idx, "needs_user_confirmation"] = bool(
            entry.get("needs_user_confirmation", True)
        )
    return out, trace


def interpret_results(
    results: "pd.DataFrame",
    *,
    llm_client: LLMClient,
    audience: str = "academic",
    model: str | None = None,
) -> tuple[str, LLMTrace]:
    """Plain-language interpretation of a computed indicator table.

    Parameters
    ----------
    results:
        Output of ``summarize_metacoupling()`` (or any DataFrame
        with the same shape).
    audience:
        One of ``"academic"`` (default) | ``"general"`` | ``"policy"``.
        Each preset uses a different system prompt style.
    """
    if audience not in _INTERPRET_AUDIENCES:
        raise ValueError(
            f"interpret_results: audience must be one of "
            f"{sorted(_INTERPRET_AUDIENCES)}, got {audience!r}."
        )
    try:
        import pandas as pd
    except ImportError as exc:  # pragma: no cover
        raise ImportError(
            "metacouplingllm.indicators requires pandas. Install with "
            "`pip install metacouplingllm[indicators]`."
        ) from exc
    if not isinstance(results, pd.DataFrame):
        raise TypeError(
            "interpret_results: results must be a pandas DataFrame."
        )

    system_prompt = INTERPRET_RESULTS_SYSTEM.format(
        audience_style=_INTERPRET_AUDIENCES[audience],
    )
    user_prompt = INTERPRET_RESULTS_USER.format(
        results_table=results.to_string(index=False),
    )
    return _call_with_prose(
        llm_client,
        system_prompt=system_prompt,
        user_prompt=user_prompt,
        prompt_version=INTERPRET_RESULTS_PROMPT_VERSION,
        model=model,
    )


def write_methods(
    indicator_spec: dict,
    *,
    llm_client: LLMClient,
    model: str | None = None,
) -> tuple[str, LLMTrace]:
    """Manuscript-ready Methods text describing the indicators.

    ``indicator_spec`` is an arbitrary dict the user provides to
    describe their study (e.g., focal system, flow type, indicators
    used).  The LLM weaves this into formal prose with formulas
    and citations per spec §15.2 example E.
    """
    if not isinstance(indicator_spec, dict):
        raise TypeError(
            "write_methods: indicator_spec must be a dict."
        )
    user_prompt = WRITE_METHODS_USER.format(
        indicator_spec_json=json.dumps(indicator_spec, indent=2),
    )
    return _call_with_prose(
        llm_client,
        system_prompt=WRITE_METHODS_SYSTEM,
        user_prompt=user_prompt,
        prompt_version=WRITE_METHODS_PROMPT_VERSION,
        model=model,
    )
