"""Test helpers for adapting pre-v0.1.0 fixture style to the current
``ParsedAnalysis`` / ``CouplingSection`` shape.

Background
----------
Before v0.1.0, ``ParsedAnalysis`` carried flat top-level fields
(``systems``, ``flows``, ``agents``, ``causes``, ``effects``,
``suggestions``).  In the v0.1.0 cleanup the parser was restructured:
those fields moved into a per-coupling-type ``CouplingSection`` object,
and accessor methods (``iter_system_entries``, ``iter_flow_entries``,
``iter_category_items``) replaced direct attribute reads.  The
production code was migrated; many test fixtures and assertions were
not, leaving ~130 tests red since the very first git commit.

Rather than rewrite every test in place, this module provides:

- :func:`make_parsed_analysis` — constructor that accepts legacy
  keyword arguments and emits a real ``ParsedAnalysis`` with the new
  ``CouplingSection`` shape.
- :func:`legacy_systems`, :func:`legacy_flows`, :func:`legacy_agents`,
  :func:`legacy_causes`, :func:`legacy_effects`,
  :func:`legacy_suggestions` — read-side adapters that materialize the
  legacy flat shape from the new structured one, so test assertions
  like ``len(result.systems)`` keep working as
  ``len(legacy_systems(result))``.

These helpers exist solely to keep the legacy test surface alive while
the production API moves on.  New tests should prefer the actual
accessor methods (``pa.iter_flow_entries()`` etc.) over these adapters.
"""

from __future__ import annotations

from typing import Any

from metacouplingllm.llm.parser import CouplingSection, ParsedAnalysis


# ---------------------------------------------------------------------------
# Construction
# ---------------------------------------------------------------------------


def make_parsed_analysis(
    *,
    coupling_classification: str = "",
    systems: dict[str, Any] | list[dict[str, Any]] | None = None,
    flows: list[dict[str, Any]] | None = None,
    agents: list[dict[str, Any]] | None = None,
    causes: dict[str, list[str]] | None = None,
    effects: dict[str, list[str]] | None = None,
    suggestions: list[str] | None = None,
    coupling_type: str = "telecoupling",
    raw_text: str = "",
    **extra: Any,
) -> ParsedAnalysis:
    """Build a ``ParsedAnalysis`` from legacy-style kwargs.

    Parameters
    ----------
    systems:
        Legacy dict-by-role shape (e.g.,
        ``{"sending": "Brazil"}`` or
        ``{"sending": {"name": "Brazil", "geographic_scope": "..."}}``)
        OR a list of system entries already in the new shape.
        Either is normalized to a ``list[dict[str, str]]`` with a
        ``role`` key on each entry.
    flows, agents:
        Legacy lists of dicts. Passed straight through.
    causes, effects:
        Legacy dict-by-category shape. Passed straight through.
    suggestions:
        Legacy list. Mapped to ``research_gaps`` on the resulting
        ``ParsedAnalysis``.
    coupling_type:
        Which ``CouplingSection`` to populate on the resulting
        ``ParsedAnalysis``. Defaults to ``"telecoupling"`` since most
        legacy fixtures are telecoupling examples.
    raw_text, **extra:
        Forwarded to ``ParsedAnalysis(...)``.
    """
    systems_list = _normalize_systems(systems)

    section = CouplingSection(
        systems=systems_list,
        flows=list(flows or []),
        agents=list(agents or []),
        causes=dict(causes or {}),
        effects=dict(effects or {}),
    )

    pa_kwargs: dict[str, Any] = {
        "coupling_classification": coupling_classification,
        "raw_text": raw_text,
        coupling_type: section,
    }
    if suggestions is not None:
        pa_kwargs["research_gaps"] = list(suggestions)
    pa_kwargs.update(extra)
    return ParsedAnalysis(**pa_kwargs)


def _normalize_systems(
    systems: dict[str, Any] | list[dict[str, Any]] | None,
) -> list[dict[str, str]]:
    """Convert legacy systems input to the new ``list[dict]`` shape."""
    if systems is None:
        return []
    if isinstance(systems, list):
        # Already a list — assume each entry is a dict already in the
        # new shape (may or may not carry a ``role`` key).
        return [dict(entry) for entry in systems]
    if isinstance(systems, dict):
        out: list[dict[str, str]] = []
        for role, value in systems.items():
            if isinstance(value, str):
                out.append({"role": role, "name": value})
            elif isinstance(value, dict):
                out.append({"role": role, **value})
            else:
                raise TypeError(
                    f"Unsupported system value for role {role!r}: "
                    f"{type(value).__name__}"
                )
        return out
    raise TypeError(f"Unsupported systems type: {type(systems).__name__}")


# ---------------------------------------------------------------------------
# Read-side adapters — materialize legacy flat shape from new structure
# ---------------------------------------------------------------------------


def legacy_systems(pa: ParsedAnalysis) -> dict[str, Any]:
    """Return systems as the legacy dict-by-role shape.

    Each role's value is either a flat string (if the entry only had
    a ``name`` field besides its role) or a dict of remaining fields.
    """
    out: dict[str, Any] = {}
    for entry in pa.iter_system_entries():
        role = entry.get("role", "")
        rest = {k: v for k, v in entry.items() if k != "role"}
        if set(rest.keys()) == {"name"}:
            out[role] = rest["name"]
        else:
            out[role] = rest
    return out


def legacy_flows(pa: ParsedAnalysis) -> list[dict[str, str]]:
    """Return flows as a flat list of dicts (legacy shape)."""
    return list(pa.iter_flow_entries())


def legacy_agents(pa: ParsedAnalysis) -> list[dict[str, str]]:
    """Return agents as a flat list of dicts (legacy shape)."""
    return list(pa.iter_agent_entries())


def legacy_causes(pa: ParsedAnalysis) -> dict[str, list[str]]:
    """Return causes as a dict-by-category (legacy shape)."""
    out: dict[str, list[str]] = {}
    for _section, category, item in pa.iter_category_items("causes"):
        out.setdefault(category, []).append(item)
    return out


def legacy_effects(pa: ParsedAnalysis) -> dict[str, list[str]]:
    """Return effects as a dict-by-category (legacy shape)."""
    out: dict[str, list[str]] = {}
    for _section, category, item in pa.iter_category_items("effects"):
        out.setdefault(category, []).append(item)
    return out


def legacy_suggestions(pa: ParsedAnalysis) -> list[str]:
    """Return ``research_gaps`` (the v0.1.0 rename of legacy
    ``suggestions``)."""
    return list(pa.research_gaps)
