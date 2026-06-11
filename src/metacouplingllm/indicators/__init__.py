"""Quantitative metacoupling indicators (PR #35).

Three indicator families documented in
``docs/metacoupling_codex_prompts_and_formulas_with_llm.md``:

1. **Metacoupled Flow Shares** (``compute_flow_shares``) -- IFS,
   PFS, TFS.  Relative size of intra-, peri-, and telecoupled flows.
2. **Metacoupled Flow Evenness** (``compute_mfe``) -- normalised
   Shannon entropy across the three coupling-type shares.
3. **Metacoupled Flow Concentration Index** (``compute_mfci``) --
   normalised HHI within each coupling type (IFCI, PFCI, TFCI).

Plus two utilities:

- ``classify_coupling`` -- assign I/P/T to each edge based on
  user-supplied adjacency
- ``summarize_metacoupling`` -- one-shot combined indicator table
  (per spec §12.5)

All five functions take a ``pandas.DataFrame`` and return a
``pandas.DataFrame``.  Pandas is an OPTIONAL dependency; install
with ``pip install metacouplingllm[indicators]``.

Design principles:

- **Deterministic calculations**: no LLM calls inside the indicator
  math.  LLM helpers (PR #36) live in a separate module and are
  optional.
- **Established statistics, not invented indices**: Shannon (1948)
  entropy; Hirschman (1945) HHI, normalised per Hannah & Kay (1977);
  Equivalent Number of Partners per Laakso & Taagepera (1979).
- **User supplies adjacency**: no hardcoded geography (spec §4).
- **Intracoupling data required**: package warns when ``F_I = 0``
  so users don't misread missing data as "no intracoupling".
"""

from __future__ import annotations

from metacouplingllm.indicators.classify import classify_coupling
from metacouplingllm.indicators.core import (
    compute_flow_shares,
    compute_mfci,
    compute_mfe,
    summarize_metacoupling,
)
# PR #36: optional LLM-assisted helpers.  Re-exported here so
# users can `from metacouplingllm.indicators import define_study`
# instead of digging into the `llm` submodule path.  Each helper
# takes any LLMClient (OpenAIAdapter / AnthropicAdapter /
# GeminiAdapter / GrokAdapter) and returns a (result, trace)
# tuple.  The trace is an LLMTrace dataclass with prompt /
# response / timestamp / model for reproducibility per spec §15.4.
from metacouplingllm.indicators.llm import (
    LLMTrace,
    check_inputs,
    classify_ambiguous_edges,
    define_study,
    interpret_results,
    write_methods,
)

__all__ = [
    # PR #35 deterministic core
    "classify_coupling",
    "compute_flow_shares",
    "compute_mfe",
    "compute_mfci",
    "summarize_metacoupling",
    # PR #36 LLM-assisted helpers
    "LLMTrace",
    "define_study",
    "check_inputs",
    "classify_ambiguous_edges",
    "interpret_results",
    "write_methods",
]
