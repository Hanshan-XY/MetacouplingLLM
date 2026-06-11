"""
Metacoupling — Apply telecoupling and metacoupling frameworks using LLMs.

This package helps researchers identify and analyze telecoupling,
pericoupling, and intracoupling processes in their studies using the
frameworks proposed by Jianguo Liu and colleagues.

Quick start
-----------
>>> from openai import OpenAI
>>> from metacouplingllm import MetacouplingAssistant, OpenAIAdapter
>>>
>>> client = OpenAI(api_key="...")
>>> advisor = MetacouplingAssistant(OpenAIAdapter(client, model="gpt-4o"))
>>>
>>> result = advisor.analyze("My research examines international coffee trade...")
>>> print(result.formatted)
>>>
>>> result2 = advisor.refine("Can you elaborate on spillover systems?")
>>> print(result2.formatted)
"""

from metacouplingllm.core import (
    AnalysisResult,
    JOURNAL_ARTICLES_2025,
    MetacouplingAssistant,
    RAGResult,
)
from metacouplingllm.knowledge.countries import (
    get_country_name,
    resolve_country_code,
)
from metacouplingllm.knowledge.framework import (
    AgentLevel,
    CauseCategory,
    CouplingType,
    EffectCategory,
    FlowCategory,
    FrameworkComponent,
    SystemRole,
)
from metacouplingllm.knowledge.literature import (
    Paper,
    format_recommendations,
    get_database_info,
    recommend_papers,
)
from metacouplingllm.knowledge.adm1_pericoupling import (
    Adm1PairType,
    Adm1PericouplingResult,
    get_adm1_codes_for_country,
    get_adm1_country,
    get_adm1_info,
    get_adm1_neighbors,
    get_cross_border_neighbors,
    is_adm1_pericoupled,
    lookup_adm1_pericoupling,
    resolve_adm1_code,
)
from metacouplingllm.knowledge.citations import (
    TURN_CITATION_PATTERN,
    extract_turn_cited_ids,
    sanitize_turn_citations,
)
from metacouplingllm.knowledge.rag import (
    RAGEngine,
    RetrievalResult,
    TextChunk,
    format_evidence,
)
from metacouplingllm.knowledge.websearch import (
    AnthropicWebSearchBackend,
    GeminiWebSearchBackend,
    GrokWebSearchBackend,
    OpenAIWebSearchBackend,
    annotate_web_citations,
    extract_web_map_signals,
    format_web_context,
    format_web_map_signals_context,
    search_web,
)
from metacouplingllm.knowledge.pericoupling import (
    PairCouplingType,
    PericouplingResult,
    get_pericoupled_neighbors,
    is_pericoupled,
    lookup_pericoupling,
)
from metacouplingllm.llm.client import (
    AnthropicAdapter,
    GeminiAdapter,
    GrokAdapter,
    LLMClient,
    LLMResponse,
    Message,
    OpenAIAdapter,
)

def _resolve_version() -> str:
    # Installed distribution metadata is authoritative when available.
    try:
        from importlib.metadata import PackageNotFoundError, version

        return version("MetacouplingLLM")
    except PackageNotFoundError:
        pass
    # Source-tree import (running from src/ without pip-installing):
    # read the version from pyproject.toml so it cannot drift.
    import re
    from pathlib import Path

    pyproject = Path(__file__).resolve().parents[2] / "pyproject.toml"
    try:
        match = re.search(
            r'^version\s*=\s*"([^"]+)"',
            pyproject.read_text(encoding="utf-8"),
            flags=re.MULTILINE,
        )
    except OSError:
        match = None
    return match.group(1) if match else "0.1.3"


__version__ = _resolve_version()


# ---------------------------------------------------------------------------
# Quantitative indicators (requires metacouplingllm[indicators]).
#
# PR #35: ``metacouplingllm.indicators`` submodule, lazy-imported so
# users who don't pip install [indicators] (pandas) still get the
# package working for the LLM-driven qualitative analysis path.
# Access the indicator functions via:
#
#     from metacouplingllm import indicators
#     indicators.compute_flow_shares(df, ...)
#
# We deliberately do NOT re-export the indicator functions into the
# top-level namespace: this keeps the top namespace focused on the
# LLM analysis API and signals that the quantitative side is a
# separate, opt-in part of the package.
# ---------------------------------------------------------------------------

from metacouplingllm import indicators  # noqa: E402


# ---------------------------------------------------------------------------
# Lazy-import wrappers for visualization (requires metacoupling[viz])
# ---------------------------------------------------------------------------


def plot_focal_country_map(focal_country, **kwargs):
    """Plot a world map by coupling type relative to a focal country.

    Requires: ``pip install metacoupling[viz]``
    """
    from metacouplingllm.visualization.worldmap import (
        plot_focal_country_map as _f,
    )

    return _f(focal_country, **kwargs)


def plot_analysis_map(parsed_analysis, **kwargs):
    """Plot a world map from LLM analysis results.

    Requires: ``pip install metacoupling[viz]``
    """
    from metacouplingllm.visualization.worldmap import plot_analysis_map as _f

    return _f(parsed_analysis, **kwargs)


def plot_focal_adm1_map(focal_adm1, **kwargs):
    """Plot a subnational (ADM1) map by coupling type relative to a focal region.

    Requires: ``pip install metacoupling[viz]``
    """
    from metacouplingllm.visualization.adm1_map import plot_focal_adm1_map as _f

    return _f(focal_adm1, **kwargs)


__all__ = [
    # Main interface
    "MetacouplingAssistant",
    "AnalysisResult",
    "RAGResult",
    "JOURNAL_ARTICLES_2025",
    # LLM client
    "LLMClient",
    "Message",
    "LLMResponse",
    "OpenAIAdapter",
    "AnthropicAdapter",
    "GeminiAdapter",
    "GrokAdapter",
    # Enums
    "AgentLevel",
    "CauseCategory",
    "EffectCategory",
    "CouplingType",
    "SystemRole",
    "FlowCategory",
    "FrameworkComponent",
    # Country-level pericoupling
    "PairCouplingType",
    "PericouplingResult",
    "lookup_pericoupling",
    "is_pericoupled",
    "get_pericoupled_neighbors",
    "resolve_country_code",
    "get_country_name",
    # ADM1 (subnational) pericoupling
    "Adm1PairType",
    "Adm1PericouplingResult",
    "lookup_adm1_pericoupling",
    "is_adm1_pericoupled",
    "get_adm1_neighbors",
    "get_cross_border_neighbors",
    "get_adm1_codes_for_country",
    "get_adm1_info",
    "get_adm1_country",
    "resolve_adm1_code",
    # Literature recommendation
    "recommend_papers",
    "format_recommendations",
    "get_database_info",
    "Paper",
    # RAG (full-text evidence retrieval)
    "RAGEngine",
    "RetrievalResult",
    "TextChunk",
    "format_evidence",
    # Citation sanitization (turn-scoped, pre-retrieval RAG)
    "TURN_CITATION_PATTERN",
    "sanitize_turn_citations",
    "extract_turn_cited_ids",
    # Web search (requires metacoupling[search])
    "search_web",
    "AnthropicWebSearchBackend",
    "GeminiWebSearchBackend",
    "GrokWebSearchBackend",
    "OpenAIWebSearchBackend",
    "format_web_context",
    "annotate_web_citations",
    "extract_web_map_signals",
    "format_web_map_signals_context",
    # Visualization (requires metacoupling[viz])
    "plot_focal_country_map",
    "plot_analysis_map",
    "plot_focal_adm1_map",
]
