# MetacouplingLLM

`metacouplingllm` helps researchers apply the telecoupling and metacoupling
frameworks to new research topics with LLMs **and** compute quantitative
metacoupling indicators on their own flow data. It can structure analyses
from plain-language study descriptions, recommend relevant literature,
validate pericoupling relationships, ground outputs with web results,
generate country- or ADM1-level maps, export scholar-ready Markdown and Word
documents, and compute deterministic flow-share / evenness / concentration
indicators with optional LLM-assisted study setup and interpretation.

## Installation

Choose the smallest install that matches your workflow:

```bash
pip install metacouplingllm                       # core (no LLM provider)
pip install "metacouplingllm[openai]"             # OpenAI / Grok adapters
pip install "metacouplingllm[anthropic]"          # Anthropic (Claude) adapter
pip install "metacouplingllm[gemini]"             # Google Gemini adapter
pip install "metacouplingllm[grok]"               # xAI Grok adapter
pip install "metacouplingllm[search]"             # DuckDuckGo web search fallback
pip install "metacouplingllm[viz]"                # country / ADM1 maps
pip install "metacouplingllm[indicators]"         # pandas-based quantitative indicators (PR #35)
pip install "metacouplingllm[export]"             # python-docx scholar export (PR #31, #32)
pip install "metacouplingllm[all]"                # everything
```

> Each provider adapter (OpenAI, Anthropic, Gemini, Grok) auto-wires its
> **native** web-search backend when `web_search=True` — Google Search
> grounding for Gemini, Live Search for Grok, web search tool for Anthropic,
> and `web_search` for OpenAI. Custom clients fall back to DuckDuckGo.

## Quick Start

### Track 1 — Qualitative LLM analysis (with scholar export)

```python
from openai import OpenAI
from metacouplingllm import (
    JOURNAL_ARTICLES_2025,
    MetacouplingAssistant,
    OpenAIAdapter,
)

client = OpenAI(api_key="your-key")
advisor = MetacouplingAssistant(
    OpenAIAdapter(client, model="gpt-4o"),
    web_search=True,
    web_search_max_results=5,
    web_structured_extraction=True,    # validated countries + flows for maps
    auto_map=True,
    rag_corpus=JOURNAL_ARTICLES_2025,
    rag_top_k=10,
    rag_min_score=0.15,
)

result = advisor.analyze("My research examines Brazil soybean exports to China.")
print(result.formatted)             # full report with [Tk:N] / [Tk:Wn] citations
print(result.abstract)              # one-paragraph scholar abstract (PR #31)

result.to_markdown("brazil_soybean.md")   # PR #31 — manuscript-ready Markdown
result.to_docx("brazil_soybean.docx")     # PR #31 / #32 — Word document with headings
if result.map:
    result.map.savefig("map.png", dpi=150, bbox_inches="tight")
```

### Track 2 — Quantitative indicators (Brazil soybean one-liner)

```python
import pandas as pd
from metacouplingllm.indicators import (
    classify_coupling, summarize_metacoupling,
)

edges = pd.DataFrame({
    "focal_system_id":   ["Brazil"] * 6,
    "origin_id":         ["Brazil", "Brazil", "Brazil", "Brazil", "Brazil", "Brazil"],
    "destination_id":    ["Brazil", "Argentina", "Paraguay", "China", "EU", "USA"],
    "flow_value":        [10.0, 5.0, 15.0, 50.0, 12.0, 8.0],   # million tonnes
})
adjacency = pd.DataFrame({
    "origin_id":      ["Brazil", "Brazil"],
    "destination_id": ["Argentina", "Paraguay"],
    "adjacent":       [1, 1],
})

classified = classify_coupling(edges, focal_id="Brazil", adjacency=adjacency)
summary    = summarize_metacoupling(classified)
print(summary)
#    focal_system_id   IFS   PFS   TFS   MFE   IFCI   PFCI   TFCI
# 0          Brazil  0.10  0.20  0.70  0.73   1.00   0.25   0.33
```

The same DataFrame plugs into the optional LLM helpers (PR #36) —
`define_study()`, `check_inputs()`, `interpret_results()`,
`write_methods()` — for natural-language study setup, validation, and
manuscript prose around the deterministic numbers.

## Core Capabilities

### Qualitative LLM analysis
- Structured metacoupling analyses from free-text research descriptions
- Multi-turn refinement across systems, flows, agents, causes, and effects
- Literature recommendations from a curated telecoupling/metacoupling database
- Optional web-search grounding with native backends for OpenAI, Anthropic,
  Gemini, and Grok — plus `evidence_coverage_note` summaries (PR #20)
- Country-level and ADM1 pericoupling validation, with dual rendering when
  both apply (PR #27); supranational unions (EU / ASEAN / USMCA) handled
  via member-state dissolution (PR #22-#23)

### Quantitative indicators (PR #35, new)
- `compute_flow_shares` — IFS / PFS / TFS per focal system
- `compute_mfe` — normalised Shannon entropy across coupling types
- `compute_mfci` — normalised HHI (IFCI / PFCI / TFCI) within each coupling
  type, plus the equivalent-number-of-partners (ENP)
- `summarize_metacoupling` — one-shot combined indicator table

### Optional LLM-assisted helpers (PR #36, new)
- `define_study`, `check_inputs`, `classify_ambiguous_edges`,
  `interpret_results`, `write_methods` — each returns
  `(result, LLMTrace)` for reproducibility; integrated into
  `classify_coupling(llm_client=...)` for automatic NaN resolution
- Scholar-ready Markdown + Word export with abstract, citations, and
  per-category section breakdown (PR #31, #32)

## Documentation

- [Introduction](INTRODUCTION.md): package overview, architecture, and examples
- [Manual](MANUAL.md): detailed usage guidance

## License

MIT. See [LICENSE](LICENSE).
