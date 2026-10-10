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
pip install "metacouplingllm[search]"             # optional ddgs backend for the DuckDuckGo fallback (the fallback also works without it)
pip install "metacouplingllm[viz]"                # country / ADM1 maps
pip install "metacouplingllm[indicators]"         # pandas-based quantitative indicators
pip install "metacouplingllm[export]"             # python-docx scholar export
pip install "metacouplingllm[all]"                # everything
```

> Each provider adapter (OpenAI, Anthropic, Gemini, Grok) auto-wires its
> **native** web-search backend when `web_search=True` — Google Search
> grounding for Gemini, the `/responses` `web_search` tool for Grok, a web
> search tool for Anthropic, and `web_search` for OpenAI. Custom clients, and
> any adapter whose native search fails or returns nothing, fall back to
> DuckDuckGo.

## Quick Start

### Track 1 — Deterministic core: flow indicators and adjacency databases (Brazil soybean one-liner)

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
cols = ["focal_system_id", "IFS", "PFS", "TFS", "MFE", "IFCI", "PFCI", "TFCI"]
print(summary[cols].round(2))
#   focal_system_id  IFS  PFS  TFS   MFE  IFCI  PFCI  TFCI
# 0          Brazil  0.1  0.2  0.7  0.73   1.0  0.25  0.33
```

`summary` has 18 columns: `focal_system_id`, the flow totals `F_I`, `F_P`,
`F_T` and `F_total`, the shares `IFS` / `PFS` / `TFS`, `MFE`,
`IFCI` / `PFCI` / `TFCI`, the partner counts `n_I` / `n_P` / `n_T`, and
`ENP_I` / `ENP_P` / `ENP_T`. Instead of writing the adjacency table by hand,
`build_adjacency` can fill its `adjacent` column from the bundled ADM0 or ADM1
databases.

The optional LLM helpers (each takes a keyword-only `llm_client=`) wrap the
same workflow: `define_study(description)` turns a plain-language description
into a study configuration; `check_inputs(data_summary, sample_rows)`
validates a summary of your table (column names, row count) and a few sample
rows; `interpret_results(summary)` explains the indicator table;
`write_methods(indicator_spec)` drafts Methods text. Each returns
`(result, LLMTrace)`.

### Track 2 — Qualitative LLM analysis (with scholar export)

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
    web_search_max_results=10,
    web_structured_extraction=True,    # validated countries + flows for maps
    auto_map=True,
    rag_corpus=JOURNAL_ARTICLES_2025,
    rag_top_k=8,
)

result = advisor.analyze("My research examines Brazil soybean exports to China.")
print(result.formatted)             # full report with [Tk:N] / [Tk:Wn] citations
print(result.abstract)              # one-paragraph scholar abstract

result.to_markdown("brazil_soybean.md")   # manuscript-ready Markdown
result.to_docx("brazil_soybean.docx")     # Word document with headings
if result.map:
    result.map.savefig("map.png", dpi=150, bbox_inches="tight")
```

Each `analyze()` / `refine()` also writes an inspectable trace folder under
`runs/` in the current directory and sets `result.trace`; pass `trace=False`
to `MetacouplingAssistant(...)` (or set `METACOUPLINGLLM_DISABLE_TRACE=1`) to
turn it off.

## Core Capabilities

### Qualitative LLM analysis
- Structured metacoupling analyses from free-text research descriptions
- Multi-turn refinement across systems, flows, agents, causes, and effects
- **Four LLM providers natively supported**: OpenAI, Anthropic, Google
  Gemini, and xAI Grok (plus a custom-client protocol for anything else),
  each with auto-wired web search and strict structured output
- Literature recommendations from a curated telecoupling/metacoupling database
- Retrieval-augmented generation (RAG, enabled with
  `rag_corpus=JOURNAL_ARTICLES_2025`) over a bundled corpus of 420
  telecoupling/metacoupling studies (192 full text, 228 structured
  summaries; BGE-base embeddings with a TF-IDF fallback); passages are cited
  inline as `[Tk:N]` and web sources as `[Tk:Wn]`, and
  `coupling_analysis=False` gives a RAG-only literature Q&A mode
- Optional web-search grounding with native backends per provider — plus
  `evidence_coverage_note` summaries
- Built-in run tracing (on by default): every `analyze()` / `refine()`
  writes an inspectable folder under `runs/` and attaches `result.trace`;
  disable with the `trace=False` constructor argument or
  `METACOUPLINGLLM_DISABLE_TRACE=1`
- Country-level and ADM1 pericoupling validation, with dual rendering when
  both apply; supranational unions (EU / ASEAN / USMCA) handled
  via member-state dissolution
- National (ADM0) and subnational (ADM1) adjacency databases
  (`lookup_pericoupling`, `is_pericoupled`, `get_pericoupled_neighbors`,
  `lookup_adm1_pericoupling`, ...) with two policy options:
  `de_facto_borders` (disputed areas; default `True`) and `coupling_standard`
  (water-only borders: `"stringent"`, `"moderate"` (default) or `"lenient"`),
  also accepted by `MetacouplingAssistant` for its validation
- Automated map generation at both country and subnational (ADM1) levels
  (`plot_focal_country_map`, `plot_analysis_map`, `plot_focal_adm1_map`)
- Scholar-ready Markdown + Word export (`result.abstract`,
  `result.to_markdown()`, `result.to_docx()`)

### Quantitative indicators
- `classify_coupling`, `build_adjacency` — label each flow intra- (I), peri-
  (P) or telecoupled (T) from your own adjacency table, or from one filled in
  from the bundled ADM0 / ADM1 databases
- `compute_flow_shares` — Metacoupled Flow Shares (MFS): IFS / PFS / TFS per focal system
- `compute_mfe` — Metacoupled Flow Evenness (MFE): normalised Shannon entropy across coupling types
- `compute_mfci` — Metacoupled Flow Concentration Index (MFCI): normalised HHI
  within each coupling type, with the Equivalent Number of Partners
  (`effective_n_partners`), as a long table (one row per focal system and
  coupling type); `summarize_metacoupling` pivots it into the wide
  IFCI / PFCI / TFCI and ENP_I / ENP_P / ENP_T columns
- `summarize_metacoupling` — one-shot combined indicator table

### Optional LLM-assisted helpers
- `define_study`, `check_inputs`, `classify_ambiguous_edges`,
  `interpret_results`, `write_methods` — each returns
  `(result, LLMTrace)` for reproducibility; integrated into
  `classify_coupling(llm_client=...)` for automatic NaN resolution

## Documentation

- [Introduction](INTRODUCTION.md): package overview, architecture, and examples
- [Manual](MANUAL.md): detailed usage guidance
- [Adjacency methods](docs/METHODS_adjacency.md): construction rules and sensitivity analyses behind the adjacency databases
- [Data provenance](src/metacouplingllm/data/PROVENANCE.md): sources, method, and known limitations of the bundled adjacency datasets

## License

MIT. See [LICENSE](LICENSE).
