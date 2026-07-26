# Metacoupling: An LLM-Powered Python Package for Applying the Metacoupling Framework

## 1. Overview

**Metacoupling** is a Python package (v0.1.3) that helps researchers apply the telecoupling and metacoupling frameworks (Liu et al., 2013; Liu, 2017) to their own research topics using Large Language Models (LLMs). Given a natural-language research description, the package produces a structured, framework-compliant analysis that identifies coupled human and natural systems, classifies coupling types, maps flows, agents, causes, and effects, and grounds the output in both a curated literature database and real-time web search results.

**Key capabilities:**

- Structured metacoupling analysis from free-text research descriptions
- Multi-turn refinement via conversational LLM interaction
- Retrieval-Augmented Generation (RAG) over 420 telecoupling/metacoupling papers — 192 indexed as full text, the other 228 as structured paraphrased summaries (used wherever redistributing the full text would not be licence-safe)
- **RAG-only literature Q&A mode** for users already familiar with the framework (`coupling_analysis=False`)
- Literature recommendation from a curated BibTeX database
- Real-time web search grounding with native backends for OpenAI, Anthropic, Gemini, and Grok (DuckDuckGo fallback for custom clients) plus `evidence_coverage_note` self-assessment
- Pericoupling validation against two geographic databases (country-level and subnational ADM1) with mode-aware presentation: region-scale adjacency for subnational queries, country-scale for national queries; spillover pairs filtered (left to the framework systems analysis)
- Automated map generation at both country and subnational levels, including supranational-union dissolution (EU / ASEAN / USMCA)
- Support for OpenAI, Anthropic, **Google Gemini**, **xAI Grok**, or any custom LLM backend (each with native web-search auto-wiring)
- **Scholar-ready export**: `result.abstract`, `result.to_markdown()`, `result.to_docx()` for manuscript-ready output
- **Quantitative metacoupling indicators** (`metacouplingllm.indicators`) — IFS / PFS / TFS, MFE, MFCI computed on user-supplied flow data
- **Optional LLM-assisted indicator helpers** — `define_study`, `check_inputs`, `classify_ambiguous_edges`, `interpret_results`, `write_methods`, each returning `(result, LLMTrace)` for reproducibility
- **Built-in run tracing** (on by default) — every `analyze()`/`refine()` records a `RunTrace` (all model calls, intermediates, token usage, git SHA) and writes an inspectable artifact folder under `runs/`; disable with `trace=False`

**Requirements:** Python 3.10+, two hard runtime dependencies: `numpy>=1.21` and `fastembed>=0.8` (offline RAG embeddings). Optional extras: `openai`, `anthropic`, `gemini`, `grok`, `geopandas`+`matplotlib` (visualization), `ddgs` (web search), `pandas` (quantitative indicators), `python-docx` (Word export).

### Two complementary tracks

The package exposes two complementary analysis tracks that can be used independently or combined:

1. **Qualitative LLM-driven case-study analysis** via `MetacouplingAssistant` — produces a structured, framework-compliant report with citations, optional maps, and scholar-ready Markdown / Word export.
2. **Quantitative deterministic indicators** via the `metacouplingllm.indicators` submodule — computes flow-share / evenness / concentration indicators on user-supplied flow data. Five optional LLM-assisted helpers (`define_study`, `check_inputs`, `classify_ambiguous_edges`, `interpret_results`, `write_methods`) wrap natural-language judgment tasks around the deterministic core.

### Quick Install

```bash
pip install metacouplingllm
pip install "metacouplingllm[openai]"
pip install "metacouplingllm[anthropic]"
pip install "metacouplingllm[gemini]"
pip install "metacouplingllm[grok]"
pip install "metacouplingllm[search]"
pip install "metacouplingllm[viz]"
pip install "metacouplingllm[indicators]"   # PR #35 quantitative indicators (pandas)
pip install "metacouplingllm[export]"       # PR #31/#32 scholar export (python-docx)
pip install "metacouplingllm[all]"
```

If you want to use the built-in OpenAI example below, install `metacouplingllm[openai]`.

### What Users Get

- A structured metacoupling analysis from a plain-language study description
- Optional literature recommendations and supporting evidence passages
- Optional web-grounded context for current trade, policy, or event information, with an `evidence_coverage_note` self-assessment of which sources back which claims
- Optional map generation for country-level and ADM1-level analyses
- Scholar-ready abstract + Markdown + Word export (`result.abstract`, `result.to_markdown()`, `result.to_docx()`)
- Quantitative metacoupling indicators (IFS / PFS / TFS, MFE, MFCI) computed on user-supplied flow data via `metacouplingllm.indicators`
- Optional LLM-assisted helpers for study setup, input validation, ambiguous-edge classification, results interpretation, and methods-text drafting — each producing reproducible `LLMTrace` records

---

## 2. Theoretical Foundations

The package operationalizes the **metacoupling framework** (Liu, 2017), which integrates three types of human-nature interactions across geographic scales:

| Coupling Type | Definition | Example |
|---|---|---|
| **Intracoupling** | Interactions *within* a single coupled human and natural system (CHANS) | Manure management impacts on local water quality in Michigan |
| **Pericoupling** | Interactions *between adjacent* coupled systems | Feed grain trade between Michigan and neighboring Indiana |
| **Telecoupling** | Interactions *between distant* coupled systems | Pork exports from Michigan to Japan |

The framework identifies five core components in each coupling:

1. **Systems** -- Sending, receiving, and spillover systems, each with human and natural subsystems
2. **Flows** -- Movements of matter, capital, energy, information, people, and organisms
3. **Agents** -- Fixed categories: individuals / households; firms / traders / corporations; governments / policymakers; organizations / NGOs; non-human agents
4. **Causes** -- Fixed categories: economic; political / institutional; ecological / biological; technological / infrastructural; cultural / social / demographic; hydrological; climatic / atmospheric; geological / geomorphological
5. **Effects** -- The same fixed categories used for causes, applied to outcomes across coupled systems

The package encodes 14 telecoupling categories from the literature (trade, migration, tourism, species invasion, water transfer, etc.) and the six-phase operationalization procedure from Liu (2017).

---

## 3. Architecture

```
                    Research Description (free text)
                              |
                    +---------v----------+
                    |  MetacouplingAssistant   |
                    +---------+----------+
                              |
          +-------------------+-------------------+
          |                   |                   |
  +-------v-------+  +-------v--------+  +-------v-------+
  | PromptBuilder  |  |  Web Search    |  | RAG Engine    |
  | (6-layer       |  |  (native +     |  | (420 papers,  |
  |  system prompt)|  |   DuckDuckGo)  |  |  embeddings   |
  |                |  |                |  |  or TF-IDF    |
  |                |  |                |  |  fallback)    |
  +-------+-------+  +-------+--------+  +-------+-------+
          |                   |                   |
          +-------------------+-------------------+
                              |
                    +---------v----------+
                    |     LLM Client     |
                    |  (OpenAI / Anthropic /|
                    |   Gemini / Grok /   |
                    |   custom backend)   |
                    +---------+----------+
                              |
                    +---------v----------+
                    |    Parser          |
                    |  (structured       |
                    |   ParsedAnalysis)  |
                    +---------+----------+
                              |
          +-------------------+-------------------+
          |                   |                   |
  +-------v-------+  +-------v--------+  +-------v-------+
  | Coupling       |  | Literature     |  | Map Generator |
  | Validation     |  | Recommendations|  | (ADM0/ADM1)   |
  +---------------+  +----------------+  +---------------+
                              |
                    +---------v----------+
                    |   AnalysisResult   |
                    |  .parsed           |
                    |  .formatted        |
                    |  .abstract         |   (PR #31)
                    |  .to_markdown()    |   (PR #31)
                    |  .to_docx()        |   (PR #31, #32)
                    |  .map              |
                    +--------------------+
```

**Quantitative indicators side-track** (PR #35 + #36):

```
       User-supplied flow DataFrame + adjacency
                       |
            +----------v-----------+
            |  classify_coupling   |  ← assigns I / P / T
            |  (optional LLM       |     to each edge
            |   fallback)          |
            +----------+-----------+
                       |
            +----------v-----------+
            |  compute_flow_shares |  ← IFS / PFS / TFS
            |  compute_mfe         |  ← Shannon evenness
            |  compute_mfci        |  ← normalised HHI
            |  summarize_metacoupling
            +----------+-----------+
                       |
            +----------v-----------+
            |  LLM helpers (PR #36)|  ← define_study,
            |  optional; each      |     check_inputs,
            |  returns (result,    |     classify_ambiguous_edges,
            |   LLMTrace)          |     interpret_results,
            |                      |     write_methods
            +----------------------+
```

Indicator math never calls an LLM (deterministic-first per PR #35
design). The LLM helpers are scoped to natural-language tasks only
(study setup, validation, interpretation, methods drafting) and
always pair their output with an `LLMTrace` for reproducibility.

---

## 4. Core Functions

### 4.1 MetacouplingAssistant

The central class that orchestrates the entire analysis pipeline.

```python
from metacouplingllm import (
    JOURNAL_ARTICLES_2025,
    MetacouplingAssistant,
    OpenAIAdapter,
)
from openai import OpenAI

advisor = MetacouplingAssistant(
    llm_client=OpenAIAdapter(OpenAI(api_key="..."), model="gpt-4o"),
    auto_map=True,              # Generate map automatically
    rag_corpus=JOURNAL_ARTICLES_2025,  # Use bundled 2025 journal corpus
    web_search=True,            # Ground analysis in web search results
    web_search_max_results=10,  # Number of web results (default)
    web_structured_extraction=True,  # Recommended with web_search + auto_map
    rag_top_k=8,                # Number of RAG evidence passages (default)
    rag_min_score=0.60,         # Min embeddings cosine sim (BGE-base scale)
    max_examples=2,             # Framework examples in prompt
    temperature=0.7,            # LLM temperature
    verbose=True,               # Print progress messages
)
```

**Key methods:**

| Method | Purpose |
|---|---|
| `analyze(research_description)` | First-turn structured analysis |
| `refine(info, focus_component=None)` | Multi-turn follow-up refinement |
| `reset()` | Clear conversation for a new topic |

### 4.2 Prompt Engineering (6-Layer System)

The system prompt is constructed in six layers:

1. **Role** -- Expert persona in metacoupling and sustainability science
2. **Knowledge** -- Full framework definitions, 14 categories, coupling transformations
3. **Methodology** -- Six-phase operationalization procedure (Liu, 2017)
4. **Examples** -- Semantically selected real-world case studies (e.g., Brazil-China soybean trade, Beijing water system)
5. **Output Format** -- Structured template for seven analysis sections
   (with a **Citation Rules sub-layer, 5b**: turn-scoped `[Tk:N]`/`[Tk:Wn]`
   citation mechanics and hallucination guardrails)
6. **Interaction** -- Multi-turn refinement guidelines

Before calling the LLM, the system **conditionally** injects (each
only fires if its precondition is met):

- **Pericoupling-database hints** — exactly one of two flavors fires per query:
  - *Country-level hint*: if ≥1 country name is detected and no
    ADM1 region applies.  If ≥2 countries are named, each
    focal-vs-other pair is reported with its database
    classification (e.g., `"Mexico (MEX) and United States (USA)
    are pericoupled"` for adjacent pairs, `"Brazil (BRA) and
    China (CHN) are telecoupled"` for distant pairs).  If exactly
    1 country is named (PR #44), the focal country's full
    pericoupled-neighbor list is injected as candidate context
    (e.g., a Mexico-only query gets Belize, Guatemala, USA
    presented as REFERENCE — the LLM is encouraged to consider
    these as potential pericoupled flows but must confirm with
    independent evidence).  Framing is REFERENCE-only in both
    cases (harmonized with the ADM1 hint in PR #43).
  - *ADM1 (subnational) hint*: if any ADM1 region (e.g., Michigan,
    Mato Grosso) is detected.  The focal region's adjacent
    neighbors are listed as **reference-only adjacency** rather
    than asserted pericoupling — the LLM is told that adjacency
    alone is NOT evidence of pericoupling and must find
    independent flow evidence (PR #20).
  - Queries with no detectable country (and no ADM1 region) get
    no hint; queries with one country named but no pericoupled
    neighbors in the database (island nations like Australia,
    Japan, Cuba) also get no hint.
- **Web search context** with `[Tk:W1]`, `[Tk:W2]` labels for
  inline citation (turn-scoped — `k` is the conversation turn).
  Only when `web_search=True` and at least one hit returned.
- **Structured web map hints** — receiving/spillover countries and
  flows extracted by a second LLM pass over the web snippets, each
  kept only if it resolves to a real country/union and cites a real
  retrieved snippet (confidence ≥ 0.7).  This grounds the *evidence*,
  not the spillover *role* (which classification the LLM still
  decides).  Only when `web_structured_extraction=True`.

### 4.3 Retrieval-Augmented Generation (RAG)

The RAG engine provides evidence grounding from a corpus of 420 telecoupling and metacoupling papers. The split is by what can be redistributed, not by topic: 192 papers whose licences allow it are indexed as **full text**, and the remaining 228 are indexed as **structured, paraphrased summaries** (metadata plus key findings, no source text) so they stay searchable without reproducing copyrighted material. Those 228 are papers the corpus cannot redistribute in full — both paywalled articles and open-to-read articles whose licences still restrict reuse:

- **Indexing**: Papers are chunked by section and indexed by one of two backends:
  - **Embeddings (default)** -- semantic retrieval via `fastembed` + the `BAAI/bge-base-en-v1.5` ONNX model. Captures synonyms, paraphrases, and related concepts (e.g., a query about "soybean trade" also matches chunks about "soya bean exports" and "Glycine max shipments"). Pre-computed corpus vectors are shipped with the package as `chunk_embeddings.npy` (~15 MB) so users never have to re-encode.
  - **TF-IDF (fallback)** -- lexical retrieval using TF-IDF + cosine similarity. Activated when `fastembed` is unavailable or the pre-computed file is missing.
- **Retrieval**: Cosine similarity with a per-paper cap (default 3 chunks/paper, configurable via `rag_max_chunks_per_paper`; within-paper section diversity preferred). The relevance floor is backend-aware — `rag_min_score` defaults to **0.60** for embeddings (BGE-base cosine) and **0.01** for TF-IDF (the two score scales are not comparable).
- **Citation**: Evidence passages are appended as `[Tk:1]`, `[Tk:2]`, ... with inline annotation (turn-scoped — `k` marks the turn so prior-turn references remain unambiguous across multi-turn conversations)
- **Lightweight**: `fastembed` + `onnxruntime` add ~20 MB to the install; no torch/GPU dependencies
- **Backend selection**: `MetacouplingAssistant(..., rag_backend="auto")` (default) picks embeddings if available and transparently falls back to TF-IDF. Explicit options: `"embeddings"`, `"tfidf"`.

### 4.4 Web Search Grounding

Web search injects current, real-world context (trade data, policies, recent events) that may not be in the LLM's training data:

- **Native auto-wiring** when `web_search=True`: each adapter routes to its own backend — `OpenAIWebSearchBackend` (PR #17), `AnthropicWebSearchBackend` (PR #28), `GeminiWebSearchBackend` (Google Search grounding, PR #29), `GrokWebSearchBackend` (xAI Live Search incl. X/Twitter, PR #29). Custom clients fall back to the built-in DuckDuckGo search cascade.
- DuckDuckGo backend has three fallback layers: `ddgs` -> `duckduckgo_search` -> stdlib (`urllib` + `html.parser`); works on Google Colab without any extra packages.
- Results cited as `[Tk:W1]`, `[Tk:W2]`, ... -- the `W` prefix distinguishes web sources from literature `[Tk:1]`, `[Tk:2]`. `k` is the turn index, so prior-turn web references stay stable across `refine()` calls.
- Recommended default for web-grounded maps: `web_structured_extraction=True` runs a second LLM pass (strict JSON / tool-output across all four providers, PR #28-#30) over the web snippets and validates map-ready countries and flows before using them in auto-maps.
- `evidence_coverage_note` (PR #20) is a one-paragraph LLM self-assessment that summarises what kinds of web sources were available, what coverage gaps remain, and whether the analysis fell back to training memory — surfaced on `result.parsed.evidence_coverage_note`.
- Supranational unions (EU / ASEAN / USMCA / NAFTA) are handled end-to-end: prompt teaches the LLM to keep the union label + list members (PR #22), Stage-3 web-summary buffer is bumped to 2500 chars (PR #26), and the map renderer dissolves union borders and colours member states by their relationship to the focal country (PR #23, #25).

### 4.5 Pericoupling Databases

Two geographic adjacency databases validate LLM coupling classifications:

| Database | Scope | Coverage |
|---|---|---|
| Country-level | Sovereign states | Full global (ISO alpha-3) |
| ADM1 (subnational) | First-level administrative regions | 3,374 regions, 8,456 shared-border pairs (8,065 under the default `moderate` coupling standard), 196 countries |

Functions: `is_pericoupled()`, `get_pericoupled_neighbors()`, `lookup_adm1_pericoupling()`, etc.

Both levels accept two orthogonal toggles: **`de_facto_borders`** (default `True`; fold disputed land into its de-facto administrator, vs the strict WB standard layer) and **`coupling_standard`** (`stringent` / `moderate` / `lenient`, default `moderate`) — for pairs sharing **only** a river/lake border, `moderate` requires a fixed crossing open to traffic, `stringent` drops all water-only pairs, and `lenient` keeps them.  Bridge presence was OSM-classified and then independently verified (web search + geometric province check + manual review); see `docs/BRIDGE_CLASSIFICATION_METHODOLOGY.md`.

Region names are resolved to ADM1 codes by `resolve_adm1_code`, which—beyond unaccented forms and hyphenated compounds—consults a bundled, deterministically-validated **English-exonym alias table** (1,145 aliases; e.g. Bavaria→Bayern, Tuscany→Toscana) and returns `None` rather than guessing on an ambiguous or padded name.  (The prompt-hint scanner additionally strips possessive suffixes, e.g. "Michigan's"→"Michigan", from free text before calling it; `resolve_adm1_code` itself does not — `resolve_adm1_code("Michigan's")` returns `None` even though `resolve_adm1_code("Michigan")` returns `"USA023"`.)

### 4.6 Literature Recommendations

From a curated BibTeX database of 265 empirical telecoupling/metacoupling journal articles (2013–2026), the system recommends the most relevant papers by matching keywords, coupling types, and domain overlap with the analysis.  Call `get_database_info()` for live counts if the corpus drifts.

### 4.7 Map Visualization

Three map functions generate matplotlib figures:

| Function | Level | Data Source |
|---|---|---|
| `plot_focal_country_map(country)` | Country | Local World Bank Admin 0 `all_layers` if available; otherwise hosted `Admin 0_all_layers` mirror; fallback official World Bank Admin 0 10m |
| `plot_analysis_map(parsed_analysis)` | Country | Local World Bank Admin 0 `all_layers` if available; otherwise hosted `Admin 0_all_layers` mirror; fallback official World Bank Admin 0 10m |
| `plot_focal_adm1_map(adm1_code)` | Subnational (ADM1) | Local World Bank Admin 1 if available; otherwise hosted `Admin 1.gpkg` mirror, with World Bank `NDLSA.gpkg` for disputed-area overlay |

Map features: coupling-colored regions, flow arrows, disputed territory hatching, customizable color palettes.

### 4.8 LLM Client Abstraction

The package uses a protocol-based design that supports any LLM backend:

```python
# Built-in adapters
OpenAIAdapter(client, model="gpt-4o")
AnthropicAdapter(client, model="claude-sonnet-4-20250514")
GeminiAdapter(client, model="gemini-2.5-flash")
GrokAdapter(client, model="grok-3")     # OpenAI-protocol-compatible

# Any custom client with a chat() method also works
class MyClient:
    def chat(self, messages, temperature=0.7, max_tokens=None):
        return LLMResponse(content="...")
```

Each built-in adapter auto-wires its native web-search backend when `web_search=True`; custom clients fall back to DuckDuckGo.

### 4.9 Scholar Export (PR #31, #32)

`AnalysisResult` exposes three exporter surfaces so users can turn an LLM analysis into something a co-author or reviewer can read:

| Surface | Description | Install needed |
|---|---|---|
| `result.abstract` | One-paragraph scholar abstract from a second, conservative LLM pass | none |
| `result.to_markdown(path=None)` | Manuscript-ready Markdown with auto-derived title, bold sub-field labels, per-category Flows / Causes / Effects subsections, RAG + web evidence blocks, classification bullets | none |
| `result.to_docx(path)` | Word document with the same structure as the Markdown export | `metacouplingllm[export]` (python-docx) |

PR #32 added the auto-derived title, bold sub-field labels, and per-category flow / cause / effect subsections; PR #33 added classification bullets at the top of every system role section.

### 4.10 Quantitative Indicators (PR #35)

The `metacouplingllm.indicators` submodule computes deterministic metacoupling indicators on user-supplied flow data:

| Function | What it does |
|---|---|
| `classify_coupling(edges, focal_id, adjacency, ...)` | Add `coupling_type` (I / P / T) to an edge table using a user-supplied adjacency table. |
| `compute_flow_shares(data, ...)` | Intracoupled / Pericoupled / Telecoupled Flow Shares (IFS / PFS / TFS) per focal system. |
| `compute_mfe(data, ...)` | Metacoupled Flow Evenness — normalised Shannon entropy across coupling types. |
| `compute_mfci(data, ...)` | Metacoupled Flow Concentration Index — normalised HHI (IFCI / PFCI / TFCI) within each coupling type. |
| `summarize_metacoupling(data, ...)` | One-shot combined indicator table. |

Built on established statistics (Shannon 1948 entropy; Hirschman 1945 HHI, normalised per Hannah & Kay 1977; Equivalent Number of Partners per Laakso & Taagepera 1979), not invented indices. Brazil-soybean worked example: IFS = 0.10, PFS = 0.20, TFS = 0.70, MFE ≈ 0.73, TFCI ≈ 0.33. See MANUAL §16 for the full math + code.

Pandas is the only added dependency; install via `pip install "metacouplingllm[indicators]"`.

### 4.11 LLM-Assisted Indicator Helpers (PR #36)

Five optional helpers in `metacouplingllm.indicators` wrap natural-language judgment tasks around the deterministic indicator core. Each returns `(result, LLMTrace)` for reproducibility (see MANUAL §17, *LLM-Assisted Indicator Helpers*):

| Helper | What it does |
|---|---|
| `define_study(description, *, llm_client)` | Natural-language description → structured study config dict. |
| `check_inputs(data_summary, sample_rows, *, llm_client)` | Validate user data; flag missing inputs, unit issues, self-loop intracoupling. |
| `classify_ambiguous_edges(edges, study_config, *, llm_client)` | Classify edges the deterministic pass couldn't resolve; returns `"I"` / `"P"` / `"T"` / `"unknown"` per edge with confidence and reason. |
| `interpret_results(results, *, llm_client, audience)` | Plain-language interpretation of an indicator table; audience presets `"academic"` / `"general"` / `"policy"`. |
| `write_methods(indicator_spec, *, llm_client)` | Manuscript-ready Methods text with formulas + standard citations (Shannon 1948, Hirschman 1945, Hannah & Kay 1977, Laakso & Taagepera 1979, Liu 2017). |

The `LLMTrace` dataclass carries `timestamp_utc`, `model`, `prompt_version`, `system_prompt`, `user_prompt`, `raw_response`, `usage` so old traces stay attributable when prompt wording evolves.

**Automatic LLM resolution of ambiguous edges in `classify_coupling()`:** pass `llm_client=` (plus optional `study_config=` and `model=`) and the function automatically calls `classify_ambiguous_edges()` on any rows the deterministic pass left as `NaN`, merging suggestions back. When the LLM returns `"unknown"`, the row stays `NaN` — the package never lets the LLM invent adjacency facts silently.

All five helpers work across OpenAI / Anthropic / Gemini / Grok adapters and use each provider's native strict-JSON / tool-output mode for the structured-output helpers.

---

## 5. Operation Procedure

### Step 1: Install

```bash
pip install "metacouplingllm[all]"          # full installation
# or selectively:
pip install "metacouplingllm[openai]"       # OpenAI support
pip install "metacouplingllm[anthropic]"    # Anthropic support
pip install "metacouplingllm[gemini]"       # Google Gemini support
pip install "metacouplingllm[grok]"         # xAI Grok support
pip install "metacouplingllm[viz]"          # maps (geopandas + matplotlib)
pip install "metacouplingllm[search]"       # web search (ddgs)
pip install "metacouplingllm[indicators]"   # pandas-based quantitative indicators
pip install "metacouplingllm[export]"       # python-docx scholar export
```

### Step 2: Initialize

```python
from openai import OpenAI
from metacouplingllm import (
    JOURNAL_ARTICLES_2025,
    MetacouplingAssistant,
    OpenAIAdapter,
)

client = OpenAI(api_key="your-key")
advisor = MetacouplingAssistant(
    llm_client=OpenAIAdapter(client, model="gpt-4o"),
    auto_map=True,
    rag_corpus=JOURNAL_ARTICLES_2025,
    web_search=True,
    web_search_max_results=10,
    web_structured_extraction=True,
    rag_top_k=8,
    rag_min_score=0.60,
)
```

### Step 3: Analyze

```python
result = advisor.analyze("""
    My research examines what the impact of Brazil's soybeans exports is.
""")

print(result.formatted)   # Full formatted report

if result.map:
    result.map.savefig("metacoupling_map.png", dpi=150, bbox_inches="tight")
```

The output includes:
- Coupling classification (intracoupling / pericoupling / telecoupling)
- Systems identification (sending, receiving, spillover with human and natural subsystems)
- Flows analysis (matter, capital, information, energy, people, organisms)
- Agents grouped by the fixed five-category agent vocabulary
- Causes grouped by the fixed cause/effect category vocabulary
- Effects grouped by the fixed cause/effect category vocabulary
- Research gaps and suggestions
- Literature evidence with `[Tk:1]`-`[Tk:N]` citations (turn-scoped)
- Web sources with `[Tk:W1]`-`[Tk:WN]` citations (turn-scoped)
- Pericoupling database validation
- An `evidence_coverage_note` self-assessment from the LLM summarising what evidence backs which claims
- Scholar-ready outputs accessible via `result.abstract`, `result.to_markdown(path)`, and `result.to_docx(path)` (the latter requires `metacouplingllm[export]`)

### Step 4: Refine (optional)

```python
result2 = advisor.refine(
    "The main export destinations are Japan and Mexico.",
    focus_component="systems",
)
print(result2.formatted)
```

### Step 5: Visualize (optional)

```python
# Or generate maps independently
from metacouplingllm import plot_focal_country_map, plot_focal_adm1_map

fig = plot_focal_country_map("USA")
fig.savefig("usa_coupling_map.png", dpi=150)

fig_wb = plot_focal_country_map(
    "USA",
    adm0_shapefile=r"C:\path\to\World Bank Official Boundaries - Admin 0_all_layers.gpkg",
)
fig_wb.savefig("usa_coupling_map_world_bank.png", dpi=150)

fig2 = plot_focal_adm1_map("USA023")  # Michigan
fig2.savefig("michigan_adm1_map.png", dpi=150)
```

If you do not pass `adm0_shapefile`, the package first looks for a local
`World Bank Official Boundaries - Admin 0_all_layers.gpkg`. If it cannot
find one, it auto-downloads the hosted `Admin 0_all_layers` mirror into
the cache on first use, then falls back to the official World Bank Admin 0
10m basemap if needed.

Similarly, if you do not pass `adm1_shapefile`, ADM1 maps first look for a
local Admin 1 file and then auto-download the hosted `Admin 1.gpkg` mirror.

### Step 6: Access Structured Data

```python
# Structured access to parsed analysis
p = result.parsed
print(p.coupling_classification)
print(list(p.iter_system_entries()))        # sending/receiving/spillover system entries
print(list(p.iter_flow_entries()))          # flow dicts
print(list(p.iter_agent_entries()))         # agent dicts
print(list(p.iter_category_items("causes")))   # (coupling_type, category, item) triples
print(list(p.iter_category_items("effects")))  # (coupling_type, category, item) triples
print(p.research_gaps)        # list of research gap strings
print(p.evidence_coverage_note)  # PR #20 — reviewer-facing evidence audit
```

Or hand the result to the scholar exporters (PR #31, #32):

```python
print(result.abstract)                # one-paragraph scholar abstract
result.to_markdown("brazil_soybean.md")
result.to_docx("brazil_soybean.docx") # requires metacouplingllm[export]
```

### Step 7 (optional): Compute Quantitative Indicators

When you have your own flow data, plug it into `metacouplingllm.indicators` (PR #35) for deterministic indicator math:

```python
import pandas as pd
from metacouplingllm.indicators import classify_coupling, summarize_metacoupling

edges = pd.DataFrame({
    "focal_system_id":   ["Brazil"] * 6,
    "origin_id":         ["Brazil"] * 6,
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

The same DataFrame plugs into the optional LLM-assisted helpers (PR #36) — `define_study`, `check_inputs`, `classify_ambiguous_edges`, `interpret_results`, `write_methods` — for natural-language study setup, validation, ambiguous-edge classification, interpretation, and manuscript prose around the deterministic numbers. Install via `pip install "metacouplingllm[indicators]"`.

---

## 6. Bundled Data

| Resource | Description |
|---|---|
| 420 papers (Papers.zip) | Markdown for RAG — 192 indexed as full text, 228 as structured paraphrased summaries (papers that cannot be redistributed in full: paywalled, or open-to-read but restrictively licensed) |
| BibTeX database (telecoupling_literature.bib) | 265 empirical journal articles (2013–2026) with metadata for literature recommendation |
| Country pericoupling database (CSV) | Global country-pair adjacency classification |
| ADM1 edge list (CSV) | 8,456 subnational shared-border pairs across 3,374 regions in 196 countries (World Bank Official Boundaries, 2026-05-14; see `data/PROVENANCE.md`) |
| ADM1 alias table (CSV) | 1,145 English exonyms / alternative spellings for 863 ADM1 regions in 136 countries (e.g. Bavaria→Bayern, Tuscany→Toscana), so `resolve_adm1_code` matches common English names; deterministically validated (see `data/PROVENANCE.md`) |
| Framework examples | Curated case studies (soybean trade, urban water) for prompt injection |

*Counts above are as of v0.1.3.  Call `get_database_info()` for the live BibTeX count and `len(zipfile.ZipFile("Papers.zip").namelist())` for the live paper count if the corpus drifts.*

---

## 7. Design Principles

- **Lean dependencies** -- Core analysis works with only `numpy`, `fastembed`, and an LLM client; visualization, web search, quantitative indicators, and scholar export are optional extras
- **Graceful degradation** -- Each optional feature (RAG, web search, maps, literature, indicators, export) can be independently enabled or disabled; RAG transparently falls back from embeddings to TF-IDF when `fastembed` is unavailable
- **Protocol-based extensibility** -- Any object with a `chat()` method works as an LLM client
- **Pre-LLM knowledge injection, post-LLM validation** -- Web search, example selection, and the pericoupling-database hint happen before the LLM call; pericoupling validation of the LLM's stated classification runs post-LLM on the parsed output, correcting hallucinated adjacency claims against the database
- **Semantic RAG** -- Pre-computed BGE-base embeddings shipped with the package; semantic matching catches synonyms and paraphrases that TF-IDF misses. TF-IDF remains available as a fallback.
- **Colab-compatible** -- Web search includes a zero-dependency stdlib fallback for restricted environments
- **Deterministic-first for quantitative analysis (PR #35, #36)** -- The `metacouplingllm.indicators` math never calls an LLM; numbers are reproducible from the input DataFrame alone. The five LLM-assisted helpers are explicitly scoped to natural-language tasks (study setup, validation, interpretation, methods drafting) and always pair their output with an `LLMTrace` record. When the LLM is uncertain, it returns `"unknown"` rather than guess; the package never lets the LLM invent adjacency facts or flow values silently.

---

## 8. Testing

The package includes a comprehensive test suite:

```bash
pip install "metacouplingllm[dev]"
pytest tests/
```

1381 tests covering all modules: core advisor logic, framework enums, prompt construction, LLM parsing, RAG retrieval, literature matching, web search (including stdlib fallback), pericoupling databases, country resolution, visualization colors, map generation, scholar export, quantitative indicators, and a CI-enforced doc-capability drift guard (PR #46) that fails the build when shipped features aren't advertised in INTRODUCTION/README/MANUAL.
