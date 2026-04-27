# Metacoupling: An LLM-Powered Python Package for Applying the Metacoupling Framework

## 1. Overview

**Metacoupling** is a Python package (v0.1.0) that helps researchers apply the telecoupling and metacoupling frameworks (Liu et al., 2013; Liu, 2017) to their own research topics using Large Language Models (LLMs). Given a natural-language research description, the package produces a structured, framework-compliant analysis that identifies coupled human and natural systems, classifies coupling types, maps flows, agents, causes, and effects, and grounds the output in both a curated literature database and real-time web search results.

**Key capabilities:**

- Structured metacoupling analysis from free-text research descriptions
- Multi-turn refinement via conversational LLM interaction
- Retrieval-Augmented Generation (RAG) with 262 full-text telecoupling papers
- **RAG-only literature Q&A mode** for users already familiar with the framework (`coupling_analysis=False`)
- Literature recommendation from a curated BibTeX database
- Real-time web search grounding (DuckDuckGo, no API key required)
- Pericoupling validation against two geographic databases (country-level and subnational ADM1)
- Automated map generation at both country and subnational levels
- Support for OpenAI, Anthropic, **Google Gemini**, **xAI Grok**, or any custom LLM backend (each with native web-search auto-wiring)

**Requirements:** Python 3.10+, no hard runtime dependencies. Optional extras: `openai`, `anthropic`, `geopandas`+`matplotlib` (visualization), `ddgs` (web search).

### Quick Install

```bash
pip install metacoupling
pip install "metacoupling[openai]"
pip install "metacoupling[anthropic]"
pip install "metacoupling[gemini]"
pip install "metacoupling[grok]"
pip install "metacoupling[search]"
pip install "metacoupling[viz]"
pip install "metacoupling[all]"
```

If you want to use the built-in OpenAI example below, install `metacoupling[openai]`.

### What Users Get

- A structured metacoupling analysis from a plain-language study description
- Optional literature recommendations and supporting evidence passages
- Optional web-grounded context for current trade, policy, or event information
- Optional map generation for country-level and ADM1-level analyses

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
  | (6-layer       |  |  (DuckDuckGo)  |  | (262 papers,  |
  |  system prompt)|  |                |  |  embeddings + |
  |                |  |                |  |  TF-IDF)      |
  +-------+-------+  +-------+--------+  +-------+-------+
          |                   |                   |
          +-------------------+-------------------+
                              |
                    +---------v----------+
                    |     LLM Client     |
                    |  (OpenAI/Anthropic/ |
                    |   custom backend)  |
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
  | Pericoupling   |  | Literature     |  | Map Generator |
  | Validation     |  | Recommendations|  | (world/ADM1)  |
  +---------------+  +----------------+  +---------------+
                              |
                    +---------v----------+
                    |   AnalysisResult   |
                    |  .parsed           |
                    |  .formatted        |
                    |  .map              |
                    +--------------------+
```

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
    llm_client=OpenAIAdapter(OpenAI(api_key="..."), model="gpt-5.2"),
    auto_map=True,              # Generate map automatically
    rag_corpus=JOURNAL_ARTICLES_2025,  # Use bundled 2025 journal corpus
    web_search=True,            # Ground analysis in web search results
    web_search_max_results=5,   # Number of web results to retrieve
    web_structured_extraction=True,  # Recommended with web_search + auto_map
    rag_top_k=10,               # Number of RAG evidence passages
    rag_min_score=0.15,         # Minimum cosine similarity for RAG
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
6. **Interaction** -- Multi-turn refinement guidelines and citation expectations

Before calling the LLM, the system also injects:
- **Pericoupling hints** from the geographic database (e.g., "Michigan and Indiana are pericoupled")
- **Web search context** with `[W1]`, `[W2]` labels for inline citation
- **Structured web map hints** with validated countries and flows when enabled
- **ADM1 neighbor information** when subnational regions are detected

### 4.3 Retrieval-Augmented Generation (RAG)

The RAG engine provides evidence grounding from 262 full-text telecoupling and metacoupling papers:

- **Indexing**: Papers are chunked by section and indexed by one of two backends:
  - **Embeddings (default)** -- semantic retrieval via `fastembed` + the `BAAI/bge-small-en-v1.5` ONNX model. Captures synonyms, paraphrases, and related concepts (e.g., a query about "soybean trade" also matches chunks about "soya bean exports" and "Glycine max shipments"). Pre-computed corpus vectors are shipped with the package as `chunk_embeddings.npy` (~15 MB) so users never have to re-encode.
  - **TF-IDF (fallback)** -- lexical retrieval using TF-IDF + cosine similarity. Activated when `fastembed` is unavailable or the pre-computed file is missing.
- **Retrieval**: Cosine similarity; top-k deduplication (at most one chunk per paper)
- **Citation**: Evidence passages are appended as `[1]`, `[2]`, ... with inline annotation
- **Lightweight**: `fastembed` + `onnxruntime` add ~20 MB to the install; no torch/GPU dependencies
- **Backend selection**: `MetacouplingAssistant(..., rag_backend="auto")` (default) picks embeddings if available and transparently falls back to TF-IDF. Explicit options: `"embeddings"`, `"tfidf"`.

### 4.4 Web Search Grounding

Web search injects current, real-world context (trade data, policies, recent events) that may not be in the LLM's training data:

- Three fallback backends: `ddgs` -> `duckduckgo_search` -> stdlib (`urllib` + `html.parser`)
- Works on Google Colab without any extra packages (stdlib fallback)
- Results cited as `[W1]`, `[W2]`, ... -- distinct from literature `[1]`, `[2]`
- Recommended default for web-grounded maps: `web_structured_extraction=True` runs a second LLM pass over the web snippets and validates map-ready countries and flows before using them in auto-maps

### 4.5 Pericoupling Databases

Two geographic adjacency databases validate LLM coupling classifications:

| Database | Scope | Coverage |
|---|---|---|
| Country-level | Sovereign states | Full global (ISO alpha-3) |
| ADM1 (subnational) | First-level administrative regions | 3,366 regions, 8,290 border pairs, 195 countries |

Functions: `is_pericoupled()`, `get_pericoupled_neighbors()`, `lookup_adm1_pericoupling()`, etc.

### 4.6 Literature Recommendations

From a curated BibTeX database of 262 telecoupling/metacoupling papers, the system recommends the most relevant papers by matching keywords, coupling types, and domain overlap with the analysis.

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

# Any custom client with a chat() method also works
class MyClient:
    def chat(self, messages, temperature=0.7, max_tokens=None):
        return LLMResponse(content="...")
```

---

## 5. Operation Procedure

### Step 1: Install

```bash
pip install metacoupling[all]   # full installation
# or selectively:
pip install metacoupling[openai]      # OpenAI support
pip install metacoupling[anthropic]   # Anthropic support
pip install metacoupling[viz]         # maps (geopandas + matplotlib)
pip install metacoupling[search]      # web search (ddgs)
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
    llm_client=OpenAIAdapter(client, model="gpt-5.2"),
    auto_map=True,
    rag_corpus=JOURNAL_ARTICLES_2025,
    web_search=True,
    web_search_max_results=5,
    web_structured_extraction=True,
    rag_top_k=10,
    rag_min_score=0.15,
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
- Literature evidence with `[1]`-`[N]` citations
- Web sources with `[W1]`-`[WN]` citations
- Pericoupling database validation

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
print(p.systems)      # dict of sending/receiving/spillover
print(p.flows)        # list of flow dicts
print(p.agents)       # list of agent dicts
print(p.causes)       # dict of cause categories
print(p.effects)      # dict of effect categories
print(p.suggestions)  # list of research gap strings
```

---

## 6. Bundled Data

| Resource | Description |
|---|---|
| 262 full-text papers (Papers.zip) | Markdown versions of telecoupling/metacoupling research papers for RAG |
| BibTeX database (telecoupling_literature.bib) | 262 curated entries with metadata for literature recommendation |
| Country pericoupling database (CSV) | Global country-pair adjacency classification |
| ADM1 edge list (CSV) | 8,290 subnational border pairs across 3,366 regions in 195 countries |
| Framework examples | Curated case studies (soybean trade, urban water) for prompt injection |

---

## 7. Design Principles

- **Lean dependencies** -- Core analysis works with only `numpy`, `fastembed`, and an LLM client; visualization and web search are optional extras
- **Graceful degradation** -- Each optional feature (RAG, web search, maps, literature) can be independently enabled or disabled; RAG transparently falls back from embeddings to TF-IDF when `fastembed` is unavailable
- **Protocol-based extensibility** -- Any object with a `chat()` method works as an LLM client
- **Pre-LLM knowledge injection** -- Pericoupling validation, web search, and example selection all happen before the LLM call, reducing hallucination
- **Semantic RAG** -- Pre-computed BGE-small embeddings shipped with the package; semantic matching catches synonyms and paraphrases that TF-IDF misses. TF-IDF remains available as a fallback.
- **Colab-compatible** -- Web search includes a zero-dependency stdlib fallback for restricted environments

---

## 8. Testing

The package includes a comprehensive test suite:

```bash
pip install metacoupling[dev]
pytest tests/
```

429 tests covering all modules: core advisor logic, framework enums, prompt construction, LLM parsing, RAG retrieval, literature matching, web search (including stdlib fallback), pericoupling databases, country resolution, visualization colors, and map generation.
