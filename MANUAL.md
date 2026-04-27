# Metacoupling Package — User Manual

**Version 0.1.0**

A Python package that helps researchers apply the telecoupling and metacoupling
frameworks (Liu et al., 2013; Liu, 2017) to their research using Large Language
Models (LLMs).

---

## Table of Contents

1. [Installation](#1-installation)
2. [Quick Start](#2-quick-start)
3. [Core Concepts](#3-core-concepts)
4. [LLM Setup](#4-llm-setup)
5. [Running an Analysis](#5-running-an-analysis)
6. [Refining an Analysis](#6-refining-an-analysis)
7. [Understanding the Output](#7-understanding-the-output)
8. [Pericoupling Database](#8-pericoupling-database)
9. [Literature Recommendations](#9-literature-recommendations)
10. [World Map Visualization](#10-world-map-visualization)
11. [Advanced Usage](#11-advanced-usage)
12. [API Reference](#12-api-reference)
13. [Troubleshooting](#13-troubleshooting)

---

## 1. Installation

### Basic installation (no LLM provider yet)

```bash
pip install metacoupling
```

### With OpenAI support

```bash
pip install metacoupling[openai]
```

### With Anthropic (Claude) support

```bash
pip install metacoupling[anthropic]
```

### With Google Gemini support

```bash
pip install metacoupling[gemini]
```

### With xAI Grok support

```bash
pip install metacoupling[grok]
```

### With visualization support (world maps)

```bash
pip install metacoupling[viz]
```

### Install everything

```bash
pip install metacoupling[all]
```

### Requirements

- Python 3.10 or higher
- An API key from OpenAI or Anthropic (for LLM-based analysis)

> **Jupyter/Colab users:** Use `!pip install` or `%pip install` (with the
> exclamation mark or percent sign) when running install commands inside
> notebook cells. Running `pip install` without the prefix will cause a
> `SyntaxError`.

---

## 2. Quick Start

```python
from openai import OpenAI
from metacouplingllm import MetacouplingAssistant, OpenAIAdapter

# 1. Set up your LLM client
client = OpenAI(api_key="sk-your-api-key-here")
adapter = OpenAIAdapter(client, model="gpt-5.2")

# 2. Create an advisor
advisor = MetacouplingAssistant(
    adapter,
    web_search=True,
    web_search_max_results=5,
    web_structured_extraction=True,  # Recommended with web_search + auto_map
    auto_map=True,
    rag_corpus="journal_articles_2025",
    rag_top_k=10,
    rag_min_score=0.15,
)

# 3. Analyze your research
result = advisor.analyze("""
    My research examines what the impact of Brazil's soybeans exports is.
""")

# 4. View the formatted output
print(result.formatted)

# 5. Save the map if it was generated
if result.map:
    result.map.savefig("map.png", dpi=150, bbox_inches="tight")
```

Recommended default: use `web_structured_extraction=True` whenever you enable
both `web_search=True` and `auto_map=True`. The advisor then performs an extra
LLM pass over the web snippets to extract validated receiving countries,
spillover countries, and map-ready flows. The validated payload is also stored
on `result.web_map_signals`.

---

## 3. Core Concepts

### The Metacoupling Framework

The metacoupling framework (Liu, 2017) is a comprehensive approach for
understanding human-nature interactions across boundaries. It extends the
telecoupling framework (Liu et al., 2013) and classifies interactions into
three types:

| Coupling Type | Definition | Example |
|---|---|---|
| **Intracoupling** | Interactions within a single system | Domestic water management |
| **Pericoupling** | Interactions between adjacent systems | US-Mexico border trade |
| **Telecoupling** | Interactions between distant systems | Brazil-China soybean trade |

### The Five Components

Every telecoupling/metacoupling analysis identifies five components:

1. **Systems** — Sending, receiving, and spillover systems (each with human
   and natural subsystems)
2. **Flows** — Material, energy, information, financial, and people transfers
3. **Agents** — Decision-makers and active entities grouped as individuals / households;
   firms / traders / corporations; governments / policymakers; organizations / NGOs;
   and non-human agents
4. **Causes** — Drivers categorized with the fixed cause/effect vocabulary:
   economic; political / institutional; ecological / biological;
   technological / infrastructural; cultural / social / demographic;
   hydrological; climatic / atmospheric; geological / geomorphological
5. **Effects** — Outcomes categorized with the same fixed vocabulary

### How the Package Works

```
Your Research Description
         │
         ▼
┌─────────────────────┐
│   Prompt Builder     │  ← 6-layer prompt architecture
│  (system + context)  │  ← Pericoupling database hints
│  (curated examples)  │  ← Framework knowledge injection
└─────────┬───────────┘
          │
          ▼
┌─────────────────────┐
│    LLM Provider      │  ← OpenAI / Anthropic / Custom
│   (GPT-4o, Claude)   │
└─────────┬───────────┘
          │
          ▼
┌─────────────────────┐
│   Response Parser    │  ← Best-effort structured extraction
│  + Pericoupling      │  ← Post-LLM database validation
│    Validation        │
└─────────┬───────────┘
          │
          ▼
┌─────────────────────┐
│   Formatted Output   │  ← Human-readable report
│  + Literature Recs   │  ← Optional paper recommendations
│  + World Map         │  ← Optional visualization
└─────────────────────┘
```

---

## 4. LLM Setup

### Option A: OpenAI

```python
from openai import OpenAI
from metacouplingllm import MetacouplingAssistant, OpenAIAdapter

client = OpenAI(api_key="sk-your-api-key-here")
adapter = OpenAIAdapter(client, model="gpt-5.2")
advisor = MetacouplingAssistant(adapter)
```

Supported models: `"gpt-5.2"`, `"gpt-5"`, `"gpt-4o"`, `"gpt-4o-mini"`, or any
OpenAI chat model.

**Using Google Colab with secure API key storage:**

```python
from google.colab import userdata
api_key = userdata.get('OPENAI_API_KEY')

from openai import OpenAI
from metacouplingllm import MetacouplingAssistant, OpenAIAdapter

client = OpenAI(api_key=api_key)
advisor = MetacouplingAssistant(OpenAIAdapter(client, model="gpt-5.2"))
```

### Option B: Anthropic (Claude)

```python
from anthropic import Anthropic
from metacouplingllm import MetacouplingAssistant, AnthropicAdapter

client = Anthropic(api_key="sk-ant-your-api-key-here")
adapter = AnthropicAdapter(client, model="claude-sonnet-4-20250514")
advisor = MetacouplingAssistant(adapter)
```

### Option C: Google Gemini

Built against the new **`google.genai` SDK** (the unified Google
GenAI SDK released in 2025; the older `google.generativeai` package
is deprecated).

```python
from google import genai
from metacouplingllm import MetacouplingAssistant, GeminiAdapter

client = genai.Client(api_key="AIza-your-api-key-here")
adapter = GeminiAdapter(client, model="gemini-2.5-flash")
advisor = MetacouplingAssistant(adapter)
```

Default model is `gemini-2.5-flash` (fast, cheap). Use
`gemini-2.5-pro` for higher-quality framework analyses.

When you set `web_search=True` with a `GeminiAdapter`, the advisor
automatically uses Gemini's **Google Search grounding** tool — the
same auto-wiring pattern as OpenAI/Anthropic.

Install with: `pip install "metacoupling[gemini]"`

### Option D: xAI Grok

```python
from openai import OpenAI
from metacouplingllm import MetacouplingAssistant, GrokAdapter

client = OpenAI(api_key="xai-your-api-key-here", base_url="https://api.x.ai/v1")
adapter = GrokAdapter(client, model="grok-3")
advisor = MetacouplingAssistant(adapter)
```

Grok's API is OpenAI-protocol-compatible, so you reuse the `openai`
SDK with xAI's base URL. The dedicated `GrokAdapter` (rather than
`OpenAIAdapter`) lets the advisor route web search to Grok's native
**Live Search** tool (which queries both the web and X/Twitter).

Install with: `pip install "metacoupling[grok]"`

### Option E: Custom LLM Client

Any object with a compatible `chat()` method will work — no inheritance
required:

```python
from metacouplingllm import MetacouplingAssistant, LLMResponse, Message

class MyLocalLLM:
    def chat(self, messages: list[Message], temperature=0.7, max_tokens=None):
        # Call your local model here
        response_text = my_model.generate(messages[-1].content)
        return LLMResponse(content=response_text)

advisor = MetacouplingAssistant(MyLocalLLM())
```

When `web_search=True` and the client is **not** one of the four
built-in adapters, web search auto-falls back to DuckDuckGo (free, no
API key).

---

## 5. Running an Analysis

### Basic analysis

```python
result = advisor.analyze("""
    My study investigates how Mexican avocado exports to the
    United States affect local biodiversity and farmer livelihoods.
""")
print(result.formatted)
```

### Advisor parameters

```python
advisor = MetacouplingAssistant(
    adapter,
    temperature=0.7,          # Creativity (0.0=deterministic, 1.0=creative)
    max_tokens=None,           # Response length limit (None=model default)
    max_examples=2,            # Number of curated examples in the prompt
    verbose=False,             # Print diagnostic info during execution
    recommend_papers=False,    # Auto-append literature recommendations
    max_recommendations=5,     # Number of papers to recommend
    rag_mode="pre_retrieval",  # see "RAG modes" below
    rag_top_k=8,               # passages retrieved per query
)
```

### RAG modes (`rag_mode`)

The package supports two RAG integration modes:

- **`"pre_retrieval"` (default).** Retrieves corpus passages from the
  research description **before** calling the LLM, embeds them in the
  user message as a `<retrieved_literature>` XML block, and instructs
  the LLM to cite them inline as `[1]..[N]`. This is the standard RAG
  pattern and gives the LLM literature to ground its analysis in
  rather than relying purely on training memory. After the LLM
  responds, any out-of-range citation tokens (e.g., `[99]` when only
  8 passages were retrieved) are stripped with a logged warning.
- **`"post_hoc"` (alternative).** Generates the analysis from training
  memory first, then runs a keyword-overlap pass to stamp `[N]`
  citations onto sentences that match retrieved passages. Use this
  mode when downstream tooling expects citations to be assigned by
  post-hoc keyword matching rather than inline by the LLM:
  ```python
  advisor = MetacouplingAssistant(
      adapter,
      rag_corpus="journal_articles_2025",
      rag_mode="post_hoc",
  )
  ```

In pre_retrieval mode, `refine()` always re-retrieves using a labeled
merged query that combines the **original** research description with
the new refinement text:

```text
Original research question:
<your analyze() research_description>

Refinement request:
<your refine() additional_info>
```

The original research question is anchored at `analyze()` time and is
**never** overwritten by subsequent refines, so multi-turn
conversations stay anchored to the topic you started with.

> ⚠️ **Phase 1 limitation.** Citation numbering is turn-local: the
> same `[1]` may refer to different papers in turn 1 and turn 2,
> because each refine() pulls a fresh passage set. The system prompt
> tells the LLM to cite only from the most recent block, but this
> isn't perfect. Until Phase 2 lands turn-scoped markers, treat each
> turn's `SUPPORTING EVIDENCE FROM LITERATURE` block as the
> authoritative mapping for that turn's citations.

### What you get back: `AnalysisResult`

```python
result = advisor.analyze("My research about coffee trade...")

result.formatted     # str — Human-readable report (print this)
result.parsed        # ParsedAnalysis — Structured data for programmatic use
result.raw           # str — Unprocessed LLM response
result.turn_number   # int — Which conversation turn (1 for first analysis)
result.usage         # dict | None — Token usage (prompt_tokens, etc.)
```

### RAG-only mode (`coupling_analysis=False`)

Power users who already understand the metacoupling framework can
turn the framework-driven structural analysis off and use the advisor
purely as a literature-grounded Q&A engine over the bundled corpus
(plus optional web search):

```python
rag_advisor = MetacouplingAssistant(
    adapter,
    coupling_analysis=False,           # <-- toggle
    rag_corpus="journal_articles_2025",
    rag_top_k=8,
    web_search=False,                  # set True to also search the web
)

result = rag_advisor.analyze(
    "research status of China–Brazil soybean trade under metacoupling and telecoupling"
)

# Easiest: one print shows the answer + a bibliography of cited papers
print(result.formatted)

# Or access the parts individually:
print(result.answer)             # raw LLM output with [N] markers
for p in result.references:      # cited Paper objects, in cite order
    print(f"  [{p.key}] {p.title} — {p.authors} ({p.year})")
print(result.usage)
```

The result is a `RAGResult` (not `AnalysisResult`) with these fields:

| field | type | notes |
|---|---|---|
| `formatted` | `str` (property) | answer with sequential `[1]`, `[2]`, ... markers + a bibliography of cited papers — use this for `print()` |
| `answer` | `str` | LLM response with `[N]` markers as the LLM emitted them (possibly sparse, e.g. `[1]`, `[3]`); out-of-range markers already sanitized |
| `references` | `list[Paper]` | cited papers, dedup'd by key, in cite order |
| `retrieved_passages` | `list[RetrievalResult]` | all K passages shown to the LLM this turn |
| `web_sources` | `list[dict] \| None` | web hits if `web_search=True`, else `None` |
| `turn_number` | `int` | 1 for the first call; increments across turns |
| `usage` | `dict \| None` | token accounting |
| `raw` | `str` | the LLM response before citation sanitization |

`formatted` is computed on access (no caching) so it always reflects
the current state of the result. If the LLM emitted sparse citations
(e.g. it cited passages 1 and 3 but not 2), `formatted` re-numbers
them to `[1]`, `[2]` so the markers in the answer line up with the
bibliography entries.

**Multi-turn by default.** Each subsequent `analyze()` call appends
to a running conversation, so follow-ups work naturally:

```python
r1 = rag_advisor.analyze("Tell me about China–Brazil soybean trade.")
r2 = rag_advisor.analyze("What about its environmental impacts then?")
print(rag_advisor.conversation_turns)   # 2
rag_advisor.clear_history()             # reset and start over
```

Each turn runs a **fresh RAG retrieval** keyed off that turn's query,
so follow-ups get the most relevant passages for *their* specific
question rather than reusing turn 1's hits.

**`[N]` markers reset each turn.** A `[1]` in turn 2's answer refers
to turn 2's first passage, not turn 1's. The system prompt instructs
the LLM about this; if a user asks about something from a prior
turn, the LLM refers to it narratively rather than re-citing by
number.

**Framework-only options are silently disabled** when
`coupling_analysis=False`: `auto_map`, `recommend_papers`,
`rag_structured_extraction`, `web_structured_extraction`. A notice is
printed if any of those are set. All other options
(`web_search`, `rag_top_k`, `rag_backend`, `rag_min_score`,
`rag_max_chunks_per_paper`, `temperature`, `max_tokens`, `verbose`)
work in both modes.

**One advisor = one mode.** The flag is set at construction time and
not mutable afterward — create a separate `MetacouplingAssistant`
instance for each mode you need.

---

## 6. Refining an Analysis

After the initial analysis, you can have a multi-turn conversation to refine
specific aspects:

```python
# First analysis
result = advisor.analyze("My study on soybean trade between Brazil and China...")

# Refine with additional context
result2 = advisor.refine("Please also consider the role of smallholder farmers.")

# Focus on a specific component
result3 = advisor.refine(
    "I have data showing 3 main trade routes.",
    focus_component="flows"
)

# Check conversation state
print(advisor.turn_count)    # 3
print(len(advisor.history))  # 7 messages (system + 3 user + 3 assistant)

# Start fresh
advisor.reset()
```

### Valid focus components

`"systems"`, `"flows"`, `"agents"`, `"causes"`, `"effects"`, `"suggestions"`

---

## 7. Understanding the Output

### The formatted report

A typical `result.formatted` output contains these sections:

```
========================================================================
METACOUPLING FRAMEWORK ANALYSIS
========================================================================

COUPLING CLASSIFICATION
----------------------------------------
Telecoupling — The study involves interactions between Brazil (sending)
and China (receiving) through soybean trade...

SYSTEMS IDENTIFICATION
----------------------------------------
  [Sending]
    Brazil soybean production regions
    Human subsystem: Farmers, agribusiness corporations
    Natural subsystem: Cerrado biome, Amazon rainforest
    Geographic scope: Mato Grosso, Goiás, Bahia
  [Receiving]
    China consumer markets
    ...
  [Spillover]
    ...

FLOWS ANALYSIS
----------------------------------------
  1. [Matter] Brazil → China
     Soybeans exported...
  2. [Financial] China → Brazil
     Payment for soybean imports...

AGENTS
----------------------------------------
  - [Organizations / NGOs] World Trade Organization
  - [Governments / policymakers] Chinese Ministry of Commerce
  ...

CAUSES
----------------------------------------
  Economic:
    - Growing demand for animal feed in China
  Ecological / Biological:
    - Favorable climate for soybean cultivation
  ...

EFFECTS
----------------------------------------
  Sending System:
    - Deforestation of Amazon and Cerrado biomes
  Receiving System:
    - Improved food security
  ...

RESEARCH GAPS & SUGGESTIONS
----------------------------------------
  - Consider investigating spillover effects on...
  - Quantify carbon footprint of transportation flows

PERICOUPLING DATABASE VALIDATION
----------------------------------------
  Brazil (BRA) ↔ China (CHN): TELECOUPLED
  Note: LLM classification is consistent with the pericoupling database.

========================================================================
```

### Accessing structured data programmatically

```python
parsed = result.parsed

# Coupling type
print(parsed.coupling_classification)

# Systems (may be str or dict depending on LLM output)
sending = parsed.get_system_detail("sending", "name")
scope = parsed.get_system_detail("sending", "geographic_scope")

# Flows
for flow in parsed.flows:
    print(f"  {flow['category']}: {flow['direction']} — {flow['description']}")

# Agents
for agent in parsed.agents:
    print(f"  [{agent['level']}] {agent['name']}")

# Causes (dict of category → list of causes)
for category, items in parsed.causes.items():
    for item in items:
        print(f"  {category}: {item}")

# Effects (dict of system/type → list of effects)
for category, items in parsed.effects.items():
    for item in items:
        print(f"  {category}: {item}")

# Suggestions
for suggestion in parsed.suggestions:
    print(f"  - {suggestion}")

# Pericoupling validation results
if parsed.pericoupling_info:
    print(parsed.pericoupling_info.get("pair_results", ""))
    print(parsed.pericoupling_info.get("note", ""))
```

### Formatting options

```python
from metacouplingllm.output.formatter import AnalysisFormatter

formatter = AnalysisFormatter()

# Full report
print(formatter.format_full(parsed))

# Brief summary
print(formatter.format_summary(parsed))

# Single component
print(formatter.format_component(parsed, "flows"))
print(formatter.format_component(parsed, "systems"))
print(formatter.format_component(parsed, "causes"))

# Compare multiple analyses side by side
print(formatter.format_comparison([parsed1, parsed2, parsed3]))
```

---

## 8. Pericoupling Database

The package includes a curated database of 308 symmetric country pairs
classified as pericoupled (geographically adjacent) or telecoupled
(geographically distant), based on ISO 3166-1 alpha-3 country codes.

### Automatic integration

When you run `advisor.analyze()`, the pericoupling database is used in two
ways:

1. **Pre-LLM hint injection** — The system prompt tells the LLM which
   countries in your research are pericoupled vs. telecoupled, so it can
   classify coupling types more accurately.
2. **Post-LLM validation** — After parsing the LLM response, the package
   validates the classification against the database and flags any
   disagreements.

Both steps use the **focal (sending) country** as the anchor, only checking
pairs between the focal country and other detected countries.

### Standalone usage (no LLM needed)

```python
from metacouplingllm import lookup_pericoupling, is_pericoupled, get_pericoupled_neighbors

# Full lookup with details
result = lookup_pericoupling("Mexico", "United States")
print(result.pair_type)      # PairCouplingType.PERICOUPLED
print(result.sending_code)   # "MEX"
print(result.receiving_code) # "USA"
print(result.confidence)     # "database"

# Quick boolean check
print(is_pericoupled("Mexico", "USA"))     # True
print(is_pericoupled("Mexico", "Canada"))  # False (telecoupled)
print(is_pericoupled("Brazil", "China"))   # False (telecoupled)
print(is_pericoupled("Atlantis", "USA"))   # None  (unknown)

# Get all pericoupled neighbors of a country
neighbors = get_pericoupled_neighbors("China")
print(neighbors)
# {'RUS', 'IND', 'PAK', 'MNG', 'MMR', 'LAO', 'VNM', 'PRK', 'KOR', ...}

neighbors = get_pericoupled_neighbors("MEX")
print(neighbors)
# {'USA', 'GTM', 'BLZ'}
```

### Country name resolution

The package accepts country names in many formats:

```python
from metacouplingllm import resolve_country_code, get_country_name

# ISO codes
resolve_country_code("USA")         # "USA"
resolve_country_code("usa")         # "USA"

# Full names
resolve_country_code("Mexico")      # "MEX"
resolve_country_code("United States of America")  # "USA"

# Common aliases
resolve_country_code("UK")          # "GBR"
resolve_country_code("South Korea") # "KOR"
resolve_country_code("Russia")      # "RUS"

# Demonyms (adjective forms)
resolve_country_code("Mexican")     # "MEX"
resolve_country_code("Brazilian")   # "BRA"
resolve_country_code("Chinese")     # "CHN"

# Partial/substring matches
resolve_country_code("Ethiopian coffee regions")  # "ETH"

# Reverse: code → name
get_country_name("MEX")  # "Mexico"
get_country_name("GBR")  # "United Kingdom"
```

---

## 9. Literature Recommendations

The package bundles a BibTeX database of ~297 telecoupling and metacoupling
papers (filtered from 444 Web of Science entries). You can get relevant paper
recommendations based on keyword matching.

### Standalone usage

```python
from metacouplingllm import recommend_papers, format_recommendations

# From a text query
papers = recommend_papers("soybean trade Brazil China deforestation", max_results=5)
print(format_recommendations(papers))

# From a parsed analysis
result = advisor.analyze("My research about coffee trade...")
papers = recommend_papers(result.parsed, max_results=10)
print(format_recommendations(papers))
```

### Auto-append to every analysis

```python
advisor = MetacouplingAssistant(
    adapter,
    recommend_papers=True,     # Enable auto-append
    max_recommendations=5,     # Number of papers
)

result = advisor.analyze("My research about avocado trade...")
print(result.formatted)  # Includes a "RECOMMENDED LITERATURE" section at the end
```

### How recommendations work

The engine scores each paper in the database against your query:

| Match location | Points | Rationale |
|---|---|---|
| Author-assigned keywords | 3.0 per match | Most precise — curated by authors |
| Title words | 2.0 per match | Captures the paper's core topic |
| Abstract (first 200 chars) | 0.5 per match | Topic sentence, noisier signal |
| Citation count | up to 2.0 (log-scaled) | Highly-cited papers are more influential |

Papers are ranked by total score, then by citation count, then by year.

### Exploring the database

```python
from metacouplingllm import get_database_info

info = get_database_info()
print(info)
# {
#     'total_papers': 297,
#     'with_keywords': 284,
#     'with_abstracts': 294,
#     'year_min': 2013,
#     'year_max': 2026,
#     'total_citations': 10051,
# }
```

### Accessing individual paper data

```python
papers = recommend_papers("land use change", max_results=3)
for p in papers:
    print(f"Key:      {p.key}")
    print(f"Title:    {p.title}")
    print(f"Authors:  {p.authors}")
    print(f"Year:     {p.year}")
    print(f"Journal:  {p.journal}")
    print(f"DOI:      {p.doi}")
    print(f"Keywords: {p.keywords}")
    print(f"Cited by: {p.cited_by}")
    print()
```

---

## 10. World Map Visualization

Generate color-coded world maps showing coupling types relative to a focal
country. Requires the `viz` optional dependency.

```bash
pip install metacoupling[viz]
```

### Map colors

| Color | Meaning |
|---|---|
| Yellow-green (`#D4E79E`) | **Intracoupling** — The focal country itself |
| Green (`#4CAF50`) | **Pericoupling** — Geographically adjacent countries |
| Light blue (`#ADD8E6`) | **Telecoupling** — Geographically distant countries |
| Grey (`#D3D3D3`) | **N/A** — Countries not in the database |

### Database-only map (no LLM needed)

```python
from metacouplingllm import plot_focal_country_map

# By country name
fig = plot_focal_country_map("China")
fig.savefig("china_metacoupling.png", dpi=150, bbox_inches="tight")

# By ISO code
fig = plot_focal_country_map("MEX")

# With custom title
fig = plot_focal_country_map("Brazil", title="Brazil: Coupling Classification")

# With custom figure size
fig = plot_focal_country_map("USA", figsize=(20, 10))
```

### Analysis-based map (from LLM output)

```python
from metacouplingllm import plot_analysis_map

# After running an analysis
result = advisor.analyze("My study on avocado trade between Mexico and the US...")

# Generate map from the parsed analysis
fig = plot_analysis_map(result.parsed)
fig.savefig("avocado_trade_map.png", dpi=150, bbox_inches="tight")

# Specify which system role is the focal country
fig = plot_analysis_map(result.parsed, focal_role="sending")
```

### Custom colors

```python
from metacouplingllm.visualization.worldmap import CouplingColors, plot_focal_country_map

custom_colors = CouplingColors(
    intracoupling="#FF6B6B",   # Red for focal
    pericoupling="#4ECDC4",    # Teal for adjacent
    telecoupling="#45B7D1",    # Blue for distant
    na="#E0E0E0",              # Grey for unknown
)

fig = plot_focal_country_map("India", colors=custom_colors)
```

### Displaying in Jupyter

```python
# In Jupyter, figures display inline automatically
fig = plot_focal_country_map("China")
# The map appears directly in the notebook

# To save to file as well:
fig.savefig("output.png", dpi=150, bbox_inches="tight")
```

---

## 11. Advanced Usage

### Multi-turn conversation

```python
# Initial broad analysis
result1 = advisor.analyze("""
    I study the impacts of international tourism between
    Europe and Southeast Asia on coral reef ecosystems.
""")
print(result1.formatted)

# Drill into flows
result2 = advisor.refine(
    "I have data on tourist arrivals, money spent, and waste generated.",
    focus_component="flows"
)
print(result2.formatted)

# Add context about agents
result3 = advisor.refine(
    "Key agents include UNWTO, national tourism boards, and local dive operators."
)
print(result3.formatted)

# Check how many turns we've had
print(f"Conversation turns: {advisor.turn_count}")
```

### Comparing multiple analyses

```python
from metacouplingllm.output.formatter import AnalysisFormatter

# Analyze several research topics
topics = [
    "Soybean trade between Brazil and China",
    "Coffee trade between Ethiopia and Europe",
    "Timber trade between Indonesia and Japan",
]

results = []
for topic in topics:
    advisor.reset()
    r = advisor.analyze(topic)
    results.append(r.parsed)

# Side-by-side comparison
comparison = AnalysisFormatter.format_comparison(results)
print(comparison)
```

### Combining all features

```python
from metacouplingllm import (
    MetacouplingAssistant, OpenAIAdapter,
    recommend_papers, format_recommendations,
    plot_analysis_map, plot_focal_country_map,
    is_pericoupled, get_pericoupled_neighbors,
)
from openai import OpenAI

# Setup
client = OpenAI(api_key="sk-...")
advisor = MetacouplingAssistant(
    OpenAIAdapter(client, model="gpt-4o"),
    recommend_papers=True,
    max_recommendations=5,
)

# Analyze
result = advisor.analyze("""
    My study examines rare earth mineral trade between China
    and the United States and its effects on environmental
    degradation in Inner Mongolia.
""")
print(result.formatted)  # Includes pericoupling validation + literature recs

# Check pericoupling status
print(is_pericoupled("China", "USA"))  # False — they are telecoupled

# See China's neighbors
print(get_pericoupled_neighbors("CHN"))

# Generate map
fig = plot_analysis_map(result.parsed)
fig.savefig("rare_earth_map.png", dpi=150, bbox_inches="tight")

# Get more specific paper recommendations
papers = recommend_papers("rare earth mining environmental impact trade", max_results=10)
print(format_recommendations(papers))
```

---

## 12. API Reference

### Core Classes

| Class | Description |
|---|---|
| `MetacouplingAssistant` | Main entry point. Runs analyses and refinements via LLM. |
| `AnalysisResult` | Container for parsed + formatted + raw analysis output. |
| `ParsedAnalysis` | Structured data extracted from LLM response. |
| `AnalysisFormatter` | Formats ParsedAnalysis into various text representations. |

### LLM Adapters

| Class | Description |
|---|---|
| `OpenAIAdapter(client, model="gpt-4o")` | Wraps an `openai.OpenAI` instance. |
| `AnthropicAdapter(client, model="claude-sonnet-4-20250514")` | Wraps an `anthropic.Anthropic` instance. |
| `LLMClient` | Protocol — implement `chat()` for custom LLM providers. |

### Pericoupling Functions

| Function | Returns | Description |
|---|---|---|
| `lookup_pericoupling(a, b)` | `PericouplingResult` | Full lookup with pair type and codes. |
| `is_pericoupled(a, b)` | `bool \| None` | Quick check: True/False/None. |
| `get_pericoupled_neighbors(country)` | `set[str]` | All pericoupled ISO codes. |
| `resolve_country_code(name)` | `str \| None` | Resolve name/alias/demonym to ISO alpha-3. |
| `get_country_name(code)` | `str` | ISO alpha-3 code to canonical English name. |

### Literature Functions

| Function | Returns | Description |
|---|---|---|
| `recommend_papers(query, max_results=5)` | `list[Paper]` | Recommend papers by keyword matching. |
| `format_recommendations(papers)` | `str` | Format papers as readable text. |
| `get_database_info()` | `dict` | Summary statistics of the literature database. |

### Visualization Functions

| Function | Returns | Description |
|---|---|---|
| `plot_focal_country_map(country, ...)` | `Figure` | World map from country name/code. |
| `plot_analysis_map(parsed, ...)` | `Figure` | World map from LLM analysis result. |

### Enums

| Enum | Values |
|---|---|
| `CouplingType` | `INTRACOUPLING`, `PERICOUPLING`, `TELECOUPLING` |
| `SystemRole` | `SENDING`, `RECEIVING`, `SPILLOVER` |
| `FlowCategory` | `MATTER`, `ENERGY`, `INFORMATION`, `FINANCIAL`, `PEOPLE` |
| `AgentLevel` | `INDIVIDUALS_HOUSEHOLDS`, `FIRMS_TRADERS_CORPORATIONS`, `GOVERNMENTS_POLICYMAKERS`, `ORGANIZATIONS_NGOS`, `NON_HUMAN_AGENTS` |
| `CauseCategory` / `EffectCategory` | `ECONOMIC`, `POLITICAL_INSTITUTIONAL`, `ECOLOGICAL_BIOLOGICAL`, `TECHNOLOGICAL_INFRASTRUCTURAL`, `CULTURAL_SOCIAL_DEMOGRAPHIC`, `HYDROLOGICAL`, `CLIMATIC_ATMOSPHERIC`, `GEOLOGICAL_GEOMORPHOLOGICAL` |
| `PairCouplingType` | `PERICOUPLED`, `TELECOUPLED`, `UNKNOWN` |

---

## 13. Troubleshooting

### `SyntaxError: invalid syntax` when running `pip install`

You are running a shell command inside Python. In Jupyter notebooks or
Google Colab, prefix with `!` or `%`:

```python
!pip install metacoupling[openai]
# or
%pip install metacoupling[openai]
```

### `AuthenticationError` with OpenAI

You need a real API key from https://platform.openai.com/api-keys — replace
the placeholder string:

```python
# Wrong:
client = OpenAI(api_key="OPENAI_API_KEY")

# Correct:
client = OpenAI(api_key="sk-proj-abc123...")

# Best (Colab):
from google.colab import userdata
client = OpenAI(api_key=userdata.get('OPENAI_API_KEY'))
```

### `ModuleNotFoundError: No module named 'geopandas'`

Install the visualization dependencies:

```python
!pip install metacoupling[viz]
```

### `ImportError` or `AttributeError` related to `naturalearth`

The package downloads the needed World Bank boundary files on first use and
caches them locally. Ensure you have internet access for the first map
generation.

### Map shows `TclError: Can't find a usable tk.tcl`

Your environment does not have a display backend. Set the matplotlib
backend to `Agg` before importing:

```python
import matplotlib
matplotlib.use("Agg")

from metacouplingllm import plot_focal_country_map
fig = plot_focal_country_map("China")
fig.savefig("output.png", dpi=150, bbox_inches="tight")
```

In Jupyter notebooks this typically does not occur because the inline
backend is used automatically.

### Pericoupling validation shows unexpected country pairs

The package only validates pairs involving the **sending (focal) country**.
If you see unexpected pairs, the LLM may have identified a different
country as the sending system than you intended. Use `result.parsed.systems`
to check what the LLM detected.

### Literature recommendations seem unrelated

The recommendation engine uses keyword matching (not semantic similarity).
Try more specific terms in your research description. The engine searches
paper titles, author-assigned keywords, and the first 200 characters of
abstracts.

### `RuntimeError: Cannot refine before an initial analysis`

You must call `advisor.analyze()` before calling `advisor.refine()`. The
`refine()` method continues a multi-turn conversation that `analyze()`
starts.

---

## References

- Liu, J. (2017). Integration across a metacoupled world. *Ecology and
  Society*, 22(4), 29.
- Liu, J., et al. (2013). Framing sustainability in a telecoupled world.
  *Ecology and Society*, 18(2), 26.
