# MetacouplingLLM — User Manual

**Version 0.2.1**

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
14. [Web Search & Web-Sourced Evidence](#14-web-search--web-sourced-evidence)
15. [Scholar Export (Markdown + Word)](#15-scholar-export-markdown--word)
16. [Quantitative Indicators](#16-quantitative-indicators)
17. [LLM-Assisted Indicator Helpers](#17-llm-assisted-indicator-helpers)
18. [References](#18-references)

---

## 1. Installation

### Basic installation (no LLM provider yet)

```bash
pip install metacouplingllm
```

### With OpenAI support

```bash
pip install "metacouplingllm[openai]"
```

### With Anthropic (Claude) support

```bash
pip install "metacouplingllm[anthropic]"
```

### With Google Gemini support

```bash
pip install "metacouplingllm[gemini]"
```

### With xAI Grok support

```bash
pip install "metacouplingllm[grok]"
```

### With visualization support (world maps)

```bash
pip install "metacouplingllm[viz]"
```

### With quantitative indicators (pandas)

```bash
pip install "metacouplingllm[indicators]"
```

Pulls in `pandas`, the only extra dependency for the
`metacouplingllm.indicators` submodule (deterministic flow-share /
evenness / concentration math; see §16).

### With scholar export (python-docx)

```bash
pip install "metacouplingllm[export]"
```

Pulls in `python-docx`, required only for `result.to_docx(...)`.
`result.to_markdown(...)` and `result.abstract` work without this extra
(see §15).

### With web-search fallback (ddgs)

```bash
pip install "metacouplingllm[search]"
```

Pulls in `ddgs`, used only by the free DuckDuckGo fallback that
`web_search=True` runs when the provider's native search backend fails
or returns nothing. Native provider web search needs no extra, and a
slower stdlib-only DuckDuckGo fallback exists even without this
(see §14).

### Install everything

```bash
pip install "metacouplingllm[all]"
```

### Requirements

- Python 3.10 or higher
- Runtime dependencies installed with the base package: `numpy>=1.21` and
  `fastembed>=0.8` (default RAG embeddings; the BAAI/bge-base-en-v1.5 ONNX
  model is downloaded the first time a RAG query runs)
- An API key from OpenAI, Anthropic, Google Gemini, or xAI Grok (for
  LLM-based analysis)

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
adapter = OpenAIAdapter(client, model="gpt-4o")

# 2. Create an advisor
advisor = MetacouplingAssistant(
    adapter,
    web_search=True,
    web_search_max_results=10,
    web_structured_extraction=True,  # the default; validated countries + flows
    auto_map=True,
    rag_corpus="journal_articles_2025",
    rag_top_k=8,
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

`web_structured_extraction=True` is the default: whenever `web_search=True` and
the search returns results, the advisor performs an extra LLM pass over the web
results (title, URL and summary text) to extract validated receiving countries,
spillover countries, and map-ready flows, plus evidence cards that are added to
the analysis prompt. The validated payload is also stored on
`result.web_map_signals`. Pass `web_structured_extraction=False` to skip the
extra call.

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
2. **Flows** — Material (matter), energy, information, financial (capital),
   organism (species invasion, seed dispersal, animal migration), and people
   transfers
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
┌──────────────────────────┐
│    LLM Provider           │  ← OpenAI / Anthropic / Gemini /
│ (GPT-4o, Claude, Gemini,  │     Grok / custom backend
│  Grok) — native           │
│  web-search auto-wiring   │
└─────────┬────────────────┘
          │
          ▼
┌─────────────────────┐
│   Response Parser    │  ← Best-effort structured extraction
│  + Coupling          │  ← Post-LLM database validation
│    Validation        │
└─────────┬───────────┘
          │
          ▼
┌──────────────────────────┐
│   Formatted Output        │  ← Human-readable report
│  + Literature Recs        │  ← Optional paper recommendations
│  + World Map              │  ← Optional visualization
│  + Scholar Export         │  ← result.abstract / to_markdown / to_docx
│  + Quantitative           │  ← optional metacouplingllm.indicators
│    Indicators (sidecar)   │     on your own flow data
└──────────────────────────┘
```

Before the prompt is built, `analyze()` runs the optional web search
(`web_search=True`; its context goes into the system prompt) and, when a
RAG corpus is configured, retrieves literature passages (they go into the
user message). After the main call the response is parsed and checked
against the adjacency databases; the optional map-data extraction
(`auto_map=True`) and the abstract (`generate_abstract=True`, the
default) then each make one more model call. With every stage on, a run
makes up to five model calls: native web search, structured web
extraction, main analysis, map-data extraction and abstract.
`rag_structured_extraction=True` adds a sixth, and a failed or non-JSON
map-data reply is retried once.

---

## 4. LLM Setup

### Option A: OpenAI

```python
from openai import OpenAI
from metacouplingllm import MetacouplingAssistant, OpenAIAdapter

client = OpenAI(api_key="sk-your-api-key-here")
adapter = OpenAIAdapter(client, model="gpt-4o")
advisor = MetacouplingAssistant(adapter)
```

Supported models: `"gpt-4o"`, `"gpt-4o-mini"`, or any OpenAI chat
model identifier OpenAI's API accepts.

**Using Google Colab with secure API key storage:**

```python
from google.colab import userdata
api_key = userdata.get('OPENAI_API_KEY')

from openai import OpenAI
from metacouplingllm import MetacouplingAssistant, OpenAIAdapter

client = OpenAI(api_key=api_key)
advisor = MetacouplingAssistant(OpenAIAdapter(client, model="gpt-4o"))
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

Uses the `google.genai` SDK (PyPI package `google-genai`); pass a
`google.genai.Client`. The older `google.generativeai` package is not
supported.

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

Install with: `pip install "metacouplingllm[gemini]"`

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
server-side `web_search` tool (xAI's `/responses` endpoint; not yet
validated against the live API, see §14).

Install with: `pip install "metacouplingllm[grok]"`

### Option E: Custom LLM Client

Any object with a compatible `chat()` method will work — no inheritance
required:

```python
from metacouplingllm import MetacouplingAssistant, LLMResponse, Message

class MyLocalLLM:
    def chat(self, messages: list[Message], temperature=0.7, max_tokens=None):
        # Call your local model here. messages[0] is the system prompt, so
        # send every message, not only the last one.
        prompt = "\n\n".join(f"{m.role.upper()}: {m.content}" for m in messages)
        response_text = my_model.generate(prompt)
        return LLMResponse(content=response_text)

advisor = MetacouplingAssistant(MyLocalLLM())
```

When `web_search=True` and the client is **not** one of the four
built-in adapters, web search auto-falls back to DuckDuckGo (free, no
API key).

### Web-search auto-wiring

Each adapter auto-wires its **native** web-search backend when you set
`web_search=True` — no extra configuration needed:

| Adapter | Native backend (auto-wired) |
|---|---|
| `OpenAIAdapter` | `OpenAIWebSearchBackend` (`web_search` tool) |
| `AnthropicAdapter` | `AnthropicWebSearchBackend` (`web_search_20250305` tool) |
| `GeminiAdapter` | `GeminiWebSearchBackend` (Google Search grounding) |
| `GrokAdapter` | `GrokWebSearchBackend` (xAI's server-side `web_search` tool on `/responses`, see §14) |
| Custom client | DuckDuckGo fallback cascade (free, no API key; no backend class) |

All four native backends share the same rich prompt template, a strict
structured-output schema, the default domain blocklist (reddit.com,
quora.com, pinterest.com; the field is `blocked_domains` on the OpenAI,
Anthropic and Gemini backends and `excluded_domains` on the Grok backend)
and an output-token budget that scales with `web_search_max_results`.
Supranational unions are handled downstream of the backends, in the
structured web extraction, the map-data extraction and the map renderer,
for every provider. See §14 for details and configuration knobs.

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
    temperature=0.7,          # Creativity (0.0=deterministic, 1.0=creative);
                              #   ignored by gpt-5* models (always run at 1) and
                              #   claude-opus-4-7; main analysis call only
    max_tokens=None,           # Response length limit (None = the provider's default);
                               #   AnthropicAdapter sends 4096 because the Anthropic API
                               #   requires a limit, which can cut a long report short —
                               #   pass a larger value for Claude
    max_examples=2,            # Number of curated examples in the prompt
    verbose=False,             # Print diagnostic info during execution
    recommend_papers=False,    # Auto-append literature recommendations
    max_recommendations=5,     # Number of papers to recommend
    rag_top_k=8,               # passages retrieved per query
    de_facto_borders=True,     # Adjacency (hint, validation, map): fold disputed
                               #   land into its de-facto administrator (vs strict WB)
    coupling_standard="moderate",  # "stringent"/"moderate"/"lenient" — how
                               #   water-only borders count as adjacency
    generate_abstract=True,    # One extra model call per analyze()/refine() to write
                               #   result.abstract; False skips it
    trace=None,                # Write a run-trace folder per analyze()/refine():
                               #   None = on unless METACOUPLINGLLM_DISABLE_TRACE=1,
                               #   True/False force it on/off
    trace_dir=None,            # Where to write it (default: runs/<utc>_<slug>/)
)
```

`de_facto_borders` and `coupling_standard` set the adjacency policy for every
use the assistant makes of the bundled adjacency databases (§8): the pre-LLM
pericoupling hint in the system prompt, the **pericoupling validation block**
that cross-checks the LLM's coupling claims, and the automatic map
(`auto_map=True`, §10). They mirror the same-named arguments on the standalone
lookup functions: `coupling_standard="stringent"` drops every water-only border
(so a river-only neighbour validates as *telecoupled* rather than
*pericoupled*), `"lenient"` keeps them all, and `de_facto_borders=False` uses
the strict World Bank standard layer instead of the de-facto view. Defaults
(`True` / `"moderate"`) match the database defaults; see §8 for the full
semantics. Because the hint and the validation block use the same settings,
they agree on adjacency: Ohio and Ontario, for example, meet only across Lake
Erie, which has no fixed crossing, so with `coupling_standard="lenient"` both
list Ontario as a cross-border neighbour of Ohio, and with the default neither
does.

### Run tracing (`trace`, `trace_dir`)

Tracing is **on by default**. Each `analyze()` / `refine()` captures every
chat call made through the assistant's client (prompts, response, token usage,
duration) plus the pipeline intermediates (raw web results, RAG chunks, parsed
analysis, map data, formatted output) and run metadata (model, the git SHA of
the repository in the working directory — `unknown` outside one — and the
environment, including the package version), then writes a folder of
human-readable artifacts:

```text
runs/<utc-timestamp>_<query-slug>/turn1/
    00_run_config.md … 11_llm_call_abstract.md
    README.md
    map.png            # when a map was rendered
```

An `analyze()` followed by `refine()` calls writes `turn1/`, `turn2/`, … under
one session root. Each new `analyze()` starts a new session; in RAG-only mode
every call is an `analyze()` call, so with the default `trace_dir` each turn
gets its own `runs/<utc>_<slug>/turnK/`. The same trace is attached to the
result as `result.trace` (a `RunTrace`); `result.trace.out_dir` is the folder
on disk, and `result.trace.calls` is the list of captured `CallRecord`s.

```python
result = advisor.analyze("…")
print(result.trace.out_dir)          # runs/2026…_…/turn1
print(result.trace.total_tokens)     # input + output across all captured calls
```

- **Disable it** with `trace=False`, or globally with the
  `METACOUPLINGLLM_DISABLE_TRACE=1` environment variable. An explicit
  `trace=True` / `trace=False` overrides the environment variable.
- **Choose the location** with `trace_dir="my/path"` (the per-turn subfolders
  are written underneath it).
- The default `runs/` directory is **gitignored**; commit a specific run with
  `git add -f runs/<slug>/`.
- Tracing never breaks an analysis: if writing fails, the result is returned
  normally with `result.trace.out_dir = None`.

> **Note.** Two web-stage model calls bypass the traced client: the provider's
> native web search (when used) and the structured web extraction, which is
> sent to the unwrapped adapter so that it can use the provider's
> schema-constrained output mode. Only their results are kept, in `01` and `03`.

### What's in the RAG corpus

The RAG evidence corpus is a curated set of **421 telecoupling and
metacoupling papers**, shipped with the package as `Papers.zip` and indexed
on first use (embeddings by default, TF-IDF fallback).  It is split by what
can be redistributed, not by topic: **219 papers are indexed as full text**
and the other **202 as structured, paraphrased summaries** (metadata + key
findings, no source text) — those 202 being papers that cannot be
redistributed in full: 125 paywalled, and 77 not cleared for
redistribution (free to read without a redistribution licence, restrictively
licensed, preprint-only, or with open-access status not established).
Retrieval runs over both, so a summary-only paper can still
surface as supporting evidence.  See INTRODUCTION §4.3 and §6 for the
corpus composition.

> **The recommendation database is the BibTeX layer of this corpus.**  The
> 265-article BibTeX literature database used by `recommend_papers` /
> `get_database_info` (§9) is matched to 265 of these 421 papers; the other
> 156 have no BibTeX entry and are never recommended. RAG *grounds* the
> analysis with quoted passages; the recommendation database *suggests*
> further reading.

### How RAG citations work

When a RAG corpus is configured, the package retrieves corpus passages
from the research description **before** calling the LLM, embeds them
in the user message as a `<retrieved_literature turn="k">` XML block,
and instructs the LLM (via a citation-rules layer in the system
prompt) to cite them inline as `[Tk:N]` (`k` is the conversation
turn) — turn-scoped so the same token never changes meaning across
follow-ups.

**What the LLM sees.** Each user message includes a labelled XML
block with the top-k retrieved passages numbered 1..N (illustrative
text):

```xml
<retrieved_literature turn="1">
  <passage turn="1" id="1"
           paper_key="liu_2017_integration_across_a_metacoupled_world"
           authors="Liu" year="2017" section="Results" score="0.74">
    Brazilian soybean exports to China are primarily destined for
    the swine and poultry feed industry, with a fivefold increase
    in shipment volume between 2000 and 2015...
  </passage>
  <passage turn="1" id="2" paper_key="fearnside_2001_amazon"
           authors="Fearnside, P." year="2001" section="Discussion" score="0.68">
    Cerrado-to-cropland conversion accounts for the bulk of
    deforestation associated with soybean expansion...
  </passage>
</retrieved_literature>
```

An empty `<retrieved_literature turn="k"/>` self-closing tag means
retrieval ran but found nothing — the LLM is instructed to emit no
new current-turn citations rather than guess.

**What the LLM writes.** The LLM cites the numbered passages inline
as it writes:

```text
Brazil's soybean exports to China are dominated by feed-industry
demand [T1:1]. The land-use footprint of this trade is concentrated
in Cerrado-to-cropland conversion [T1:2], with secondary effects
on Amazon deforestation reported in recent satellite analyses
[T1:W2].
```

After the analysis, the package appends a `SUPPORTING EVIDENCE FROM
LITERATURE (turn k)` block that resolves each `[Tk:N]` marker back
to its source paper. Web sources use the `[Tk:Wn]` variant and are
rendered in a separate `WEB SOURCES (turn k)` block.

**What the sanitizer enforces.** After the LLM responds,
`sanitize_turn_citations` (in
`metacouplingllm.knowledge.citations`) scans the response and
strips any citation token the LLM should not have emitted:

- **Out-of-range tokens** — e.g. `[T1:99]` when only 8 passages
  were retrieved this turn
- **Forward references** — e.g. `[T9:1]` in turn 2 (no such turn
  has happened yet)
- **Bare tokens** — `[N]` or `[W1]` without the `Tk:` prefix

Out-of-range and forward-reference strips are reported as WARNING log
records (`metacouplingllm.knowledge.citations`: "Stripped N invalid
turn-scoped citation(s): ..."; in framework mode also
`metacouplingllm.core`: "MetacouplingAssistant sanitized N invalid
citation token(s) ...") so the gap is auditable; bare `[N]` / `[W1]`
tokens are stripped silently (no warning). Prior-turn back-references
(e.g. `[T1:3]` appearing in a turn-2 answer) are **kept** verbatim —
the LLM is encouraged to back-reference earlier evidence by copying
the original token.

`refine()` always re-retrieves using a labeled merged query that
combines the **original** research description with the new
refinement text:

```text
Original research question:
<your analyze() research_description>

Refinement request:
<your refine() additional_info>
```

The original research question is anchored at `analyze()` time and is
**never** overwritten by subsequent refines, so multi-turn
conversations stay anchored to the topic you started with.

### What you get back: `AnalysisResult`

```python
result = advisor.analyze("My research about coffee trade...")

result.formatted     # str — Human-readable report (print this)
result.parsed        # ParsedAnalysis — Structured data for programmatic use
result.raw           # str — LLM response after citation sanitization (invalid [Tk:N] /
                     #   [Tk:Wn] and bare [N] / [W1] tokens removed, whitespace
                     #   normalised)
result.turn_number   # int — Which conversation turn (1 for first analysis)
result.usage         # dict | None — Token usage (keys vary by provider)
```

> **Note:** the `usage` keys are provider-dependent: OpenAI and Grok
> report `prompt_tokens` / `completion_tokens` / `total_tokens`;
> Anthropic reports `input_tokens` / `output_tokens` (no total); Gemini
> reports `input_tokens` / `output_tokens` / `total_tokens` (the adapter
> renames the SDK's `*_token_count` fields). `usage` is `None` when the
> client reports none, so use `(result.usage or {}).get(...)` rather than
> assuming a key exists.

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
print(result.answer)             # LLM output with [Tk:N] markers
for p in result.references:      # cited Paper objects, in cite order
    print(f"  [{p.key}] {p.title} — {p.authors} ({p.year})")
print(result.usage)
```

The result is a `RAGResult` (not `AnalysisResult`) with these fields:

| field | type | notes |
|---|---|---|
| `formatted` | `str` (property) | answer with turn-scoped `[Tk:N]` / `[Tk:Wn]` markers + a bibliography of cited papers (and a web-sources block when relevant) — use this for `print()` |
| `answer` | `str` | LLM response with stable `[Tk:N]` / `[Tk:Wn]` markers as emitted; invalid tokens already stripped |
| `references` | `list[Paper]` | papers cited in the **current** turn, dedup'd by key, in cite order — prior-turn back-references are NOT included here |
| `retrieved_passages` | `list[RetrievalResult]` | all K passages shown to the LLM this turn |
| `web_sources` | `list[dict] \| None` | web hits if `web_search=True`, else `None` |
| `turn_number` | `int` | 1 for the first call; increments across turns |
| `usage` | `dict \| None` | token accounting |
| `raw` | `str` | the LLM response before citation sanitization |
| `retrieval_backend` | `str` | `"embeddings"` or `"tfidf"`, the backend that scored the passages |
| `reference_passage_ids` | `list[int]` | parallel to `references`: the passage number `N` of the `[Tk:N]` at which each paper was first cited |
| `trace` | `RunTrace \| None` | run trace of this turn (see Run tracing above); `None` when tracing is off |

`formatted` is computed on access (no caching) so it always reflects
the current state of the result. The LLM emits stable turn-scoped
tokens like `[T1:1]` and `[T1:3]` directly — no remapping is
performed, so each marker keeps the passage number the model used.
The bibliography lists one entry per paper, labelled with the first
passage cited from it; a later marker for another passage of the same
paper (for example `[T1:5]` after `[T1:1]`) has no entry of its own —
find it in `retrieved_passages`. Prior-turn back-references (e.g.
`[T1:3]` appearing in a turn-2 answer) are kept verbatim but excluded
from the current turn's references list.

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
question rather than reusing turn 1's hits. `refine()` is framework-only:
in RAG-only mode it raises `RuntimeError`, so ask follow-ups with further
`analyze()` calls.

**Citations are turn-scoped.** A `[T2:1]` in turn 2's answer refers
to turn 2's first passage, and a `[T1:3]` in turn 2's answer
back-references turn 1's third passage — both meanings remain stable
forever. The LLM may freely back-reference prior-turn evidence by
copying the original token. Bare `[N]` / `[W1]` tokens (no `Tk:`
prefix) are invalid and are silently stripped by the sanitizer.

**Framework-only options are ignored** when `coupling_analysis=False`:
`auto_map`, `recommend_papers`, `rag_structured_extraction`,
`web_structured_extraction`. A notice is printed at construction if any
of those four is set (`web_structured_extraction`, which is on by
default, only when `web_search=True`). `max_examples`,
`max_recommendations`, `generate_abstract`, `de_facto_borders`,
`coupling_standard`, `adm0_shapefile`, `adm1_shapefile`,
`web_structured_min_confidence` and `web_structured_max_targets` are
also ignored, without a notice. `rag_corpus`, `rag_papers_dir`,
`web_search`, `web_search_max_results`, `rag_top_k`, `rag_backend`,
`rag_min_score`, `rag_max_chunks_per_paper`, `temperature`,
`max_tokens`, `verbose`, `trace` and `trace_dir` work in both modes.

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

These are recommended values matching the report's section names —
`focus_component` is not validated; it is a free-form hint passed
verbatim into the refinement prompt, so any descriptive string works.

---

## 7. Understanding the Output

### The formatted report

A typical `result.formatted` output uses a **coupling-first numbered
layout**: §1 is the classification, §§2–4 are per-coupling-type
analysis blocks (2 = Intracoupling, 3 = Pericoupling,
4 = Telecoupling — each with N.1 Systems Identification, N.2 Flows
Analysis, N.3 Agents, N.4 Causes, N.5 Effects), §5 is Cross-coupling
Interactions, §6 Research Gaps, §7 Evidence Coverage. Coupling types
with no content are skipped, so the numbering can jump (e.g. 2 → 4 when
the analysis has no pericoupling section):

```
========================================================================
METACOUPLING FRAMEWORK ANALYSIS
========================================================================

1. Coupling Classification
----------------------------------------
- Intracoupling — soybean production and handling within Brazil...
- Telecoupling (primary) — the study involves interactions between
  Brazil (sending) and China (receiving) through soybean trade...

2. Intracoupling Analysis
----------------------------------------
  ...

4. Telecoupling Analysis
----------------------------------------
  4.1 Systems Identification
  [Sending System (distant)]
    Brazil soybean production regions
    Human subsystem: Farmers, agribusiness corporations
    Natural subsystem: Cerrado biome, Amazon rainforest
    Geographic scope: Mato Grosso, Goiás, Bahia
  [Receiving System (distant)]
    China consumer markets
    ...
  [Spillover System]
    ...

  4.2 Flows Analysis
  1. [Matter] Brazil → China
     Soybeans exported...
  2. [Capital] China → Brazil
     Payment for soybean imports...

  4.3 Agents
  - [Organizations / Ngos] World Trade Organization
  - [Governments / Policymakers] Chinese Ministry of Commerce
  ...

  4.4 Causes
  Economic:
    - Growing demand for animal feed in China
  Ecological / Biological:
    - Favorable climate for soybean cultivation
  ...

  4.5 Effects
  Ecological / Biological:
    - Deforestation of Amazon and Cerrado biomes
  Cultural / Social / Demographic:
    - Improved food security in receiving system
  ...

5. Cross-coupling Interactions
----------------------------------------
  ...

6. Research Gaps and Suggestions
----------------------------------------
  - Consider investigating spillover effects on...
  - Quantify carbon footprint of transportation flows

7. Evidence Coverage
----------------------------------------
The trade-volume claims are grounded in retrieved literature
[T1:2] and recent export statistics; spillover-system evidence
is thinner and partly relies on framework reasoning...

COUPLING DATABASE VALIDATION
----------------------------------------
  This block cross-checks the LLM's coupled-system
  claims against the bundled coupling databases
  (country adjacency).  Pairs are labeled PERICOUPLED
  (adjacent) countries or TELECOUPLED (distant) countries
  per the database.  Core subnational regions are
  subnational regions identified based on the analysis
  results and are provided for reference only.

  Focal System: Brazil (BRA)
    Core subnational regions: Mato Grosso, Para

  Telecoupled Countries:
    Brazil (BRA) ↔ China (CHN)

  Note: LLM classification is consistent with the coupling database.

========================================================================
```

Countries the LLM placed in the spillover role (such as the
`[Spillover System]` above) are not listed, because adjacency cannot
validate that indirect role; a group with no pairs is not printed. The
Note compares coupling types, not individual partners: if the database
shows a pericoupled pair but the classification does not mention
pericoupling (or shows only telecoupled pairs but the classification
does not mention telecoupling), the Note is printed as
`Note (country): ...` and ends with "Consider revising."

The `Core subnational regions:` sub-line appears only for
national-scope queries (user named a country, not a state) and
lists the LLM-mentioned subnational regions inside the focal
country.  For subnational-scope queries (e.g. "avocado in
Jalisco, Mexico"), the `Focal System:` line shows the ADM1
region explicitly instead — e.g.
`Focal System: Jalisco (MEX014), Mexico (MEX)` — no
`Core subnational regions:` sub-line appears, and the group
labels change to `Pericoupled Countries/Subnational Regions:`
and `Telecoupled Countries/Subnational Regions:` to reflect
that pairs can mix country and ADM1 partners.

In subnational mode, foreign-partner classification is done at
the **region** scale: a country is pericoupled iff the focal
ADM1 region has a cross-border neighbor inside it.  An interior
focal state (e.g. Jalisco) therefore has zero pericoupled
foreign countries, while a border state (e.g. Chihuahua) keeps
the adjacent foreign country pericoupled.  National mode
classifies at the country scale (does the focal country border
the partner?).

### Accessing structured data programmatically

Structured data lives in per-coupling-type `CouplingSection` objects;
the `iter_*` accessor methods walk all of them for you (each yields
plain dicts/strings, optionally filtered with `coupling_type=`):

```python
parsed = result.parsed

# Coupling type
print(parsed.coupling_classification)

# System details (always returned as str; empty string when missing)
sending = parsed.get_system_detail("sending", "name")
scope = parsed.get_system_detail("sending", "geographic_scope")

# Flows (dicts; keys e.g. category/direction/description)
for flow in parsed.iter_flow_entries():
    print(f"  {flow.get('category')}: {flow.get('direction')} — {flow.get('description')}")

# Agents (dicts; keys e.g. level/name)
for agent in parsed.iter_agent_entries():
    print(f"  [{agent.get('level')}] {agent.get('name')}")

# Causes and effects — (coupling_type, category, item) triples
for ctype, category, item in parsed.iter_category_items("causes"):
    print(f"  [{ctype}] {category}: {item}")

for ctype, category, item in parsed.iter_category_items("effects"):
    print(f"  [{ctype}] {category}: {item}")

# Research gaps / suggestions (plain list attribute)
for suggestion in parsed.research_gaps:
    print(f"  - {suggestion}")

# Coupling-database validation results.  Country-level (always
# populated when ≥2 countries are detected) and subnational ADM1
# (populated only for subnational-scope queries — i.e. the user
# named a region that resolves to an ADM1 code; skipped in
# national mode) live in separate fields and can both be present
# at once.
if parsed.country_pericoupling_info:   # country-level
    print(parsed.country_pericoupling_info.get("pair_results", ""))
    print(parsed.country_pericoupling_info.get("note", ""))
if parsed.pericoupling_info:            # subnational (ADM1)
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

# Single component — valid names: "classification" / "coupling",
# "intracoupling", "pericoupling", "telecoupling",
# "cross_coupling" / "interactions", "research_gaps" / "suggestions"
print(formatter.format_component(parsed, "classification"))
print(formatter.format_component(parsed, "telecoupling"))
print(formatter.format_component(parsed, "suggestions"))

# Compact comparison: one summary block per analysis (classification line,
# active coupling types, flow count, research-gap count)
print(formatter.format_comparison([parsed1, parsed2, parsed3]))
```

### Scholar-ready abstract (`result.abstract`)

Every `AnalysisResult` has a one-paragraph scholar abstract in
`result.abstract`, generated by an extra model call after the main analysis:

```python
result = advisor.analyze("Brazil soybean exports to China...")
print(result.abstract)
# "This analysis applies the metacoupling framework to Brazil's soybean
#  export system to China.  Brazil's Cerrado and Amazon agricultural
#  regions are identified as the sending system and Chinese feed and oil
#  markets as the receiving system..."
```

The abstract is meant to be **standalone**: the generation prompt tells
the model not to use the `[Tk:N]` / `[Tk:Wn]` citation markers of the
main report. The reply is not post-processed apart from stripping code
fences, so look for stray markers. It is model-generated text: check it
against the analysis and its sources before putting it in a manuscript.
It is generated by one extra model call when each result is built, i.e.
at every `analyze()` and every `refine()` turn, and cached on that
result. Pass `generate_abstract=False` to `MetacouplingAssistant(...)`
to skip the call; `result.abstract` is an empty string if the call is
skipped or fails.

### Markdown + Word export (`result.to_markdown` / `result.to_docx`)

```python
result = advisor.analyze("Brazil soybean exports to China...")

# Manuscript-ready Markdown (no extra dependency)
result.to_markdown("brazil_soybean.md")

# Word document with headings / bold sub-field labels / per-category
# Flows + Causes + Effects subsections; requires metacouplingllm[export]
result.to_docx("brazil_soybean.docx")
```

The exporters carry over:

- a **title** auto-derived from the research description
- the scholar **abstract**
- bold sub-field labels (each system's Human subsystem / Natural
  subsystem / Geographic scope / Description labels, and the agent
  `level`); each system's role and name is a heading, and flow
  `direction` is plain text
- per-category **Flows** subsections (Matter, Capital, Information,
  Energy, People, Organisms)
- per-category **Causes** + **Effects** subsections matching the eight
  fixed categories
- **RAG evidence** + **web sources** blocks
- multi-bullet **§1 classification** rendered as proper bullet
  paragraphs in the docx export, and every system role line promoted
  to a **§N.1.K subheading**

### Evidence coverage note (`result.parsed.evidence_coverage_note`)

Every framework analysis requests a §7 **Evidence Coverage**
self-assessment (2–5 short paragraphs) summarising **what evidence
was available** (retrieved literature, web sources when
`web_search=True`), **what coverage gaps remain**, and **whether the
analysis relied on training memory** to fill those gaps.  Surface it
alongside the formatted report so reviewers can see where the analysis
is well grounded and where it is thin (it is the model's own assessment:
check it against the cited sources):

```python
print(result.parsed.evidence_coverage_note)
# "Web search returned recent USDA and ABIOVE export figures (2023-2025)
#  with strong coverage of the Brazil-China leg.  Coverage of spillover
#  effects on Argentine producers is thinner; the analysis flags this
#  gap rather than fabricating numbers."
```

---

## 8. Pericoupling Database

The package includes a curated country-pair database with 326 adjacent
pairs (the `lenient` view with `de_facto_borders=True`: 323 pairs that touch
in the World Bank standard layer plus 3 de-facto pairs across disputed
land); under the default adjacency settings (`de_facto_borders=True`,
`coupling_standard="moderate"`) **321** of them are exposed as pericoupled
(geographically adjacent) — the moderate standard drops 5 water-only pairs
with no fixed crossing, and the stringent standard leaves 307.
All other pairs are telecoupled (geographically distant). Based on
current ISO 3166-1 alpha-3 country codes (World Bank Official
Boundaries, 2026-05-14 release; see `data/PROVENANCE.md`). A second,
subnational database connects 3,374 first-level administrative (ADM1)
regions in 196 countries through border pairs (with `de_facto_borders=True`:
8,462 under `lenient`, 8,026 under `moderate`, 7,638 under `stringent`);
regions are identified by World Bank codes such as `MEX014`.

### Automatic integration

When you run `advisor.analyze()`, the pericoupling database is used in two
ways:

1. **Pre-LLM hint injection** — At most one reference-only hint is added to
   the system prompt (it says that adjacency alone does not prove a
   coupling), computed with the assistant's adjacency settings
   (`de_facto_borders`, `coupling_standard`; §5). If the
   research description names a subnational region that resolves to a World
   Bank ADM1 code (for example "Michigan"), the hint lists that region's
   same-country and cross-border neighbors from the ADM1 database.
   Otherwise, if it names countries, the hint gives the database verdict
   (pericoupled or telecoupled) for each pair formed by the first-named
   country and each other named country; when only one country is named, it
   lists that country's pericoupled neighbors instead. A description that
   names no country or region, or a lone country with no pericoupled
   neighbor (an island nation), gets no hint.
2. **Post-LLM validation** — After parsing the LLM response, the package
   looks up each pair between the focal system and the other named systems
   (skipping systems the model gave the spillover role) and shows the
   database verdict for each pair. It adds a "Consider revising" note only
   when a partner is pericoupled in the database but the classification
   does not mention pericoupling, or when all partners are telecoupled but
   it does not mention telecoupling; individual partner labels are not
   checked.

Both steps work from one focal system rather than from every pair. Before
the LLM call, that system is the first subnational region found in the
research description or, if there is none, the first country named (assumed
to be the focal one). After it, the validation anchors on the sending (or
focal) system of the parsed analysis (when your query names a subnational
region, on the focal region found in the analysis) and shows only that
system's pairs with the other systems found (partner countries and, for a
focal region, other mentioned regions); systems the model placed in the
spillover role are skipped. For a focal region, a partner country counts as
pericoupled only if the region has a cross-border neighbor in it (§7).

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
# {'RUS', 'IND', 'PAK', 'MNG', 'MMR', 'LAO', 'VNM', 'PRK', 'KAZ', ...}

neighbors = get_pericoupled_neighbors("MEX")
print(neighbors)
# {'USA', 'GTM', 'BLZ'}
```

### Adjacency standards (disputed borders & water-separated pairs)

The country-level and the ADM1 lookup functions accept two orthogonal
toggles (defaults shown):

- **`de_facto_borders=True`** — fold each disputed tract, whole, into the
  administrator that Natural Earth (v5.1.2) records for the largest part of
  it (so China–Pakistan, Israel–Syria and Morocco–Mauritania are adjacent);
  this represents de-facto connectivity and takes no position on
  sovereignty; pass `False` for the strict WB standard-layer view.
- **`coupling_standard="moderate"`** — for pairs that share **only** a
  river/lake border, `moderate` keeps a pair only if a completed fixed
  crossing links the two units, `stringent` drops every water-only pair, and
  `lenient` keeps all. A completed fixed crossing is a road or rail bridge,
  causeway, dam-top road or tunnel (not a ferry, ford or footbridge, and not
  one still under construction); a finished bridge on a politically closed
  border still counts.

`MetacouplingAssistant` (§5) and the map functions (§10) take the same two
settings; the assistant applies them to the pre-LLM hint, the post-LLM
validation block and the automatic map.

```python
# Romania <-> Moldova share only rivers, the Prut and a short Danube reach (bridges exist):
is_pericoupled("Romania", "Moldova")                                # True  (moderate default)
is_pericoupled("Romania", "Moldova", coupling_standard="stringent") # False (water-only)
# DR Congo <-> Central African Republic share only rivers, no bridge:
is_pericoupled("COD", "CAF")                                        # False (moderate default)
is_pericoupled("COD", "CAF", coupling_standard="lenient")           # True
# Lakes follow the same logic — DR Congo <-> Tanzania meet only across
# Lake Tanganyika (no fixed crossing):
is_pericoupled("COD", "TZA")                                        # False (moderate default)
is_pericoupled("COD", "TZA", coupling_standard="lenient")           # True
```

The same `coupling_standard` argument is accepted by the ADM1 functions
(`lookup_adm1_pericoupling`, `is_adm1_pericoupled`, `get_adm1_neighbors`,
`get_cross_border_neighbors`).  See `data/PROVENANCE.md`,
`docs/METHODS_adjacency.md` §8, and `docs/BRIDGE_CLASSIFICATION_METHODOLOGY.md`.

`resolve_adm1_code` also consults a bundled **English-exonym alias table**
(`data/adm1_aliases.csv`, 1,163 validated aliases for 872 regions across 136
countries) before its name/substring strategies, so common English
names resolve without the exact World Bank spelling — `"Bavaria"` → `DEU002`
(Bayern), `"Tuscany"` → `ITA016` (Toscana), `"Andalusia"` → `ESP002`.  The
table is additions-only and deterministically validated; a name that is
ambiguous or denotes a different place (e.g. `"New Mexico"` with
`country="MEX"`) returns `None` rather than a confident wrong answer.  See
`data/PROVENANCE.md` for the generation and validation methodology.

A name that is a country's ISO code, name or alias (`resolve_country_code`
with `substring=False`, below) returns `None` unless `country` is given,
because it is read as the country: `"Georgia"`, `"Niger"`, `"Wales"`,
`"México"`, or `"Est"`, which is Estonia's ISO code.  The same holds when
the name stands among connector or administrative words, which the matcher
ignores (`"of Mexico"`, `"region of Mexico"`, `"State of Mexico"`, `"Niger
State"`): pass `country` for these.  A name with any other word besides a
country's name is a region's and resolves without `country`: `"New South
Wales"` → `AUS004`, `"Mexico City"` → `MEX009`, `"Lesser Poland"` →
`POL006`.

When no strategy matches a name, `resolve_adm1_code` removes one trailing
administrative designator and resolves the rest: `"Narok County"` →
`KEN033`, `"Kagoshima Prefecture"` → `JPN017`, `"Rangpur Division"` →
`BGD007`.  Region, province, state and oblast are removed for any country.
County, prefecture, division, department, district, governorate and
voivodeship are removed only for countries whose first-level units carry
them, so `"Washington County"` stays `None` rather than becoming the State of
Washington.  The result is also refused when the rest names another unit of
the same country (`"Almaty Province"`: `"Almaty"` is Almaty City, the
province a separate unit) or matches several regions (`"La Paz Department"`
without `country`).

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

# Accents, names in the country's own language, World Bank spellings
resolve_country_code("México")            # "MEX"
resolve_country_code("Brasil")            # "BRA"
resolve_country_code("Congo, Dem. Rep.")  # "COD"
resolve_country_code("Kyrgyz Republic")   # "KGZ"

# Partial/substring matches
resolve_country_code("Ethiopian coffee regions")  # "ETH"
resolve_country_code("New South Wales")   # "GBR" (finds "Wales")

# Whole names only
resolve_country_code("New South Wales", substring=False)  # None
resolve_country_code("the Gambia", substring=False)       # "GMB"

# Reverse: code → name
get_country_name("MEX")  # "Mexico"
get_country_name("GBR")  # "United Kingdom"
```

---

## 9. Literature Recommendations

The package bundles a BibTeX database of 265 empirical telecoupling
and metacoupling journal articles (2013–2026, filtered from a larger
Web of Science collection).  Call `get_database_info()` for the live count if
the corpus drifts.  You can get relevant paper recommendations
ranked by author keywords, title words, full-text relevance and citation
count (see "How recommendations work").

> This 265-article recommendation database is the BibTeX layer of the
> 421-paper RAG evidence corpus (§5, "What's in the RAG corpus"): every one
> of the 265 articles is matched to a corpus paper, while the other 156
> corpus papers have no BibTeX entry and are never returned by
> `recommend_papers`. The BibTeX database drives `recommend_papers`
> (suggested reading; its full-text score queries the RAG index); the corpus
> supplies quoted evidence passages during analysis.

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
| Full-text relevance (RAG index over bundled paper texts) | up to 3.0 (cosine similarity × 3.0) | Matches the paper body, not just the front matter (embeddings by default, TF-IDF as the fallback) |
| Citation count | up to 2.0 (log-scaled) | Highly-cited papers are more influential |

Papers are ranked by total score, then by citation count, then by year.

### Exploring the database

Live values as of v0.2.1; call `get_database_info()` for
current counts:

```python
from metacouplingllm import get_database_info

info = get_database_info()
print(info)
# {
#     'total_papers': 265,
#     'with_keywords': 249,
#     'year_min': 2013,
#     'year_max': 2026,
#     'total_citations': 7626,
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
pip install "metacouplingllm[viz]"
```

### Map colors

| Color | Meaning |
|---|---|
| Yellow-green (`#D4E79E`) | **Intracoupling** — The focal country itself |
| Green (`#4CAF50`) | **Pericoupling** — Geographically adjacent countries |
| Light blue (`#ADD8E6`) | **Telecoupling** — Geographically distant countries |
| Grey (`#D3D3D3`) | **N/A** — Countries not colored by the database/analysis |
| Hatched grey (`#BFBFBF`, `///`) | **Disputed / Indeterminate** — World Bank disputed or indeterminate territories, overlaid on every map |

On **database-only** maps every country in the adjacency database is
colored (all non-adjacent countries show as telecoupled). On
**analysis-based** maps (next subsections) coloring is
analysis-driven: only countries identified from the analysis (and, on
automatic maps, from web extraction) are colored, and everything else stays
grey. The two routes differ in detail. The automatic map (`auto_map=True`,
stored on `result.map`) colors the focal country (or focal ADM1 region), the
receiving countries found by its map-data extraction call and the validated
receiving targets from web extraction; spillover countries stay grey (NA) by
design. A direct `plot_analysis_map(result.parsed)` call instead colors
every country that a text scan finds in the parsed systems, flows, coupling
classification, cross-coupling interactions, research gaps, causes and
effects (spillover systems included), and colors web-extraction targets only
if you pass them as `extra_mentioned_countries=`. If the map-data
extraction fails or returns no focal country, `result.map` falls back to
that same text scan.
Map adjacency follows the two adjacency settings of §8: the plot functions
take `de_facto_borders` and `coupling_standard` as keyword arguments
(defaults `True` / `"moderate"`), and the automatic map uses the assistant's
settings (§5).

### Database-only map (no LLM needed)

```python
from metacouplingllm import plot_focal_country_map

# By country name
fig = plot_focal_country_map("China")
fig.savefig("china_metacoupling.png", dpi=150, bbox_inches="tight")
# First render downloads + caches the World Bank ADM0 basemap
# (internet required once; cached afterwards)

# By ISO code
fig = plot_focal_country_map("MEX")

# With custom title
fig = plot_focal_country_map("Brazil", title="Brazil: Coupling Classification")

# With custom figure size
fig = plot_focal_country_map("USA", figsize=(20, 10))

# Under another adjacency standard (§8): Romania, which borders Moldova
# only along rivers, shows as telecoupled
fig = plot_focal_country_map("Moldova", coupling_standard="stringent")
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

Analysis-based maps also draw **flow arrows** extracted from the
parsed analysis, color-coded by flow category with their own legend
entries (matter, capital, information, energy, people, organisms).

```python
# Arrows default to the flows in the parsed analysis; pass your own to override them
fig = plot_analysis_map(
    result.parsed,
    flows=[{"category": "matter", "direction": "Mexico → United States"}],
)
```

### Automatic maps (`auto_map=True`)

`auto_map` is off by default. When it is on, each `analyze()` / `refine()`
turn makes an extra model call to extract the map data, draws a map and
stores it on `result.map`: an ADM1 (subnational) map
(`plot_focal_adm1_map`) when your query names a subnational region (by name
or ADM1 code, e.g. "Michigan" or "USA023"), otherwise the country-level map
described above. Maps are drawn only at country or ADM1 scale: a city,
watershed or protected-area study is mapped through its parent country or
ADM1 region when the model resolves one, and otherwise gets no map.

`result.map_notice` is a short status note that is also appended to
`result.formatted`. It says which kind of map was drawn or, when
`result.map` is `None`, why: no country or ADM1 focal system could be
resolved (for example a city, watershed or protected-area study), the map
dependencies (`geopandas`, `matplotlib`) are not installed, or the map could
not be rendered. It stays `None` when `auto_map=False`.

### Subnational (ADM1) map

`plot_focal_adm1_map` takes a World Bank ADM1 code (look one up with
`resolve_adm1_code("Michigan")`) and colors the focal region, the regions
that border it and distant systems with the same coupling colors.

```python
from metacouplingllm import plot_focal_adm1_map

fig = plot_focal_adm1_map("USA023")   # Michigan
fig.savefig("michigan_adm1.png", dpi=150, bbox_inches="tight")
# Needs the World Bank Admin 1 GeoPackage: found in the local cache, otherwise
# downloaded on first use (or pass shapefile="path/to/Admin 1.gpkg")
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

Figures returned by `plot_focal_country_map`, `plot_analysis_map` and
`plot_focal_adm1_map` stay open in pyplot and display automatically.
`result.map` (from `auto_map=True`) is closed on purpose and does not
display by itself: end the cell with `result.map` or call
`display(result.map)`.

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

# Compact comparison: one summary block per analysis (classification line,
# active coupling types, flow count, research-gap count)
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

### Combining qualitative + quantitative analysis

Once you have *both* an LLM-driven case-study analysis **and** your own
flow dataset, you can ground the qualitative narrative in the
deterministic indicators:

```python
import pandas as pd
from metacouplingllm.indicators import summarize_metacoupling

# Qualitative — LLM-driven framework analysis
result = advisor.analyze("Brazil soybean exports to China...")
print(result.formatted)

# Quantitative — your own classified flow data
flows = pd.read_csv("brazil_soybean_flows_classified.csv")
summary = summarize_metacoupling(flows)
print(summary)        # one row per system, 18 columns: focal_system_id, F_I/F_P/F_T/F_total,
                      # IFS/PFS/TFS, MFE, IFCI/PFCI/TFCI, n_I/n_P/n_T, ENP_I/ENP_P/ENP_T
```

See §16 for the indicator math and the Brazil-soybean worked example,
and §17 for the optional LLM helpers that wrap study setup, data
validation, ambiguous-edge classification, results interpretation, and
methods-section drafting around the deterministic core.

---

## 12. API Reference

### Core Classes

| Class | Description |
|---|---|
| `MetacouplingAssistant` | Main entry point. Runs analyses and refinements via LLM. |
| `OpenAIAdapter` / `AnthropicAdapter` / `GeminiAdapter` / `GrokAdapter` | LLM-provider adapters; each auto-wires its native web-search backend. See "LLM Adapters" below for signatures. |
| `LLMClient` | Protocol any custom client can implement (just a `chat()` method). |
| `AnalysisResult` | Container for parsed + formatted + raw + abstract + exporter output. |
| `ParsedAnalysis` | Structured data extracted from LLM response (incl. `evidence_coverage_note`). Import from `metacouplingllm.llm.parser` (not the top-level package). |
| `AnalysisFormatter` | Formats ParsedAnalysis into various text representations. Import from `metacouplingllm.output.formatter` (not the top-level package). |
| `RAGResult` | Container returned when `coupling_analysis=False` (RAG-only Q&A mode). |
| `RunTrace` / `CallRecord` | Run-trace records attached as `result.trace` (see "Run tracing" in §5): a `RunTrace` holds the run's `calls` (one `CallRecord` per traced chat call), `total_tokens`, `git_sha`, `env` and `out_dir`. |
| `Paper` | One entry of the literature database (`key`, `title`, `authors`, `year`, `journal`, `doi`, `keywords`, `cited_by`); returned by `recommend_papers`. |

### MetacouplingAssistant

`MetacouplingAssistant(llm_client, ...)` takes the parameters below; every
parameter except `llm_client` is optional.

| Parameter | Default | Description |
|---|---|---|
| `llm_client` | required | Any `LLMClient`: an adapter or your own object with a `chat()` method. |
| `temperature` | `0.7` | Sampling temperature of the main analysis call (§5). |
| `max_tokens` | `None` | Response-length limit; `None` uses the provider's default (§5). |
| `max_examples` | `2` | Curated examples placed in the system prompt. |
| `verbose` | `False` | Print diagnostic messages while running. |
| `recommend_papers` | `False` | Append a "RECOMMENDED LITERATURE" section to `result.formatted` (§9). |
| `max_recommendations` | `5` | Number of papers to recommend. |
| `auto_map` | `False` | Draw a map for each result and store it on `result.map` (§10). |
| `adm1_shapefile` | `None` | Path to a World Bank Admin 1 GeoPackage for ADM1 maps; skips the cache and the download (§13). |
| `adm0_shapefile` | `None` | Path to a World Bank Admin 0 GeoPackage for country maps; skips the cache and the download (§13). |
| `rag_papers_dir` | `None` | Folder of your own markdown papers to use as the RAG corpus instead of the bundled one; cannot be combined with `rag_corpus`. |
| `rag_corpus` | `None` | Built-in RAG corpus: `"journal_articles_2025"` (exported as `JOURNAL_ARTICLES_2025`). With neither this nor `rag_papers_dir`, RAG is off (§5). |
| `rag_top_k` | `8` | Passages retrieved per analysis. |
| `rag_backend` | `"auto"` | `"auto"` (embeddings, TF-IDF fallback), `"embeddings"` or `"tfidf"`. |
| `rag_max_chunks_per_paper` | `3` | Most passages one paper can contribute to a retrieval. |
| `rag_structured_extraction` | `False` | With RAG, one extra model call that extracts systems and flows from the retrieved passages into `result.structured_supplement`. |
| `web_search` | `False` | Search the web before the analysis (§14). |
| `web_search_max_results` | `10` | Web hits requested (§14). |
| `web_structured_extraction` | `True` | With `web_search=True`, an extra model call that extracts validated countries, flows and evidence cards (§14). |
| `web_structured_min_confidence` | `0.7` | Minimum confidence for a web-extracted country or flow to be kept (§14). |
| `web_structured_max_targets` | `6` | Cap on web-extracted receiving and spillover targets (§14). |
| `rag_min_score` | `None` | Minimum retrieval score; `None` applies a floor for the active backend (0.60 for embeddings, 0.01 for TF-IDF). |
| `coupling_analysis` | `True` | `False` switches to RAG-only Q&A, which returns a `RAGResult` (§5). |
| `de_facto_borders` | `True` | Disputed-territory setting of the adjacency lookups: the prompt hint, the pericoupling validation and the automatic map (§8). |
| `coupling_standard` | `"moderate"` | Water-only-border setting of the adjacency lookups: the prompt hint, the pericoupling validation and the automatic map (§8). |
| `generate_abstract` | `True` | One extra model call per result to write `result.abstract`; `False` skips it. |
| `trace` | `None` | Write a run-trace folder per `analyze()` / `refine()`: `None` = on unless `METACOUPLINGLLM_DISABLE_TRACE=1`, `True` / `False` force it (§5). |
| `trace_dir` | `None` | Where trace folders are written (default `runs/<utc>_<slug>/`). |

Methods and properties:

| Method / property | Returns | Description |
|---|---|---|
| `analyze(research_description)` | `AnalysisResult` (`RAGResult` when `coupling_analysis=False`) | Runs an analysis. In framework mode it starts a new conversation; in RAG-only mode it answers the question and adds a turn to the running conversation. |
| `refine(additional_info, focus_component=None)` | `AnalysisResult` | Continues the conversation with more information. Framework mode only: it raises `RuntimeError` before `analyze()` and in RAG-only mode. |
| `reset()` | `None` | Clears the framework conversation. |
| `clear_history()` | `None` | Clears both the framework and the RAG-only conversations. |
| `history` | `list[Message]` | Copy of the framework conversation's messages (system prompt included). |
| `turn_count` | `int` | Completed framework turns. |
| `conversation_turns` | `int` | Completed RAG-only turns (0 in framework mode). |

### AnalysisResult properties + exporters

| Method / property | Returns | Description |
|---|---|---|
| `result.formatted` | `str` | Human-readable report (the default `print()` surface). |
| `result.parsed` | `ParsedAnalysis` | Structured fields for programmatic access. |
| `result.raw` | `str` | The response text that was parsed: invalid `[Tk:N]` / `[Tk:Wn]` markers and bare `[N]` / `[W1]` tokens have already been stripped and whitespace normalised. The verbatim provider response is the `response_content` of the `main_analysis` call in `result.trace.calls` (tracing on). |
| `result.turn_number` | `int` | 1 for the first turn; increments across `refine()`. |
| `result.usage` | `dict \| None` | Token-usage accounting (provider-dependent). |
| `result.map` | `Figure \| None` | Generated map (when `auto_map=True`). |
| `result.abstract` | `str` | Scholar-ready one-paragraph abstract. |
| `result.trace` | `RunTrace \| None` | Per-run trace: `out_dir` (artifact folder, or `None` if writing failed), `calls` (`list[CallRecord]`: prompts, response, usage, duration), `total_tokens`, `git_sha`, `env`; `None` when tracing is off (`trace=False`). See "Run tracing" in §5. |
| `result.web_map_signals` | `dict \| None` | Validated countries, flows and evidence cards from the structured web extraction (`web_search=True`); `None` without it. |
| `result.evidence_coverage_note` | `str` | Copy of `result.parsed.evidence_coverage_note`. |
| `result.map_notice` | `str \| None` | Why a map was or was not produced (`auto_map=True`; §10). |
| `result.structured_supplement` | `dict \| None` | Output of `rag_structured_extraction=True`; `None` when it is off or failed. |
| `result.flow_parse_warnings` | `list[dict]` | Flow directions the automatic map's text-scan fallback could not resolve (keys `direction`, `category`, `reason`); empty otherwise. |
| `result.suggested_followup_queries` | `list[str]` | Follow-up web-search queries from the web-extraction call; empty without web search. |
| `result.to_markdown(path=None)` | `str` | Manuscript-ready Markdown; writes to `path` if given. |
| `result.to_docx(path=None)` | `Path` | Word document; defaults to `./metacoupling_report.docx` and returns the written `Path` (requires `metacouplingllm[export]`). |

### LLM Adapters

| Class | Description |
|---|---|
| `OpenAIAdapter(client, model="gpt-4o")` | Wraps an `openai.OpenAI` instance. |
| `AnthropicAdapter(client, model="claude-sonnet-4-20250514", extended_thinking=None)` | Wraps an `anthropic.Anthropic` instance; `extended_thinking` (e.g. `{"type": "enabled", "budget_tokens": 10000}`) is forwarded as the `thinking` request parameter and is not sent when `None`. |
| `GeminiAdapter(client, model="gemini-2.5-flash")` | Wraps a `google.genai.Client` instance. |
| `GrokAdapter(client, model="grok-3")` | Wraps an `openai.OpenAI` instance pointed at `https://api.x.ai/v1`. |
| `LLMClient` | Protocol — implement `chat()` for custom LLM providers. |
| `Message(role, content)` | One chat message (`role` is `"system"`, `"user"` or `"assistant"`); what `LLMClient.chat()` receives. |
| `LLMResponse(content, usage={}, tool_uses=None)` | What `LLMClient.chat()` returns: the reply text, a token-`usage` dict and optional `tool_uses`. |

### Web-Search Backends (auto-wired when `web_search=True`)

| Class | Native API |
|---|---|
| `OpenAIWebSearchBackend` | OpenAI `web_search` tool |
| `AnthropicWebSearchBackend` | Anthropic `web_search_20250305` tool |
| `GeminiWebSearchBackend` | Google Search grounding |
| `GrokWebSearchBackend` | xAI `/responses` server-side `web_search` tool (see §14) |
| `search_web()` built-in fallback | `ddgs` → `duckduckgo_search` → stdlib cascade (function-based; no backend class) |

### Pericoupling Functions (country level)

| Function | Returns | Description |
|---|---|---|
| `lookup_pericoupling(country_a, country_b, de_facto_borders=True, coupling_standard="moderate")` | `PericouplingResult` | Full lookup with pair type and codes. |
| `is_pericoupled(country_a, country_b, de_facto_borders=True, coupling_standard="moderate")` | `bool \| None` | Quick check: True/False/None. |
| `get_pericoupled_neighbors(country, de_facto_borders=True, coupling_standard="moderate")` | `set[str]` | All pericoupled ISO codes. |
| `resolve_country_code(name_or_code, *, substring=True)` | `str \| None` | Resolve name/alias/demonym to ISO alpha-3; `substring=False` accepts only a whole code, name or alias. |
| `get_country_name(code)` | `str` | ISO alpha-3 code to canonical English name. |

`de_facto_borders` toggles the disputed-territory overlay;
`coupling_standard` (`"stringent"`/`"moderate"`/`"lenient"`) controls
water-only pairs — see §8 "Adjacency standards".

### ADM1 (Subnational) Pericoupling Functions

| Function | Returns | Description |
|---|---|---|
| `lookup_adm1_pericoupling(adm1_a, adm1_b, de_facto_borders=True, coupling_standard="moderate")` | `Adm1PericouplingResult` | Full ADM1 lookup with pair type. |
| `is_adm1_pericoupled(adm1_a, adm1_b, de_facto_borders=True, coupling_standard="moderate")` | `bool \| None` | Quick adjacency check at the ADM1 scale. |
| `get_adm1_neighbors(adm1_code, de_facto_borders=True, coupling_standard="moderate")` | `set[str]` | All ADM1 neighbors (domestic + cross-border). |
| `get_cross_border_neighbors(adm1_code, de_facto_borders=True, coupling_standard="moderate")` | `set[str]` | ADM1 neighbors in other countries only. |
| `get_adm1_codes_for_country(country)` | `set[str]` | All ADM1 codes inside the given country (name or ISO alpha-3). |
| `get_adm1_info(adm1_code)` | `dict \| None` | Dict with keys `name` (canonical ADM1 name), `country_name`, `iso_a3` and `wb_region` (World Bank region code, e.g. `LCR`); `None` if the code is unknown. |
| `get_adm1_country(adm1_code)` | `str \| None` | ISO alpha-3 country of the ADM1 region. |
| `resolve_adm1_code(name, country=None)` | `str \| None` | Resolve a region name to its ADM1 code. Handles unaccented forms and native-script letters NFKD cannot decompose — `"Łódź"` → `POL003`, `"Đắk Lắk"` → `VNM016` — and **English exonyms / alternative spellings** via a bundled alias table — e.g. `"Bavaria"` → `DEU002`, `"Tuscany"` → `ITA016` — and a trailing administrative designator, removed when the full name does not resolve — `"Narok County"` → `KEN033`; possessives and hyphenated compounds are stripped/split by the validator's text cleaner before resolution. Returns `None` rather than guessing when a name is ambiguous or padded with a meaningful extra word. |
| `get_adm1_aliases(adm1_code)` | `list[str]` | English-exonym alias keys recorded for an ADM1 region (e.g. `"DEU002"` → `["bavaria"]`); empty list if none. |

### Literature Functions

| Function | Returns | Description |
|---|---|---|
| `recommend_papers(query, *, max_results=5)` | `list[Paper]` | Rank papers by keywords/title + full-text relevance + citations; `query` may be a string or a `ParsedAnalysis`. |
| `format_recommendations(papers)` | `str` | Format papers as readable text. |
| `get_database_info()` | `dict` | Summary statistics of the literature database. |

### RAG, citation and web-evidence helpers

Lower-level building blocks behind the RAG, citation and web-evidence steps,
exported for custom pipelines.

| Name | Description |
|---|---|
| `JOURNAL_ARTICLES_2025` | `"journal_articles_2025"`, the name of the bundled RAG corpus (`rag_corpus=JOURNAL_ARTICLES_2025`). |
| `RAGEngine(papers_dir, max_chunk_words=250, overlap_words=50, verbose=False, backend="auto")` | Retrieval engine over a folder of markdown papers: call `load()` once, then `retrieve(query, top_k=5, min_score=None, max_chunks_per_paper=3)`, which returns a `list[RetrievalResult]`. |
| `RetrievalResult(chunk, score)` / `TextChunk` | A retrieved passage with its score / the passage itself (`paper_key`, `paper_title`, `authors`, `year`, `section`, `text`, `chunk_index`). |
| `format_evidence(results, anchor_text="", backend="tfidf", turn=1, *, max_chars=None)` | Render retrieved passages as the human-readable evidence block with `[Tk:N]` headers. |
| `TURN_CITATION_PATTERN` | Compiled regex matching the turn-scoped citation markers `[Tk:N]` (literature) and `[Tk:Wn]` (web). |
| `sanitize_turn_citations(text, turn_passage_counts, turn_web_counts, current_turn)` | Strip out-of-range citation markers; returns `(sanitized_text, invalid_markers)`. |
| `extract_turn_cited_ids(text)` | The `(k, kind, N)` tuples of the turn-scoped markers used in `text`. |
| `format_web_context(results, turn=1)` | Format `search_web()` results as the prompt's web-context section. |
| `annotate_web_citations(formatted, web_results, min_keyword_overlap=3, min_overlap_ratio=0.25, turn=1)` | Add inline `[Tk:Wn]` citations to the lines of formatted text that match a web result. |
| `extract_web_map_signals(query, results, llm_client, *, min_confidence=0.7, max_targets=6, temperature=0.0, max_tokens=8192)` | One model call that extracts validated map-ready countries and flows from web results (the source of `result.web_map_signals`); returns a dict or `None`. |
| `format_web_map_signals_context(signals)` | Format those signals as a prompt-ready section. |

### Visualization Functions

| Function | Returns | Description |
|---|---|---|
| `plot_focal_country_map(focal_country, ...)` | `Figure` | World map from country name/code. |
| `plot_analysis_map(parsed_analysis, ...)` | `Figure` | World map from LLM analysis result. |
| `plot_focal_adm1_map(focal_adm1, ...)` | `Figure` | Subnational (ADM1) map for the focal region + its neighbors. |

### Quantitative Indicator Functions (`metacouplingllm.indicators`)

| Function | Returns | Description |
|---|---|---|
| `classify_coupling(edges, focal_id, adjacency, ...)` | `pd.DataFrame` | Add I/P/T column to an edge table (optional `llm_client=` for ambiguous edges). |
| `build_adjacency(pairs, level="adm0", ...)` | `pd.DataFrame` | Fill the `adjacent` flag (`1`/`0`/`<NA>`) from the bundled pericoupling DB; `level="adm0"` (countries) or `"adm1"` (WB subnational), with `de_facto_borders=` / `coupling_standard=` passthrough. |
| `compute_flow_shares(data, ...)` | `pd.DataFrame` | Metacoupled Flow Shares (IFS / PFS / TFS) per focal system. |
| `compute_mfe(data, ...)` | `pd.DataFrame` | Normalised Shannon entropy across coupling types; input is the **shares table from `compute_flow_shares`** (needs IFS/PFS/TFS columns), not the raw edge table. |
| `compute_mfci(data, ...)` | `pd.DataFrame` | Normalised HHI per coupling type (long-format: one row per system × coupling type). |
| `summarize_metacoupling(data, ...)` | `pd.DataFrame` | One-shot combined indicator table (this is what produces the wide IFCI/PFCI/TFCI + ENP_* columns — ENP_I/P/T = Equivalent Number of Intra-/Peri-/Telecoupled Partners). |

### LLM-Assisted Indicator Helpers (`metacouplingllm.indicators`)

| Function | Returns | Description |
|---|---|---|
| `define_study(description, *, llm_client)` | `(dict, LLMTrace)` | Natural-language → structured study config. |
| `check_inputs(data_summary, sample_rows=None, *, llm_client)` | `(dict, LLMTrace)` | Validate user data; flag missing inputs / unit issues. `data_summary` needs at least `"columns"` and `"row_count"`. |
| `classify_ambiguous_edges(edges, study_config, *, llm_client)` | `(pd.DataFrame, LLMTrace)` | Classify edges the deterministic pass couldn't resolve. |
| `interpret_results(results, *, llm_client, audience="academic")` | `(str, LLMTrace)` | Plain-language interpretation of an indicator **DataFrame** (`"academic"` / `"general"` / `"policy"`). |
| `write_methods(indicator_spec, *, llm_client)` | `(str, LLMTrace)` | Manuscript-ready Methods text with formulas + standard citations. |
| `LLMTrace` | dataclass | `timestamp_utc`, `model`, `prompt_version`, `system_prompt`, `user_prompt`, `raw_response`, `usage`. |

### Enums

| Enum | Values |
|---|---|
| `CouplingType` | `INTRACOUPLING`, `PERICOUPLING`, `TELECOUPLING` |
| `SystemRole` | `SENDING`, `RECEIVING`, `SPILLOVER` |
| `FrameworkComponent` | `SYSTEMS`, `FLOWS`, `AGENTS`, `CAUSES`, `EFFECTS` |
| `FlowCategory` | `CAPITAL`, `ENERGY`, `INFORMATION`, `MATTER`, `ORGANISMS`, `PEOPLE` |
| `AgentLevel` | `INDIVIDUALS_HOUSEHOLDS`, `FIRMS_TRADERS_CORPORATIONS`, `GOVERNMENTS_POLICYMAKERS`, `ORGANIZATIONS_NGOS`, `NON_HUMAN_AGENTS` |
| `CauseCategory` / `EffectCategory` | `ECONOMIC`, `POLITICAL_INSTITUTIONAL`, `ECOLOGICAL_BIOLOGICAL`, `TECHNOLOGICAL_INFRASTRUCTURAL`, `CULTURAL_SOCIAL_DEMOGRAPHIC`, `HYDROLOGICAL`, `CLIMATIC_ATMOSPHERIC`, `GEOLOGICAL_GEOMORPHOLOGICAL` |
| `PairCouplingType` | `PERICOUPLED`, `TELECOUPLED`, `UNKNOWN` |
| `Adm1PairType` | `PERICOUPLED`, `TELECOUPLED`, `SAME_REGION`, `UNKNOWN` |

---

## 13. Troubleshooting

### `SyntaxError: invalid syntax` when running `pip install`

You are running a shell command inside Python. In Jupyter notebooks or
Google Colab, prefix with `!` or `%`:

```python
!pip install "metacouplingllm[openai]"
# or
%pip install "metacouplingllm[openai]"
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

### `ImportError: Visualization requires geopandas, matplotlib`

The map functions (`plot_focal_country_map`, `plot_analysis_map`,
`plot_focal_adm1_map`) need the `[viz]` extra (geopandas and matplotlib;
the message names whichever of the two is missing):

```python
!pip install "metacouplingllm[viz]"
```

### `ImportError: metacouplingllm.indicators requires pandas`

The quantitative indicators submodule needs pandas. Install
the `[indicators]` extra:

```python
!pip install "metacouplingllm[indicators]"
```

### `ImportError: render_docx requires the optional python-docx dependency`

The Word exporter needs `python-docx`. Install the
`[export]` extra:

```python
!pip install "metacouplingllm[export]"
```

`result.to_markdown(...)` and `result.abstract` work without this
extra.

### `FileNotFoundError` for the World Bank boundary data / map download failures

The package downloads the needed World Bank boundary GeoPackages on
first use and caches them locally. Ensure you have internet access for
the first map generation; subsequent renders use the cache. When the file
can be neither found nor downloaded, country maps raise `Country-level maps
require World Bank ADM0 boundary data ...` and ADM1 maps raise `No World
Bank Admin 1 GeoPackage found in cache ...`.

If the machine is offline or the download is blocked, download the World
Bank Official Boundaries GeoPackages once (Admin 0 and Admin 1; World Bank
Data Catalog, dataset 0038272) and pass their paths:

```python
fig = plot_focal_country_map("China", adm0_shapefile="path/to/Admin 0_all_layers.gpkg")
fig = plot_focal_adm1_map("MEX001", shapefile="path/to/Admin 1.gpkg")

advisor = MetacouplingAssistant(
    adapter,
    auto_map=True,
    adm0_shapefile="path/to/Admin 0_all_layers.gpkg",
    adm1_shapefile="path/to/Admin 1.gpkg",
)
```

Downloads are cached under `~/.cache/metacoupling` (set
`METACOUPLING_CACHE_DIR` to use another folder).

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

The package only validates pairs involving the **focal system**: the
sending (or focal) system of the parsed analysis, which is the focal
subnational region when your query names one. If you see unexpected
pairs, the LLM may have identified a different country as the sending
system than you intended. Use
`list(result.parsed.iter_system_entries())` to check what the LLM
detected.

### A subnational region the LLM clearly mentioned isn't in the validator block

The validator can only show regions whose name `resolve_adm1_code` can
match against the bundled ADM1 database. Recognition covers unaccented
forms (`Michoacan` → `Michoacán de Ocampo`), possessives (`Michoacán's`),
hyphenated compounds (`Michoacán-Jalisco`) and English exonyms through the
alias table (`Bavaria` → `Bayern`, `Tuscany` → `Toscana`). When the
research description names a subnational region, LLM-mentioned ADM1
partners are emitted as `pair_results` lines, which the formatter buckets
into the COUPLING DATABASE VALIDATION block. A region can still be missing
from the block in these cases:

- **Country-scope query.** The subnational (ADM1) part of the validator
  runs only when the research description passed to `analyze()` names a
  subnational region: a capitalised region name that `resolve_adm1_code`
  resolves and that is not itself a country's name (`New South Wales`
  counts, `Mexico` does not), or an ADM1 code such as `MEX016`. Adjacent
  capitalised words are read as one name, so write `Michoacán, Mexico`
  rather than `Michoacán Mexico`. For any other
  (country-scope) query, mentioned regions of the focal country appear only
  on the `Core subnational regions:` line, never as pair lines, and regions
  of other countries are not listed.
- **Spillover role.** In subnational-scope queries, regions the model lists
  as spillover systems are deliberately not classified (adjacency cannot
  show whether a place plays that indirect role).
- **Country not named.** When the analysis names other countries but not
  the region's own country, the region is dropped as a likely false match.
- **Non-canonical names.** The DB uses official names (e.g.
  `Michoacán de Ocampo`). Short or colloquial
  forms work via the substring / folded fallback (which also
  transliterates letters NFKD cannot decompose, e.g. `Łódź`,
  `Đắk Lắk`), a trailing designator is removed when the full name
  fails (`Michoacan State`, `Narok County`), and many English
  exonyms resolve through the alias table, but truly different
  names (e.g. local short names not in the gazetteer or the alias table)
  won't.
- **Region mentioned only in `name` or `geographic_scope` fields.**
  By design `_extract_mentioned_adm1_from_text` skips those two fields
  because they often hold echo-back enumerations from the prompt's
  pericoupling hint. Move the substantive mention into a `description`
  or flow `direction` to surface it.
- **Region buried in a hedge phrase** like `"such as Michoacán"` or
  `"especially Michoacán"`. The cleaner's hedge-marker bailout filters
  these to avoid false positives from speculative language.

To diagnose, inspect the parsed systems and flows
(`list(result.parsed.iter_system_entries())` +
`list(result.parsed.iter_flow_entries())`) for where the region is
mentioned, and try
`from metacouplingllm.knowledge.adm1_pericoupling import resolve_adm1_code`
to confirm whether the resolver finds it at all.

### Literature recommendations seem unrelated

The engine ranks papers by author-assigned keywords, title words,
full-text relevance from the bundled RAG index (embeddings by default,
with a TF-IDF fallback when fastembed or the precomputed vectors are
unavailable), and citation count — see §9 "How recommendations
work". Try more specific terms in your research description so the
keyword and full-text components have something to bite on.

### `RuntimeError: Cannot refine before an initial analysis`

You must call `advisor.analyze()` before calling `advisor.refine()`. The
`refine()` method continues a multi-turn conversation that `analyze()`
starts. In RAG-only mode (`coupling_analysis=False`) `refine()` is not
available and always raises this error; ask follow-up questions with
further `analyze()` calls instead (see §5, RAG-only mode).

---

## 14. Web Search & Web-Sourced Evidence

The package can ground its analysis in real-time web search results. Each
LLM adapter auto-wires its **native** web-search backend; custom clients,
and any run in which the native backend raises or returns no usable
results, fall back to the built-in DuckDuckGo cascade. All four native
backends share the same prompt template and strict structured-output
schema, so the calling code is identical across providers; their
backend-level options differ (see "Configuration knobs").

### Enabling web search

```python
advisor = MetacouplingAssistant(
    OpenAIAdapter(client, model="gpt-4o"),
    web_search=True,                  # turn on web search
    web_search_max_results=10,        # number of web hits requested (default 10)
    web_structured_extraction=True,   # validated countries + flows (structured web extraction; on by default)
)
```

When structured extraction runs, the validated payload is surfaced as
an **output** field at `result.web_map_signals` (it is not a
constructor setting).

Set `web_search=True` and the right backend is selected automatically
from the adapter type — see §4 LLM Setup, "Web-search auto-wiring".

### Backend matrix

| Adapter | Backend | Native API |
|---|---|---|
| `OpenAIAdapter` | `OpenAIWebSearchBackend` | OpenAI `web_search` tool with `tool_choice="required"` |
| `AnthropicAdapter` | `AnthropicWebSearchBackend` | Claude `web_search_20250305` tool + `submit_results` strict-output tool |
| `GeminiAdapter` | `GeminiWebSearchBackend` | Google Search grounding |
| `GrokAdapter` | `GrokWebSearchBackend` | xAI `/responses` server-side `web_search` tool, through the `openai` SDK: the strict JSON schema in `text.format`, `reasoning={"effort": ...}`, the xAI-only `max_turns` in `extra_body`, and the domain list in the tool's `filters` |
| Custom client | DuckDuckGo fallback | `ddgs` → `duckduckgo_search` → stdlib cascade (function-based; no backend class) |

Auto-wiring reuses **your adapter's model** for the search call; the
backend dataclasses' own model defaults only apply if you instantiate
a backend directly.

### Configuration knobs

| Setting | Default | Notes |
|---|---|---|
| `web_search` | `False` | Master switch (constructor kwarg). |
| `web_search_max_results` | `10` | Number of web hits requested. Scales `max_output_tokens` automatically. |
| `web_structured_extraction` | `True` | Runs a second, strict-JSON LLM pass over the web results to extract validated receiving countries, spillover countries, and map-ready flows, plus evidence cards for the analysis prompt. Runs only when `web_search=True` and the search returns results; pass `False` to skip it. |
| `web_structured_min_confidence` | `0.7` | Minimum confidence for a country or flow from the structured web extraction to be kept in the validated payload. |
| `web_structured_max_targets` | `6` | Cap on the validated receiving systems and, separately, the spillover systems kept (6 each by default); validated map-ready flows are capped at this value + 2 (8 by default). |
| `blocked_domains` | `["reddit.com", "quora.com", "pinterest.com"]` | Field of the backend dataclasses (not an assistant or adapter kwarg): `blocked_domains` on the OpenAI, Anthropic and Gemini backends (Gemini filters the returned URLs client-side), `excluded_domains` (at most 5 entries) on `GrokWebSearchBackend`, where an `allowed_domains` list, if given, replaces it (xAI accepts only one of the two). Low-authority domains are excluded **out of the box**. `MetacouplingAssistant` always auto-wires a backend with these defaults and has no parameter to change them; to use other values, instantiate the backend and call `search_web(query, max_results, backend=OpenAIWebSearchBackend(client, blocked_domains=[]))` yourself. |
| `search_context_size`, `return_token_budget` | `"high"` / `"default"` | Fields of `OpenAIWebSearchBackend` — `"high"` deliberately maximises evidence depth. `MetacouplingAssistant` always auto-wires a backend with these defaults and has no parameter to change them; to use other values, instantiate the backend and call `search_web(query, max_results, backend=OpenAIWebSearchBackend(client, search_context_size="medium"))` yourself. |

The validated structured-extraction payload appears at
`result.web_map_signals` (an output field, not a setting). The same pass
proposes 3-5 follow-up web-search queries, available as
`result.suggested_followup_queries` and appended to `result.formatted`
under "Suggested follow-up web searches".

With `auto_map=True`, the map-data extraction call (§10) also receives the
web hits: up to 100 of them, each summary truncated to 2,500 characters,
so country names that appear late in a long summary are not cut off.

### Citation grammar

Web hits are labelled `[Tk:W1]`, `[Tk:W2]`, ... — the `W` prefix
distinguishes web sources from literature `[Tk:1]` / `[Tk:2]`, and `k`
is the conversation turn so prior-turn references stay stable across
`refine()` calls. Web search (and the structured extraction) runs once,
inside `analyze()`; `refine()` does not search again, so in later turns
web sources are cited by their original `[T1:Wn]` tokens and listed under
`WEB SOURCES (turn 1)`.

### Supranational flow handling

When the analysis touches a supranational union (EU, ASEAN, USMCA,
NAFTA), the package:

1. **Structured web extraction** lets the LLM name a bloc (European
   Union/EU, ASEAN, USMCA, NAFTA) as the country or flow endpoint when
   a source names only the bloc, and tells it to prefer the member
   states (omitting the bloc) when the source names them; the package
   then expands the bloc to its member ISO codes (27, 10, 3 and 3
   member states for the EU, ASEAN, USMCA and NAFTA).
2. **Map renderer** dissolves the member states into one region and
   fills it as a whole: the pericoupling colour when the focal country
   borders at least one member, otherwise the telecoupling colour.

### Evidence coverage note

Every framework analysis, with or without web search, is instructed to
close with a §7 Evidence Coverage self-assessment of 2-5 short
paragraphs. The model weighs all the evidence it was given (retrieved
literature and, when present, web sources) and says which claims rest on
several sources, on one weak source, or only on training-data knowledge.
The text is stored in `result.parsed.evidence_coverage_note` (mirrored on
`result.evidence_coverage_note`; empty when the model left §7 out):

```python
print(result.parsed.evidence_coverage_note)
# "Web search returned recent USDA and ABIOVE export figures (2023-2025)
#  with strong coverage of the Brazil-China leg.  Coverage of spillover
#  effects on Argentine producers is thinner; the analysis flags this
#  gap rather than fabricating numbers."
```

This is the model's own self-assessment of its evidence base. It helps
reviewers, but it is model-generated, so check it against the cited
sources and the run trace (`result.trace`).

### Strict structured output across all four providers

The structured web-extraction call uses each provider's native
strict-output mode so the validated payload is guaranteed to match
the expected schema:

| Adapter | Strict-output mode |
|---|---|
| OpenAI / Grok | `response_format = {"type": "json_schema", ...}` |
| Anthropic | `tools = [submit_web_map_signals]` + `tool_choice = {"type": "tool", "name": "submit_web_map_signals"}` |
| Gemini | `response_schema` + `response_mime_type = "application/json"` |

The web-search call itself uses a separate `submit_results` tool (see the
backend matrix).

Falls back to a defensive JSON-object extractor only when the strict
path returns malformed output.

---

## 15. Scholar Export (Markdown + Word)

Once you have an `AnalysisResult`, three exporter surfaces help you
turn it into something a co-author or reviewer can read:

- `result.abstract` — a one-paragraph scholar abstract
- `result.to_markdown(path=None)` — manuscript-ready Markdown
- `result.to_docx(path=None)` — Word document with
  headings; defaults to `./metacoupling_report.docx` and returns the
  written `Path`

Markdown export works with the core install. Word export needs
`metacouplingllm[export]` (python-docx).

### End-to-end example

```python
from openai import OpenAI
from metacouplingllm import (
    JOURNAL_ARTICLES_2025,
    MetacouplingAssistant,
    OpenAIAdapter,
)

client = OpenAI(api_key="sk-...")
advisor = MetacouplingAssistant(
    OpenAIAdapter(client, model="gpt-4o"),
    web_search=True,
    web_search_max_results=10,
    web_structured_extraction=True,
    rag_corpus=JOURNAL_ARTICLES_2025,
    rag_top_k=8,
    recommend_papers=True,
)

result = advisor.analyze("""
    My research examines how Mexican avocado exports to the
    United States affect local biodiversity and farmer livelihoods.
""")

print(result.formatted)        # full report with citations
print(result.abstract)         # one-paragraph scholar abstract
result.to_markdown("avocado.md")
result.to_docx("avocado.docx")
```

### What ends up in the exports

Both exporters carry over:

| Element | Markdown | Word |
|---|---|---|
| Title from research description | ✓ | ✓ |
| Scholar abstract | ✓ | ✓ |
| Bold sub-field labels (Human subsystem / Natural subsystem / Geographic scope / Description) and agent level | ✓ | ✓ |
| Per-category Flows subsections (Matter / Capital / Information / Energy / People / Organisms) | ✓ | ✓ |
| Per-category Causes + Effects subsections (eight fixed categories) | ✓ | ✓ |
| RAG evidence + web sources block | ✓ | ✓ |
| §1 classification as a bullet list when the model wrote bullets (Markdown: printed as written; Word: each bullet becomes a bullet paragraph) | ✓ | ✓ |
| §N.1.K sub-heading for each system (role and name, e.g. `Sending: Mexico avocado regions`) | ✓ | ✓ |
| Inline `[Tk:N]` / `[Tk:Wn]` citation markers | ✓ | ✓ |
| §5 Cross-Coupling Interactions, §6 Research Gaps, §7 Evidence Coverage | ✓ | ✓ |
| Flows table (Category / Source / Target / Bidirectional / Description); Source and Target show — unless structured map data exists (an `auto_map=True` run) | ✓ | ✓ |
| Map | `<stem>_map.png` written beside the file and linked (only when `path` is given and a map exists) | embedded figure (when a map exists) |

Web sources are listed as `[W1]`, `[W2]`, ... in hit order, without the
turn prefix; match them to the inline `[T1:Wn]` markers by number.

### Customising the abstract pass

The abstract is generated by one extra LLM call at the end of every
`analyze()` and `refine()` turn and stored as a plain attribute of that
turn's result; to customise it, reassign `result.abstract = "..."` before
exporting (both exporters read the attribute). Pass
`generate_abstract=False` to `MetacouplingAssistant(...)` to skip the
call: `result.abstract` is then `""` (as it is if the call fails) and the
exports leave out the Abstract section. There is no public regeneration
API; the internal hook is private (`advisor._generate_abstract(formatted_text)`)
and takes the fully-assembled formatted report string, so editing
`result.parsed` alone does not change the abstract.

---

## 16. Quantitative Indicators

The `metacouplingllm.indicators` submodule computes
deterministic metacoupling indicators on the user's own flow data.
This section gives the formulas first, then the public functions and
how to feed them your data.

Install: `pip install "metacouplingllm[indicators]"` (pulls in pandas).

### Formulas

**Metacoupled flow shares (per focal system *i*):**

$$
\begin{aligned}
\mathrm{IFS}_i &= \frac{F_{iI}}{F_{iI} + F_{iP} + F_{iT}}, \\
\mathrm{PFS}_i &= \frac{F_{iP}}{F_{iI} + F_{iP} + F_{iT}}, \\
\mathrm{TFS}_i &= \frac{F_{iT}}{F_{iI} + F_{iP} + F_{iT}}
\end{aligned}
$$

where $F_{iI}$, $F_{iP}$, $F_{iT}$ are system $i$'s intra-, peri-, and telecoupled flow magnitudes.

**MFE** — Metacoupled Flow Evenness (Shannon 1948 entropy, normalised to $[0, 1]$):

$$
\mathrm{MFE}_i = -\frac{1}{\ln 3} \sum_{c \,\in\, \{I,\, P,\, T\}} s_c \ln s_c,
\qquad s_c \in \{\mathrm{IFS}_i,\ \mathrm{PFS}_i,\ \mathrm{TFS}_i\}
$$

**MFCI** — Metacoupled Flow Concentration Index (Hirschman 1945 / Herfindahl 1950 HHI, normalised per Hannah & Kay 1977; per coupling type *c*):

$$
\mathrm{HHI}_{ic} = \sum_j \left( \frac{f_{ij}^{c}}{F_{ic}} \right)^{2},
\qquad
\mathrm{MFCI}_{ic} = \frac{\mathrm{HHI}_{ic} - 1/n_{ic}}{1 - 1/n_{ic}},
\qquad
\mathrm{ENP}_{ic} = \frac{1}{\mathrm{HHI}_{ic}}
$$

where $f_{ij}^{c}$ is the type-$c$ flow from system $i$ to partner $j$, $F_{ic} = \sum_j f_{ij}^{c}$, $n_{ic}$ is the number of partners with type-$c$ flow (partners with zero flow are not counted), and ENP is the Equivalent Number of Partners (Laakso & Taagepera 1979) — reported per coupling type as `ENP_I` / `ENP_P` / `ENP_T`, the Equivalent Number of Intra-/Peri-/Telecoupled Partners.

Two edge-case conventions (each emits a `UserWarning`): when
$n_{ic} = 1$ the normalisation is undefined and MFCI is defined as
**1** by convention (a single partner is maximal concentration —
e.g. the intracoupling self-loop, which is why `IFCI = 1` in the
worked example below); when $F_{ic} = 0$ MFCI is **NaN**. In the MFE
sum, $0 \ln 0 = 0$ by convention.

### Six public functions

| Function | What it does |
|---|---|
| `classify_coupling(edges, focal_id, adjacency, ...)` | Add `coupling_type` column (`I` / `P` / `T`) to an edge table using a user-supplied adjacency table. |
| `build_adjacency(pairs, level, ...)` | Fill the `adjacent` flag (`1` / `0` / `<NA>`) from the bundled pericoupling DB (`level="adm0"` or `"adm1"`) — feeds `classify_coupling` (see below). |
| `compute_flow_shares(data, ...)` | Metacoupled Flow Shares (IFS / PFS / TFS) per focal system. |
| `compute_mfe(data, ...)` | Metacoupled Flow Evenness — normalised Shannon entropy. `data` is the shares table returned by `compute_flow_shares` (needs `IFS` / `PFS` / `TFS`), not the edge table. |
| `compute_mfci(data, ...)` | Metacoupled Flow Concentration Index — normalised HHI per coupling type. |
| `summarize_metacoupling(data, ...)` | One-shot combined indicator table. |

`compute_mfci` returns a **long-format** table (one row per system ×
coupling type, with `n_partners`, `HHI`, `MFCI`,
`effective_n_partners`); the wide `IFCI` / `PFCI` / `TFCI` columns plus
`ENP_I` / `ENP_P` / `ENP_T` are produced by `summarize_metacoupling`.

### Preparing your data (CSV or Excel)

The indicators are **DataFrame-in / DataFrame-out** — you load your own
data with pandas and pass it in; the package never reads files itself.
A real study has tens to thousands of flows, so the natural workflow is
to keep them in a spreadsheet and read it.

You provide **two tables**:

**1. A flows table** — one row per flow (the unit of analysis):

| column | meaning | required |
|---|---|---|
| `focal_system_id` | the system the analysis is centred on (e.g. `Brazil`) | ✓ |
| `origin_id` | where the flow starts | ✓ |
| `destination_id` | where the flow ends | ✓ |
| `flow_value` | the flow magnitude (tonnes, USD, head, …; use one unit per analysis) | ✓ |
| `flow_type` | optional Liu-2017 category (`matter`, `capital`, …) for per-type breakdowns | optional |

Partners (for MFCI, ENP and the `n_*` counts) are read from
`destination_id` (keyword `partner_col`, default `"destination_id"`), so
each row should be a flow from the focal system to a partner. For inbound
flows (partner in `origin_id`, focal system in `destination_id`) pass
`partner_col="origin_id"` to `summarize_metacoupling` / `compute_mfci`;
otherwise every inbound partner is counted as the focal system itself and
MFCI, ENP and `n_*` are wrong (shares and MFE are unaffected). For a table
that mixes exports and imports, add a column with each row's partner (the
end that is not the focal system) and pass its name as `partner_col`.

**2. An adjacency table** — which partners physically border the focal
system (this is what distinguishes *peri-* from *telecoupling*):

| column | meaning |
|---|---|
| `origin_id`, `destination_id` | a bordering pair |
| `adjacent` | `1` if they share a border |

If you already have a coupling label for every flow, skip
`classify_coupling`: put it in a `coupling_type` column (`I` / `P` / `T`,
case-insensitive; `intra` / `peri` / `tele`, `intracoupling` /
`pericoupling` / `telecoupling` and `intracoupled` / `pericoupled` /
`telecoupled` are also accepted) and pass the table straight to
`summarize_metacoupling` (or to `compute_flow_shares` / `compute_mfci`).
That table needs only `focal_system_id`, `destination_id` (the partner),
`flow_value` and `coupling_type`; `origin_id` is not read. Rows with any
other label are left out of the totals, with a `UserWarning` from
`summarize_metacoupling` and `compute_flow_shares` (`compute_mfci` alone
drops them silently).

> **CSV needs nothing beyond pandas.** **Excel** (`.xlsx`) additionally
> needs `pip install openpyxl`. If your spreadsheet uses different
> headers, you don't have to rename anything — every function takes
> column-name keywords: `classify_coupling` (`origin_col`,
> `destination_col` for the flows table; `adjacency_origin_col`,
> `adjacency_destination_col`, `adjacency_flag_col` for the adjacency
> table; `coupling_col_out` names the new column), `build_adjacency`
> (`origin_col`, `destination_col`, `flag_col`), `compute_flow_shares`
> (`system_col`, `coupling_col`, `weight_col`), `compute_mfci` and
> `summarize_metacoupling` (`system_col`, `partner_col`, `coupling_col`,
> `weight_col`), `compute_mfe` (`share_cols`).

### Auto-filling `adjacent` from the bundled database (ADM0 & ADM1)

Hand-authoring the adjacency table is tedious for a large study — and
unnecessary when your partners are countries or World Bank ADM1 regions.
`build_adjacency` fills the `adjacent` flag for you from the package's
curated pericoupling databases (§8): you supply only `origin_id` /
`destination_id`, and it writes `1` (pericoupled / bordering) or `0`
(telecoupled / not bordering). It works at both scales:

| `level=` | what `origin_id` / `destination_id` may contain |
|---|---|
| `"adm0"` | country **names** (`Brazil`) or **ISO alpha-3** codes (`BRA`) |
| `"adm1"` | WB ADM1 **codes** (`MEX008`), region **names** (`Chihuahua`), or `"Region, Country"` (`Chihuahua, Mexico`) |

Because the focal system's `origin_id` / `destination_id` columns are
already in your flows table, you can point it straight at `edges`:

```python
import pandas as pd
from metacouplingllm.indicators import build_adjacency

edges = pd.read_csv("examples/brazil_soybean_flows.csv")   # sample flows file, see the worked example below
adjacency = build_adjacency(edges, level="adm0")   # fills 1/0 from the DB
print(adjacency)
#   origin_id destination_id  adjacent
# 0    Brazil      Argentina         1
# 1    Brazil       Paraguay         1
# 2    Brazil          China         0
# 3    Brazil             EU      <NA>   ← unresolved (see note)
# 4    Brazil            USA         0
```

It **collapses duplicate pairs and drops self-pairs** (`Brazil→Brazil`;
intracoupling is handled by `classify_coupling` via `focal_id`, not by
adjacency). Anything the database can't resolve — here `EU`, a bloc
rather than a country — is left as `<NA>` and reported in a single
`UserWarning`, so an unknown ID is never silently read as "distant".
Feeding this table into `classify_coupling` reproduces exactly the same
classification as the hand-authored file below (`<NA>` and `0` both count
as not-adjacent).

The same call at the subnational scale, mixing codes and names:

```python
import pandas as pd
from metacouplingllm.indicators import build_adjacency

pairs = pd.DataFrame({
    "origin_id":      ["MEX008", "Chihuahua, Mexico"],
    "destination_id": ["USA044", "Florida, United States"],
})
build_adjacency(pairs, level="adm1")
#           origin_id          destination_id  adjacent
# 0            MEX008                  USA044         1   # Chihuahua–Texas (border)
# 1 Chihuahua, Mexico  Florida, United States         0   # not adjacent
```

`build_adjacency` takes the same `de_facto_borders=` and
`coupling_standard=` arguments as the §8 lookups, so disputed-territory
and water-separated conventions carry through. **Review the table before
relying on it** — the helper builds a transparent, editable adjacency
table; it doesn't replace your judgement.

### Worked example: Brazil soybean (from CSV files)

Sample files are in the repository's `examples/` folder
(`examples/brazil_soybean_flows.csv` and
`examples/brazil_soybean_adjacency.csv`); they are not installed by
`pip install`. Run the snippet from a clone of the repository, or save
the two CSV blocks below under those names. The flows file looks like this
(a real study would have many more rows and several focal systems):

```csv
focal_system_id,origin_id,destination_id,flow_value,flow_type
Brazil,Brazil,Brazil,10,matter
Brazil,Brazil,Argentina,5,matter
Brazil,Brazil,Paraguay,15,matter
Brazil,Brazil,China,50,matter
Brazil,Brazil,EU,12,matter
Brazil,Brazil,USA,8,matter
```

and the adjacency file marks Brazil's land neighbours:

```csv
origin_id,destination_id,adjacent
Brazil,Argentina,1
Brazil,Paraguay,1
```

Load, classify, compute, and save the results:

```python
import pandas as pd
from metacouplingllm.indicators import (
    build_adjacency, classify_coupling, summarize_metacoupling,
)

edges     = pd.read_csv("examples/brazil_soybean_flows.csv")
adjacency = pd.read_csv("examples/brazil_soybean_adjacency.csv")
# Excel instead?  edges = pd.read_excel("flows.xlsx")   # needs: pip install openpyxl
# Or skip authoring the adjacency file and auto-fill it from the bundled DB:
#   adjacency = build_adjacency(edges, level="adm0")    # see the subsection above

classified = classify_coupling(edges, focal_id="Brazil", adjacency=adjacency)
summary    = summarize_metacoupling(classified)
summary.to_csv("indicators.csv", index=False)          # write results back out

cols = ["focal_system_id", "IFS", "PFS", "TFS", "MFE", "IFCI", "PFCI", "TFCI"]
print(summary[cols].round(2))
#   focal_system_id  IFS  PFS  TFS   MFE  IFCI  PFCI  TFCI
# 0          Brazil  0.1  0.2  0.7  0.73   1.0  0.25  0.33
```

`classify_coupling` reads each row's (`origin_id`, `destination_id`) pair:
both ends are the focal system → `I`; exactly one end is the focal system
and the pair is listed with `adjacent` = 1 (in either direction) → `P`;
exactly one end is the focal system and the pair is unlisted, `0` or
`<NA>` → `T`; neither end is the focal system → `NaN` with a
`UserWarning` (the row is then left out of the indicators). The full
`indicators.csv` of this example carries 18 columns (`focal_system_id`
plus the 17 columns listed here): raw magnitudes
(`F_I`/`F_P`/`F_T`/`F_total`), shares (`IFS`/`PFS`/`TFS`), `MFE`,
concentration (`IFCI`/`PFCI`/`TFCI`), partner counts
(`n_I`/`n_P`/`n_T`), and equivalent partners (`ENP_I`/`ENP_P`/`ENP_T`).
If a coupling type has no flow in any group, its `xFCI` and `ENP_x`
columns are omitted (e.g. 16 columns with no pericoupled flow).

Interpretation: 70 % of Brazil's classified soybean-equivalent flow is
**telecoupled** (China + EU + USA), 20 % **pericoupled** (Argentina +
Paraguay), 10 % **intracoupled**. `MFE ≈ 0.73` shows a moderately even
mix across coupling types; `TFCI ≈ 0.33` shows the telecoupled flows
are moderately concentrated (China dominates, with the EU and USA
sharing the remainder).

`focal_id` names one focal system per call (a list means internal
subunits of one focal system, so flows among them count as `I`); the
`focal_system_id` column is not read. With several focal systems,
classify each system's rows separately and concatenate (the
`focal_system_id` values must match the IDs used in `origin_id` /
`destination_id`):

```python
classified = pd.concat([
    classify_coupling(g, focal_id=k, adjacency=adjacency)
    for k, g in edges.groupby("focal_system_id")
])
```

### Orthogonal flow_type × coupling_type analysis

If your flows file carries a `flow_type` column, pass
`group_cols=["flow_type"]` to break the indicators down **per flow
type** as well as per coupling type:

```python
summary_by_flow = summarize_metacoupling(classified, group_cols=["flow_type"])
```

Useful to see, for example, whether telecoupling is dominated by matter
flows (soybean tonnage) while pericoupling is dominated by financial
flows (cross-border payments).

### Guardrails

The functions emit `UserWarning`s when:

- **`F_total == 0`** — IFS/PFS/TFS and MFE are returned as NaN.
- **`F_I == 0`** but P or T is non-zero — `IFS = 0` is shown but may be
  a data artifact; the warning stops users misreading "missing
  intracoupling data" as "no intracoupling".
- **Unrecognised coupling-type labels** in the input — dropped from
  totals, with a one-shot warning.
- **Negative or non-numeric `flow_value`** — the row is excluded from the
  totals, with a `UserWarning`.
- **Edges with no focal endpoint** (neither `origin_id` nor
  `destination_id` is in `focal_id`) — `classify_coupling` leaves
  `coupling_type` as `NaN`, with a `UserWarning`, and the row is left out
  of the indicators.

`classify_coupling` raises `ValueError` when called without an
adjacency table on cross-system edges (it refuses to silently guess
geography); pass `llm_client=` to opt into LLM-assisted classification
of ambiguous edges (see §17).

### Design principles

- **Deterministic-first.** The indicator math never calls an LLM;
  identical input → identical output.
- **Established statistics, not invented indices.** Shannon (1948)
  entropy; Hirschman (1945) / Herfindahl (1950) HHI, normalised per
  Hannah & Kay (1977); Equivalent Number of Partners per Laakso &
  Taagepera (1979).
- **User supplies adjacency** — no hardcoded geography in the indicator
  math. `build_adjacency` is an opt-in helper that fills the `adjacent`
  flag from the bundled pericoupling DB and warns on anything it can't
  resolve; you still review the table and pass it to `classify_coupling`.
- **Intracoupling data required** — the package warns when `F_I = 0`
  so missing data isn't misread as "no intracoupling".

---

## 17. LLM-Assisted Indicator Helpers

The optional `metacouplingllm.indicators.llm` submodule
wraps natural-language judgment tasks around the deterministic
indicator core. Five helpers cover the workflow steps where LLM help
adds value, each returning `(result, LLMTrace)` for reproducibility.

Install: any of the LLM-provider extras (`[openai]`, `[anthropic]`,
`[gemini]`, `[grok]`) plus `[indicators]`.

### Five public helpers

| Function | What it does |
|---|---|
| `define_study(description, *, llm_client)` | Natural-language description → structured study config dict (keys: `focal_system`, `flow_type`, `flow_unit`, `intracoupling_rule`, `pericoupling_rule`, `telecoupling_rule`, `recommended_input_level`, `required_columns`, `warnings`). |
| `check_inputs(data_summary, sample_rows=None, *, llm_client)` | Validate user data: which indicator families can be computed, what's missing, unit / intracoupling-self-loop warnings. |
| `classify_ambiguous_edges(edges, study_config, *, llm_client)` | Classify edges the deterministic pass couldn't resolve. Returns a DataFrame with `suggested_coupling_type` (`"I"` / `"P"` / `"T"` / `"unknown"`), `confidence`, `reason`, `needs_user_confirmation`. `edges` needs `origin_id` / `destination_id` columns (override with `origin_col=` / `destination_col=`). |
| `interpret_results(results, *, llm_client, audience="academic")` | Plain-language interpretation of a computed indicator table. Audience presets: `"academic"` / `"general"` / `"policy"`. |
| `write_methods(indicator_spec, *, llm_client)` | Manuscript-ready Methods text with formulas + standard citations (Shannon 1948, Hirschman 1945, Hannah & Kay 1977, Laakso & Taagepera 1979, Liu 2017). |

### The `LLMTrace` dataclass

Every helper returns an `LLMTrace` alongside the result:

```python
@dataclass
class LLMTrace:
    timestamp_utc: str      # ISO 8601 with Z suffix
    model: str              # e.g., "gpt-4o"
    prompt_version: str     # e.g., "define_study_v1"
    system_prompt: str
    user_prompt: str
    raw_response: str
    usage: dict | None      # token accounting if provider returned it
```

Save it however you like — the package doesn't make filesystem
assumptions. Common pattern: write `trace.__dict__` to JSON next to
your output file.

### LLM-assisted edge resolution in `classify_coupling()`

`classify_coupling()` takes three keyword-only arguments for
LLM-assisted edge resolution: `llm_client`, `study_config` and `model`.
`model` runs the LLM call on another model than the adapter's: a
built-in adapter is copied with that model (its other settings kept;
the adapter you passed is unchanged), and a custom client receives
`model=` on `chat()`. The same keyword works on every LLM helper.

When `llm_client` is supplied AND the deterministic pass leaves some
edges as `NaN`, the function automatically calls
`classify_ambiguous_edges()` on just those rows and merges the suggested
`I`/`P`/`T` labels back. The deterministic pass leaves `NaN` in two cases
only: no `adjacency` table was passed (then every edge that is not
intracoupling is left to the LLM), or neither endpoint of the edge is in
`focal_id`. When a table is passed, a focal-system edge that it does not
list as adjacent (no row, flag `0`, or `<NA>`) is classified `T`
deterministically and never reaches the LLM; to have the LLM look at
pairs that `build_adjacency` left as `<NA>`, call
`classify_ambiguous_edges()` on those rows directly. Without a table:

```python
llm_client = OpenAIAdapter(client, model="gpt-4o")  # the adapter you built in §4

classified = classify_coupling(
    edges,
    focal_id="Brazil",
    adjacency=None,                 # no table: the LLM classifies every non-intracoupling edge
    llm_client=llm_client,          # ← opt-in LLM fallback
    study_config={"focal_system": "Brazil", "flow_unit": "Mt"},
)

# Inspect what the LLM suggested.  The trace is attached to the
# DataFrame's metadata ONLY when the LLM actually ran (llm_client
# passed AND at least one edge was unresolved) — use .get() so a
# fully-deterministic run doesn't raise KeyError:
trace = classified.attrs.get("llm_classify_trace")
if trace is not None:
    print(trace.raw_response)   # the reply JSON (with AnthropicAdapter, the input of its tool call)
# trace is None  ⇒  every edge was classified deterministically.
```

When the LLM returns `"unknown"`, the row stays `NaN`; the prompt tells
it to answer `"unknown"` rather than guess. Any `I`/`P`/`T` label it does
return is written into the coupling column (`coupling_type` by default)
as is: `confidence`, `reason` and `needs_user_confirmation` are not
copied into the returned table, and no other column marks the row as
LLM-assigned. To get those fields call `classify_ambiguous_edges()`
yourself on the unresolved rows, and review every LLM-assigned label
before relying on it.

Without `llm_client`, no LLM is called: edges the deterministic pass
cannot classify stay `NaN` with a `UserWarning`; with no adjacency table,
any edge that is not intracoupling raises `ValueError` instead.

> **Why no `(result, trace)` pair here?** Unlike the five helpers,
> `classify_coupling` returns a single DataFrame; when the LLM ran, the
> trace rides along in `DataFrame.attrs["llm_classify_trace"]` instead.

### End-to-end workflow

```python
from openai import OpenAI
from metacouplingllm import OpenAIAdapter
from metacouplingllm.indicators import (
    classify_coupling, summarize_metacoupling,
    define_study, check_inputs,
    interpret_results, write_methods,
)

client     = OpenAI(api_key="sk-...")
llm_client = OpenAIAdapter(client, model="gpt-4o")

# 1. Turn a study description into a structured config
study_config, trace_def = define_study(
    "I want to study Brazil's soybean export footprint across "
    "intracoupling (within-Brazil consumption), pericoupling "
    "(Argentina + Paraguay), and telecoupling (China + EU + USA).",
    llm_client=llm_client,
)

# 2. Validate that the user's CSV has the right columns
#    (here the sample files from the repository's examples/ folder, see §16)
import pandas as pd
edges = pd.read_csv("examples/brazil_soybean_flows.csv")
check, trace_chk = check_inputs(
    data_summary={"row_count": len(edges), "columns": list(edges.columns)},
    sample_rows=edges.head(5).to_dict("records"),
    llm_client=llm_client,
)

# 3. Classify with the adjacency table (edges it does not list as adjacent
#    become T; the LLM is consulted only for edges that do not touch the
#    focal system)
adjacency = pd.read_csv("examples/brazil_soybean_adjacency.csv")
classified = classify_coupling(
    edges, focal_id="Brazil", adjacency=adjacency,
    llm_client=llm_client, study_config=study_config,
)

# 4. Compute the deterministic indicators
summary = summarize_metacoupling(classified)

# 5. Plain-language interpretation for a target audience
#    (pass the indicator DataFrame itself, not a list of dicts)
interpretation, trace_int = interpret_results(
    summary,
    llm_client=llm_client,
    audience="academic",
)

# 6. Manuscript-ready Methods text
methods_text, trace_m = write_methods(
    {"indicator_families": ["flow_shares", "MFE", "MFCI"]},
    llm_client=llm_client,
)
```

### Guardrails baked into every prompt

- LLM **must not** invent numerical flow values.
- LLM **must not** calculate final indicator values — deterministic
  code does that.
- LLM **must** say `"unknown"` / surface gaps rather than guess.
- All structured-output helpers (`define_study`, `check_inputs`,
  `classify_ambiguous_edges`) return strict JSON. The package uses
  each adapter's native strict-output mode and falls back to a
  defensive JSON-object extractor only if the strict path fails.

### Provider parity

All five helpers work across the same four adapters as the rest of
the package: OpenAI, Anthropic, Gemini, Grok. Each adapter dispatches
to its native strict-JSON mode (see §14 for the matrix).

---

## 18. References

- Liu, J. (2017). Integration across a metacoupled world. *Ecology and
  Society*, 22(4), 29.
- Liu, J., et al. (2013). Framing sustainability in a telecoupled world.
  *Ecology and Society*, 18(2), 26.
- Shannon, C. E. (1948). A mathematical theory of communication. *Bell
  System Technical Journal*, 27(3), 379-423.
- Hirschman, A. O. (1945). *National Power and the Structure of
  Foreign Trade.* Berkeley: University of California Press.
- Herfindahl, O. C. (1950). *Concentration in the Steel Industry.* PhD
  thesis, Columbia University, New York.
- Hannah, L., & Kay, J. A. (1977). *Concentration in Modern Industry:
  Theory, Measurement and the UK Experience.* London: Macmillan
  (Springer).
- Laakso, M., & Taagepera, R. (1979). "Effective" number of parties: A
  measure with application to West Europe. *Comparative Political
  Studies*, 12(1), 3-27.
