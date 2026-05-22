# Changelog

All notable changes to the `MetacouplingLLM` package are documented in this
file. The format is loosely based on
[Keep a Changelog](https://keepachangelog.com/), and this project follows
[Semantic Versioning](https://semver.org/).

## [Unreleased]

### Fixed

- **Parser-level fix for fragmented LLM output patterns.**  PR #33
  shipped a defensive renderer-side workaround for two real bugs in
  `llm/parser.py` that left scholar docx exports unreadable.  PR #34
  fixes the parser itself so the broken intermediate shape never
  reaches the export layer.  PR #33's defensive helpers
  (`_merge_fragmented_flows`, `_split_collapsed_causes_effects`)
  become idempotent no-ops on the now-clean parser output and can
  be removed in a follow-up cleanup PR.

  **Two coordinated changes in `src/metacouplingllm/llm/parser.py`:**

  1. **New `_merge_fragmented_flow_entries()` helper** runs as a
     post-processing pass at the end of `_parse_flows()`.  The LLM
     sometimes emits each logical flow as 3 lines wrapped in
     top-level bullets:

     ```
     - Matter Flow
       - Direction: Orchards -> Packinghouses
       - Description: Avocados moved locally [T1:W3].
     ```

     Without the merge, the parser produced 3 separate flow dicts
     per logical flow (one header with `{category: matter,
     description: "Matter Flow"}`, then direction-only and
     description-only dicts with no category).  The merge
     recognises canonical-category header dicts (description
     matches the `"{Category} Flow"` placeholder OR no direction
     at all) and merges subsequent uncategorised entries into them
     until the next canonical header.  Idempotent: well-formed
     flows pass through unchanged.

  2. **`_extract_categorized_bullets()` now recognises plain-text
     category names** as section dividers, not just bold headings.
     The LLM sometimes emits unbold category names:

     ```
     General:
     - Economic
     - Strong U.S. demand and price incentives.
     - Political / Institutional
     - Phytosanitary requirements from SENASICA.
     - Hydrological
     - Local water availability conditioning yields.
     ```

     Previously the parser collapsed every cause/effect under one
     `"general"` key, with real category names ending up as
     siblings of the actual content under that key.  Now a bullet
     whose entire text matches a known Liu framework category name
     (looked up via the existing `_CAUSE_EFFECT_CATEGORY_ALIASES`
     table — including short forms like "Political" or "Cultural")
     is treated as a section divider, splitting subsequent items
     into proper per-category buckets.  Existing bold-heading
     format is unchanged.

  Both fixes are **strictly additive**: existing parser tests
  (53 cases across `TestMultilineFlows`, `TestNumberedFlows`,
  `TestNestedSystems`, `TestParseAnalysis`, `TestParsedAnalysis`,
  `TestGPT51SystemParsing`) all pass without modification.

  9 new tests in `tests/test_parser.py`:
  - `TestParseFragmentedFlows` (4): fragmented-block merge into one
    flow per logical entry, idempotence on clean input, orphan
    sub-entry without header, empty input
  - `TestParseUnboldCauseEffectCategories` (5): plain-text category
    split into real buckets, bold-heading backward compatibility,
    short-form alias normalisation ("Political" → "political /
    institutional"), non-category bullets stay as content items,
    fallback "general" bucket kept when no inline categories

  **Local: 62/62 parser tests pass** (53 prior + 9 new).

  See PR #33 for the original scholar-review issue and the
  defensive renderer-side workaround.

- **Defensive renderer fix for fragmented parser output in
  Markdown / Word exports.**  Scholar opened the PR #32 live
  trace docx and the structure of §2.2 Flows and §2.4 Causes /
  §2.5 Effects was unreadable.  Root cause turned out to be in
  `llm/parser.py` (not the renderer):

  - **Flows fragmented into 3 entries per logical flow** — the
    LLM emits each flow as a 3-line block (category header,
    direction, description) but the parser stored each line as
    its own dict.  PR #32's per-category grouping then put the
    orphan "Matter flow" / "Capital flow" placeholders in
    canonical groups (1 item each) and dumped all 12 real
    flow descriptions into a §2.2.7 Unspecified mega-list.
  - **Causes / Effects collapsed under one "General" key** —
    the LLM emits a 2-level hierarchy (`General: → Economic
    → Strong U.S. demand...`) but the parser flattened to
    `{"General": ["Economic", "Strong U.S. demand...", ...]}`,
    losing the nested structure.  The real category names
    (Economic, Political / Institutional, Hydrological, …)
    ended up as siblings of the actual cause prose.

  Per the parser-fix-deferred decision, PR #33 ships a
  **defensive renderer fix** (workarounds in
  `output/export.py::_build_sections`) so scholars get a
  usable docx today.  PR #34 will fix the parser properly,
  at which point these helpers become no-ops and can be
  removed.  Both helpers are **idempotent**: clean parser
  output passes through unchanged.

  Two new helpers in `src/metacouplingllm/output/export.py`:

  1. **`_merge_fragmented_flows(flows)`** — walks the flows
     list and stitches header-direction-description triples
     back into single logical flow dicts.  A flow is a
     "header" when its category is in
     `_CANONICAL_FLOW_CATEGORY_ORDER` AND it has no real
     direction OR its description matches the
     `"{Category} flow"` placeholder.  Subsequent
     unspecified-category flows get their direction +
     description merged into the current header until the
     next canonical header.
  2. **`_split_collapsed_causes_effects(items_dict)`** —
     detects the single-key dict pattern (commonly
     `"General"`) and splits items by inline Liu framework
     category names sourced from `prompts/templates.py`
     (Economic, Political / Institutional, Ecological /
     Biological, Technological / Infrastructural, Cultural /
     Social / Demographic, Hydrological, Climatic /
     Atmospheric, Geological / Geomorphological).  Safety
     belt: requires ≥2 inline category names to trigger, so a
     legitimate single-category dict that happens to contain
     one Liu category word passes through unchanged.

  Also fixed two smaller render-side issues spotted during the
  same scholar review:

  - **§1 Coupling Classification in docx** now renders as
    List Bullet paragraphs when the LLM emits `- Intracoupling
    ... \n- Pericoupling ... \n- Telecoupling ...` (was one
    Normal-paragraph block in docx).  Markdown was already
    fine.
  - **§N.1 Systems role lines** (`Focal: ...`, `Sending: ...`,
    `Receiving: ...`) promoted to Heading 3 subsections
    (§N.1.K), parallel to the §N.2.K Flows / §N.4.K Causes
    treatment.  Each system now shows up in Word's navigation
    pane.

  15 new tests in `tests/test_output.py`:
  - `TestMergeFragmentedFlows` (5 cases): 3-entry merge, multi-
    category merge, idempotence on clean input, empty input,
    orphan sub-flow without header
  - `TestSplitCollapsedCausesEffects` (4 cases): General →
    real-categories split, idempotence on multi-key input,
    safety belt when <2 inline categories, empty input
  - `TestBrokenParserResultRendering` (4 cases): end-to-end
    Markdown rendering against a new `broken_parser_result`
    fixture modeling the real Mexico avocado trace's parser
    output — verifies the symptoms the scholar reported are
    gone (real flow content under each canonical category, no
    placeholder-only items, no §2.2.7 Unspecified mega-list,
    Causes split into real categories)
  - `TestDocxClassificationBullets` (2 cases): docx multi-
    bullet split + single-paragraph fallback

  Verified end-to-end via live OpenAI GPT-5 retrace.

### Changed

- **Polish rendering of Markdown / Word exports for scholar
  review.**  After PR #31 shipped the export feature, scholar
  spot-check of the live trace docx surfaced four concrete
  readability gaps.  This PR polishes them.

  1. **Title uses the user's original query.**  Was
     `# Metacoupling Analysis — Case: - Intracoupling — present.
     Within Jalisco's coupled human–natural system, avocad…` (the
     first 80 chars of `coupling_classification`, which often
     starts with bullet markers and reads like garbage).  Now
     `# Metacoupling Analysis: {user_query}` so the document
     describes itself when filed away.  `_build_result()` attaches
     `result._original_query_for_export` (mirroring the existing
     `_web_sources_for_export` / `_rag_hits_for_export` private-
     attribute pattern); the renderers prefer it over the old
     `{focal}: {topic}` heuristic, which stays as a fallback for
     test code that builds `AnalysisResult` outside the
     `analyze()` pipeline.

  2. **Sub-field labels are bolded** in both §N.1 Systems
     (`Human subsystem`, `Natural subsystem`, `Geographic scope`,
     `Description`) and §N.3 Agents (level prefix:
     `Individuals / Households`, `Firms / Traders / Corporations`,
     etc.).  Was italic — too weak for scholars scanning a long
     document.  In docx, label runs use `bold=True`; in Markdown,
     `*Label*` → `**Label**`.

  3. **§N.2 Flows grouped into per-category subsections.**  Was
     one flat numbered list 1.-15. across all 6 categories,
     making category boundaries invisible.  Now Flows are split
     by canonical Liu 2017 category order
     (`matter, capital, information, energy, people, organisms`,
     unknowns last) into `### N.2.K {Category}` (Markdown) /
     Heading 3 (Word) subsections.  Numbering restarts inside
     each subsection.  New `_group_flows_by_category()` helper
     shared by both renderers.

  4. **§N.4 Causes / §N.5 Effects categories rendered as
     separate subsections.**  In docx the old code rendered
     each category as one paragraph
     `Category: item1; item2; item3` with semicolon-separated
     items; with 4-6 categories per CouplingSection scholars
     saw a wall of semicolons.  Now each category becomes its
     own Heading 3 (`### N.4.K {Category}` / `### N.5.K
     {Category}`) followed by one bullet per item.  Markdown
     also moves to nested `####` subsections for visual
     consistency with the new Flows treatment.

  No new LLM calls, no schema changes, no new dependencies.
  Categories within Causes / Effects preserve LLM emission
  order (Python 3.7+ dict insertion order).

  12 new tests in `tests/test_output.py` covering:
  - Title uses original query when attached; falls back to the
    `{focal}: {topic}` heuristic otherwise
  - Systems sub-field labels are bold (Markdown `**…**` + docx
    bold runs)
  - Agents level labels are bold
  - Flows grouped by canonical category order (Matter, Capital,
    Information, …) with `#### N.2.K` Markdown subsections / docx
    Heading 3
  - Flows preserves canonical order even when LLM emits
    categories out of order
  - Causes / Effects rendered as per-category subsections with
    bullets per item (not jammed into one paragraph)

  Existing `minimal_result` fixture extended to cover the new
  paths (added agents, causes, effects entries +
  `_original_query_for_export`).

  Verified end-to-end via live OpenAI GPT-5 retrace.

- **Scholar-friendly output: `result.abstract` + `result.to_markdown()`
  + `result.to_docx()`.**  The package's analysis output was rich
  (a `result.formatted` text + a `result.map` matplotlib Figure) but
  not in a form scholars could use directly when writing a paper.
  Three additions on `AnalysisResult`:

  1. **`result.abstract: str`** -- a 150-250 word, single-paragraph
     summary suitable for a paper introduction or grant proposal.
     Generated by one small extra LLM call in `_build_result()` after
     the main analysis + map data + RAG / web evidence assembly are
     complete, using the assembled `formatted` text as input.  New
     `ABSTRACT_GENERATION_SYSTEM` / `ABSTRACT_GENERATION_USER` prompts
     in `prompts/templates.py` ask for formal academic prose covering
     focal system → coupling-type classification → key flows /
     destinations → headline finding.  Non-fatal on LLM failure
     (field left as empty string so callers that don't read
     `.abstract` are unaffected).

  2. **`result.to_markdown(path=None) -> str`** -- renders the
     analysis as scholar-friendly Markdown with `#` / `##` headings
     for §1-§7, a Markdown table for flows
     (Category / Source / Target / Bidirectional / Description), a
     Web Sources list with `[W1]` / `[W2]` IDs matching the existing
     citation markers in the prose, and an embedded `![map](...)`
     reference when `path=` is given (so the map PNG gets saved
     alongside the .md file).  Zero new dependencies.

  3. **`result.to_docx(path=None) -> Path`** -- renders the same
     content as a Word document via `python-docx`.  Word tables for
     flows + web sources, embedded map figure via `BytesIO`,
     Heading 1 / Heading 2 styles.  Returns the `pathlib.Path`
     written.  `python-docx` is an OPTIONAL dependency under the
     new `[project.optional-dependencies].export` extra -- install
     with `pip install metacouplingllm[export]`.  Raises a clear
     `ImportError` with install instructions when the dep is missing.

  Both renderers share an internal `_build_sections(result) -> dict`
  intermediate (in `output/export.py`) that pulls the structured
  pieces out of `result.parsed`
  (`ParsedAnalysis.coupling_classification`, `.intracoupling` /
  `.pericoupling` / `.telecoupling` `CouplingSection` objects,
  `.cross_coupling_interactions`, `.research_gaps`,
  `.evidence_coverage_note`) and `result.parsed.map_data["flows"]`
  (Stage-3 structured output).  The intermediate is internal-only --
  not exposed as `result.sections` per the PR scope discussion
  ("most extra work is not necessary"); scholars consume the
  rendered output, not the dict.  Flows fall back to walking
  `CouplingSection.flows` prose entries when `map_data` is absent.
  Web sources are pulled from `_last_web_results` via a private
  `result._web_sources_for_export` attribute that `_build_result()`
  attaches when the pipeline ran web search.

  The exporters also render a **"Evidence from Literature"** section
  when the pipeline ran with an RAG engine in `pre_retrieval` mode:
  `_build_result()` attaches `result._rag_hits_for_export` (a list
  of `RetrievalResult` objects) the same way it attaches the web
  sources.  Markdown renders one `### [Tk:N]` subheading per hit
  with the paper title, an italic author / year line, the section
  heading, and the excerpt as a blockquote.  Word renders the same
  shape with Heading 2 + italic citation + Quote-style paragraph.
  Excerpts truncated at 600 chars for readability.  Known
  limitation: `post_hoc` RAG mode retrieves AFTER the LLM call and
  writes evidence directly to `formatted` without keeping the hits
  in an attribute, so post_hoc exports won't show the RAG section
  today.

  **Out of scope per the PR scope discussion** -- deliberately
  deferred to keep the change small:
  - Map auto-caption generation
  - Reproducibility metadata (model, date, web-search backend, etc.)
  - Bibliography / proper citation formatting beyond the existing
    `[W1]` markers
  - CSV / JSON data appendices
  - Mutable `result.sections` + `rerender()` round-trip workflow
  - LaTeX / HTML exports
  - Promoting the internal `_build_sections` to a public API

  Verified end-to-end via live OpenAI GPT-5 trace on a Mexico
  avocado exports query with the existing Papers/ corpus
  (262 markdown files indexed, 8 RAG passages retrieved per turn);
  the resulting docx surfaces both Web Sources and Evidence from
  Literature sections with proper headings and excerpts.

  32 new tests in `tests/test_output.py` (`TestAbstractField`,
  `TestBuildSections`, `TestRenderMarkdown`, `TestRenderDocx`)
  covering field defaults, section extraction, table rendering,
  file write, method delegation, headings / tables in the
  produced Word document, RAG hit shape (turn-scoped IDs, 600-char
  excerpt cap), and the new "Evidence from Literature" section in
  both Markdown and docx outputs.  Tests for the docx path skip
  automatically when `python-docx` isn't installed.

- **Stage-1 strict-output dispatch for Gemini and Grok in
  `extract_web_map_signals`.**  After PR #29 brought the
  `GeminiWebSearchBackend` and `GrokWebSearchBackend` themselves
  to parity with the OpenAI/Anthropic web-search templates,
  Stage-1 (the structured-extraction call in `websearch.py`
  that converts raw web results into typed map signals) was
  still only dispatching strict-output kwargs to OpenAI
  (`response_format=json_schema`, PR #17/#21) and Anthropic
  (`submit_web_map_signals` tool + `tool_choice` forcing,
  PR #28).  `GeminiAdapter` and `GrokAdapter` fell through to
  the prompt-based JSON path, leaving `_extract_json_object` to
  do best-effort recovery from whatever free text the model
  emitted.  This was wasted capability -- both APIs natively
  support strict JSON output, and when the prompt-based path
  failed (markdown-fenced JSON, prose preamble, mid-call
  truncation), Stage-1 returned `None` and Stage-3 had no
  signals to build the map from.

  Two coordinated changes:

  1. **`GrokAdapter.chat()` -- whitelist + `**kwargs` pattern**
     mirroring `AnthropicAdapter` (PR #28).  New
     `_ALLOWED_FORWARDED_KWARGS = frozenset({"response_format"})`;
     `chat()` accepts `**kwargs` and raises `TypeError` on
     anything outside the whitelist so typos surface at the
     call site rather than being silently dropped by the
     underlying OpenAI SDK passthrough.  Whitelisted kwargs are
     merged into the request dict before
     `client.chat.completions.create(**request_kwargs)`.
  2. **`GeminiAdapter.chat()` -- named-parameter pattern**
     mirroring `OpenAIAdapter`.  Two new optional kwargs --
     `response_schema: dict | None = None` and
     `response_mime_type: str | None = None` -- that always
     travel as a pair in practice.  Both are merged into the
     `config=` dict the adapter already builds before
     `client.models.generate_content(config=config)`.  When
     either is `None`, the corresponding key is omitted from
     the config (Gemini's SDK would reject
     `response_schema=None`).

  The Stage-1 dispatch in
  `extract_web_map_signals` (`websearch.py:2627-2680`) gains
  two new `elif` branches in the existing `isinstance` chain:

  - **Gemini**: `chat_kwargs["response_schema"] =
    _WEB_MAP_SIGNALS_SCHEMA` and
    `chat_kwargs["response_mime_type"] = "application/json"`.
    The schema is verified compatible with Gemini's
    `response_schema` subset (no `oneOf`/`anyOf`/`allOf`, uses
    `{"type": ["string", "null"]}` nullable syntax which
    Gemini accepts).
  - **Grok**: `chat_kwargs["response_format"] = {"type":
    "json_schema", "json_schema": {"name": "web_map_signals",
    "strict": True, "schema": _WEB_MAP_SIGNALS_SCHEMA}}` --
    same shape as the existing OpenAI dispatch.

  **No response-parsing changes needed.**  Both strict modes
  return the structured JSON as text in `response.content`, so
  the existing `_extract_json_object(response.content)` path
  recovers it identically to the OpenAI strict-text path.  The
  fallback path also runs unconditionally after the strict path
  for refusal / malformed-output resilience -- mirroring
  PR #28's Anthropic dispatch design.

  **Mixed adapter pattern, per user choice**: whitelist for
  Grok (Grok runs on the OpenAI SDK which passes arbitrary
  kwargs through, so a whitelist is the right typo safety
  belt), named-parameter for Gemini (the paired strict kwargs
  fit the config-dict assembly pattern already in
  `GeminiAdapter.chat()`).

  **Out of scope** for this PR, all deferred to separate
  follow-ups:
  - **Stage-2 (main framework analysis) and Stage-3
    (`_extract_map_data_from_analysis`)** still use the
    prompt-based JSON path for all four adapters.  Stage-3
    promotion has real trade-offs -- refusal risk on
    `required: [bidirectional]`, supranational-shorthand
    (`'European Union'` / `'EU'` etc.) vs strict enum
    tension, schema lowest-common-denominator across four
    backends (Gemini's `response_schema` doesn't support
    `oneOf`/`anyOf`), ~15-20 existing `test_core.py` tests
    to update.  Stage-2 promotion is a wrong-tool issue --
    its output is structured prose parsed by `ParsedAnalysis`
    heuristics, not JSON.
  - **Harmonising `OpenAIAdapter` to the whitelist pattern**
    (would be a minor breaking change if any caller passes
    `response_format=` positionally).

  11 new tests in `tests/test_gemini_grok_support.py` (4
  Gemini + 3 Grok in `TestGeminiAdapter` / `TestGrokAdapter`)
  + `tests/test_websearch.py` (2 Gemini dispatch + 2 Grok
  dispatch in `TestStructuredWebMapSignals`).  The Grok
  adapter class includes a critical regression guard
  `test_chat_raises_typeerror_on_unknown_kwarg` that locks
  in the typo-safety property.

  Live trace validation deferred until user provides
  `GEMINI_API_KEY` / `XAI_API_KEY` (same deferral as PR #29).

- **Gemini 3.1 Pro + Grok 4.3 web-search backend parity.**  After
  PR #28 brought `AnthropicWebSearchBackend` level with the OpenAI
  template (PRs #17 / #18 / #21 / #24), the two remaining
  backends were still on the pre-PR #18 baseline (3-line minimal
  prompt, no blocklist, no strict structured output, no token
  scaling).  On top of that, `GrokWebSearchBackend` was BROKEN
  in production: the prior code called `chat.completions.create`
  with a `search_parameters` body, which xAI started returning
  HTTP 410 Gone for on 2026-01-12 (deprecation of the legacy
  Live Search API).  Every Grok web-search call against the
  live xAI API failed until this PR.  Bundled into one PR per
  the user request; the two streams share the same parity
  template but differ in API mechanics.

  **Stream A -- `GeminiWebSearchBackend` rewrite**
  (`src/metacouplingllm/knowledge/websearch.py`):

  1. **Default model** bumped from `gemini-2.5-flash` to
     `gemini-3.1-pro-preview` per user request ("latest Gemini
     3.1 pro").  Per stored memory preference (cost is not a
     constraint; optimise for quality), Pro is preferred over
     Flash.  The earlier `gemini-3-pro-preview` was discontinued
     2026-03-09 and now redirects to 3.1 Pro per Google's
     official changelog, so 3.1 Pro is the safe default.
     Switch to `gemini-3.1-pro` (no `-preview` suffix) when GA
     lands.
  2. **Rich prompt template** copying OpenAI PR #18 verbatim:
     "web research collector" role, DO-NOTs for prose / off-topic
     answering, 6 source-selection rules, 5 grounding rules, and
     a 200-400 word `model_summary` requirement.  Closing
     instruction adapted to Gemini's native combined
     grounding + structured output (no submit_results dance like
     Anthropic).
  3. **Strict structured output via native combined config**:
     `tools=[{"google_search": {}}]` plus
     `response_mime_type="application/json"` and
     `response_schema=_OPENAI_WEB_SEARCH_RESULTS_SCHEMA` in one
     call.  Gemini 3.x supports schema + grounding in a single
     request (per the structured-output docs at
     `https://ai.google.dev/gemini-api/docs/structured-output`),
     so we re-use the same schema OpenAI uses and downstream
     `_normalise_backend_results` consumes both identically.
  4. **`thinking_config={"thinking_level": "high"}`** per user
     memory preference.  Gemini 3.1 Pro forces extended thinking
     ON regardless of caller config; this parameter controls
     the budget (`minimal` | `low` | `medium` | `high`).
  5. **Default `blocked_domains`** = `["reddit.com",
     "quora.com", "pinterest.com"]` (mirrors OpenAI PR #18 and
     Anthropic PR #28).  Applied as a post-hoc client-side
     filter via the new `_apply_domain_filters` method: Gemini's
     `google_search` tool does NOT expose server-side domain
     filtering, so we filter the structured response after the
     fact as a parity belt against the other two backends.
     Documented as a known parity gap in the docstring.
  6. **`max_output_tokens` scaling**: dataclass default bumped
     8192 → 12000 (matches OpenAI PR #24 and Anthropic PR #28);
     per-call computes `min(max(self.max_tokens,
     max_results * 1000), 64000)`.  The 64000 cap is Gemini 3.1
     Pro's hard output ceiling.
  7. **Silent-fallback warning** when the structured response is
     empty / malformed; backend then falls through to
     `_extract_gemini_grounding_results` (title + url only, no
     `model_summary`) so the call still returns something useful
     but the failure mode never goes silent.  Mirror of
     OpenAI PR #24 and Anthropic PR #28 diagnostics.

  **Stream B -- `GrokWebSearchBackend` MAJOR rewrite**
  (same file):

  1. **Endpoint migration**: `client.chat.completions.create(...)`
     → `client.responses.create(...)`.  xAI returned HTTP 410
     Gone for the prior endpoint since 2026-01-12; this was a
     production bug, not just a deprecation warning.  Still uses
     the OpenAI SDK with
     `base_url="https://api.x.ai/v1"` (no dedicated xAI SDK).
  2. **Default model** bumped from `grok-3` to `grok-4.3` per
     user request.
  3. **Dropped deprecated dataclass fields**: `search_mode`,
     `max_search_results`, `sources` (parameters of the legacy
     Live Search API, gone with the endpoint migration).
  4. **New built-in `web_search` tool spec**:
     `{"type": "web_search", "excluded_domains": [...],
     "enable_image_understanding": False}`.  xAI's `/responses`
     endpoint runs the search loop server-side and returns one
     final assistant message; no separate `tool_result` blocks
     are surfaced to the client.
  5. **Rich prompt template** copying OpenAI PR #18 verbatim,
     adapted with stronger language: `**REQUIRED**: you MUST
     use the web_search tool to gather sources; do not answer
     from training data.`  xAI's `/responses` endpoint does NOT
     expose a `tool_choice="required"` analogue (documented as
     a known parity gap), so prompt language is the only lever
     for forcing tool use.
  6. **Strict json_schema `response_format`** combined with the
     `web_search` tool (per
     `https://docs.x.ai/docs/guides/structured-outputs`).  Uses
     a new module-level constant
     `_GROK_WEB_SEARCH_RESULTS_SCHEMA` that mirrors the OpenAI
     schema PLUS a required `evidence_urls: string[]` field per
     result -- because xAI doesn't return separate tool_result
     blocks, citations must live inside the structured response
     itself.  `_normalise_backend_results` ignores the extra
     field gracefully.
  7. **New `reasoning_effort="high"`** field per user memory
     preference (`none | low | medium | high`; applies to Grok
     4.3).  **New `max_turns=5`** field (xAI's "balanced"
     deep-search cap; 1-2 quick / 3-5 balanced / 10+ deep).
     **New `enable_image_understanding=False`** field (default
     off for speed; text-only).
  8. **Default `excluded_domains`** = `["reddit.com",
     "quora.com", "pinterest.com"]` (mirrors OpenAI PR #18).
     xAI's API caps each domain list at 5 entries; the
     3-entry default fits comfortably, and the backend silently
     truncates to 5 if more are passed.  New constant
     `_MAX_DOMAIN_LIST_SIZE = 5` documents the limit.
  9. **`max_output_tokens` scaling**: dataclass default bumped
     8192 → 12000 (matches OpenAI PR #24, Anthropic PR #28,
     Gemini Stream A above); per-call computes
     `max(self.max_tokens, max_results * 1000)`.  Grok 4.3 has
     no documented hard ceiling.
  10. **Two-variant silent-fallback warning** distinguishing
      "web_search NOT invoked (model answered from training
      data; strengthen prompt)" from "web_search ran but
      structured output was empty / malformed".  Reads
      `response.server_side_tool_usage.web_search_count` to tell
      the two cases apart, then falls through to
      `_extract_grok_citation_results` if older xAI clients
      still surface a `citations` field.

  **Breaking changes for direct consumers** of either backend:
  - `GeminiWebSearchBackend(model=...)` default changed from
    `gemini-2.5-flash` to `gemini-3.1-pro-preview`.  Pin
    `model="gemini-2.5-flash"` explicitly if you need the
    cheaper / faster model.
  - `GrokWebSearchBackend(model=...)` default changed from
    `grok-3` to `grok-4.3`.  Pin `model="grok-3"` explicitly
    if you need it (though `chat.completions` Live Search is
    gone regardless, so most callers are effectively forced
    onto 4.3 + `/responses`).
  - Both backends now default to the
    `["reddit.com", "quora.com", "pinterest.com"]` blocklist.
    Pass an empty list to restore the prior unfiltered
    behaviour.
  - `GrokWebSearchBackend` dataclass fields `search_mode`,
    `max_search_results`, and `sources` are GONE.  Callers
    using them will get a `TypeError` at construction.

  **Live trace validation deferred** until the user provides
  `GEMINI_API_KEY` / `XAI_API_KEY` per their message ("I will
  attach related api when you finish this revision").  The
  PR #28 diagnostic pattern (`scripts/diagnose_anthropic_web_search.py`)
  can be adapted as `scripts/diagnose_gemini_web_search.py` /
  `scripts/diagnose_grok_web_search.py` for live bisecting
  when keys arrive.

  26 new tests in `TestGeminiWebSearchBackend` (12 cases) and
  `TestGrokWebSearchBackend` (14 cases) covering each axis
  above.  The Grok class includes a critical regression guard
  -- `test_search_uses_responses_endpoint_not_chat_completions`
  -- that uses a `_ChatCompletionsExploder` stub which raises
  `AssertionError` if the old endpoint is touched.

  Also deleted the 4 pre-existing
  `TestGrokWebSearchBackend` tests in
  `tests/test_gemini_grok_support.py` because they asserted
  parameters of the gone Live Search API (`extra_body`,
  `search_parameters.mode`, `search_parameters.max_search_results`,
  `search_parameters.sources`).  The new test class in
  `test_websearch.py` provides stricter coverage of the
  modern surface.  Module docstring updated to point readers
  to the new location.

- **End-to-end Claude API support across the pipeline (Stream B:
  pipeline strict-output).**  Stream A of this PR brings
  `AnthropicWebSearchBackend` to web-search-backend parity with
  OpenAI; this stream extends Claude support up the pipeline so
  that `AnthropicAdapter` can replace `OpenAIAdapter` for Stage-1
  web extraction without losing the strict-structured-output
  rigor that OpenAI gets via `response_format=json_schema`.

  Three coordinated changes in `src/metacouplingllm/llm/client.py`:

  1. **New optional field `LLMResponse.tool_uses`** (`list[dict]
     | None = None`).  Backward compatible -- consumers that
     only read `.content` and `.usage` continue to work.  When a
     Claude response contains `tool_use` blocks, the adapter
     populates this field so call sites can read structured
     output without re-parsing the raw provider response.

  2. **`AnthropicAdapter.chat()` opened for kwargs forwarding**
     via `**kwargs: Any` with a whitelist
     (`_ALLOWED_FORWARDED_KWARGS = {"tools", "tool_choice"}`).
     Unknown kwargs raise `TypeError` at the boundary so typos
     surface immediately rather than being silently swallowed.

  3. **Anthropic response extraction collects `tool_use` blocks**
     in addition to text blocks, populating
     `LLMResponse.tool_uses` when present.

  4. **Defensive belt against extended-thinking default drift**:
     new `AnthropicAdapter(extended_thinking=...)` constructor
     arg.  Defaults to `None` -> we do NOT send the `thinking`
     parameter -> Anthropic's documented default (extended
     thinking disabled) applies.  Callers who want extended
     thinking opt in explicitly with
     `extended_thinking={"type": "enabled", "budget_tokens": N}`.
     This locks in the current OFF behaviour against any future
     Anthropic API default change that could silently turn
     extended thinking on, which would balloon wall-clock for
     long-context calls (Stage-2's 63k-char system prompt would
     be especially sensitive).  Two new tests in
     `TestAnthropicAdapter` regression-guard both paths.

     Note: the 62-minute avocado-25 trace at
     `runs/avocado_2026-05-21_pr28_claude_25results_v4_retries/`
     was NOT caused by extended thinking (it was already off);
     it's just Sonnet 4.6's base latency on a 63k-char system
     prompt.  This belt is preventative, not remediation.

  Stage-1 extraction (`extract_web_map_signals` in
  `websearch.py`) gets a new Anthropic dispatch branch parallel
  to the existing OpenAI one.  When the LLM client is an
  `AnthropicAdapter`, the chat call now passes a new
  `_ANTHROPIC_WEB_MAP_SIGNALS_SUBMIT_TOOL` (wrapping the
  existing `_WEB_MAP_SIGNALS_SCHEMA`) via `tools=[...]` plus
  `tool_choice={"type":"tool","name":"submit_web_map_signals"}`
  to force Claude to call the submit tool.  The response parser
  now checks `response.tool_uses` first; on miss it falls back
  to `_extract_json_object(response.content)` for graceful
  degradation when Claude ignores the tool_choice.

  `scripts/trace_pipeline.py` gains a `LoggingAnthropicAdapter`
  class parallel to the existing `LoggingOpenAIAdapter`, so
  users who want to run end-to-end traces through Claude can
  swap adapters in `main()`.  Captures all the same artifact
  fields as the OpenAI logger, plus `response_tool_uses` for
  inspecting Claude's structured outputs.

  **Stage-2 (main framework analysis) and Stage-3
  (`_extract_map_data_from_analysis`) are unchanged.**  They
  already work today with `AnthropicAdapter` because neither has
  any backend-specific strict-mode dispatch -- both use plain
  chat with prompt-based JSON instructions.  Promoting Stage-3
  to strict mode for both backends would be a separable PR.

- **Bring `AnthropicWebSearchBackend` to parity with
  `OpenAIWebSearchBackend` after PRs #17 / #18 / #21 / #24.**
  Before this PR the Anthropic backend was significantly behind:
  a 3-line minimal prompt vs OpenAI's 45-line structured
  template, no default blocklist, no `tool_choice` forcing, no
  output-token scaling, no structured-JSON enforcement, no
  silent-fallback diagnostic, and `claude-opus-4-7` as the
  default model (expensive).  Eight changes brought it level:

  1. **Default model** bumped from `claude-opus-4-7` to
     `claude-sonnet-4-6` (~1.7-5× cheaper depending on
     input/output mix; Sonnet 4.6 still gets dynamic-filtering
     `web_search_20260209` via the existing
     `_WEB_SEARCH_MODEL_VERSIONS` table).
  2. **Rich prompt template** copied from OpenAI PR #18:
     "web research collector" role, DO-NOT clauses for
     prose/off-topic answering, 6 source-selection rules
     (peer-reviewed / government / international organizations
     preferred, avoid SEO/forums/duplicates, no padding), 5
     grounding rules (no inventing URLs / titles / dates /
     findings), and a 200-400 word `model_summary`
     requirement.
  3. **`submit_results` user-defined tool with `strict: True`**
     so Claude returns structured JSON via `tool_use` rather
     than free-form text.  Chosen over `output_config.format`
     json_schema directly because Anthropic's docs don't show
     an end-to-end example of strict-output mode combined with
     a server tool like `web_search`.  The tool schema mirrors
     `_OPENAI_WEB_SEARCH_RESULTS_SCHEMA` so both backends
     produce identical downstream shape.
  4. **`tool_choice = {"type": "tool", "name": "web_search"}`**
     forces the first turn to actually call `web_search`
     rather than answering from training data.  Anthropic
     analogue of OpenAI's `tool_choice="required"` (PR #18).
  5. **`max_tokens` scaling**: dataclass default bumped
     8192 → 12000 (matches OpenAI PR #24); per-call computes
     `effective_max_tokens = max(self.max_tokens,
     max_results * 1000)` so the model has room for 25+ rich
     summaries without truncation.
  6. **Default `blocked_domains`** = `["reddit.com",
     "quora.com", "pinterest.com"]` (mirrors OpenAI PR #18).
     Users who want everything can pass `blocked_domains=[]`.
  7. **Auto-include `code_execution` tool** when
     `web_search_20260209` is selected -- Anthropic
     requires the code execution tool to be enabled for
     dynamic filtering (per
     https://platform.claude.com/docs/en/agents-and-tools/tool-use/web-search-tool).
     Conditional on the resolved tool version; older
     `web_search_20250305` does not need it.
  8. **Silent-fallback diagnostic warning** when Claude
     responds without calling `submit_results`.  Falls
     through to the existing citation-based parser
     (`_extract_anthropic_web_results`) so the call still
     returns something useful, but prints a warning so the
     "Claude obeyed search but ignored structured-output
     instruction" failure mode never goes silent.  Mirror of
     OpenAI PR #24's silent-fallback warnings.
  9. **Default to BASIC `web_search_20250305` instead of auto-
     inferring from the model.**  The live avocado-25 trace at
     `runs/avocado_2026-05-21_pr28_claude_25results_v4_retries/`
     showed Claude bypassing `submit_results` when the dynamic-
     filtering tool (`web_search_20260209` + `code_execution`)
     was in use, causing the structured-output path to fall
     through to bare title+url+page_age strings (each
     `model_summary` was just `(page age: <date>)`).  The
     dynamic-filtering combination forces `tool_choice` to
     `code_execution`, and Claude's turn effectively ends after
     code_execution returns -- so submit_results never gets
     called and citations never get written either.

     Basic `web_search_20250305` lets Claude call submit_results
     naturally OR write text with citations (~150 chars each),
     both significantly richer than the dynamic-filtering
     fallback.  Auto-inference (`_infer_web_search_tool_version`)
     is retained for callers who explicitly opt in via
     `tool_version="web_search_20260209"`.

     Also strengthened the prompt's submit_results instruction:
     "**REQUIRED FINAL ACTION**: After completing your web_search
     invocations, you MUST call the `submit_results` tool ... Do
     NOT end your turn with just text or tool_results."  Recency-
     bias plus imperative language reduces the bypass rate even
     in edge cases.
  10. **Always stream `messages.stream()` for web search**, not
      `messages.create()`.  Anthropic's SDK refuses non-streaming
     requests whose estimated generation time would exceed 10
     minutes ("Streaming is required for operations that may
     take longer than 10 minutes"), but the upfront refusal is
     calibrated only to `max_tokens` -- it can't see the
     wall-clock cost of server-side `web_search` +
     `code_execution` sub-calls.  Even small `max_tokens` calls
     can exceed 10 minutes when Claude dispatches multiple
     sub-searches with dynamic filtering, in which case the
     non-streaming connection would hang / error mid-call.
     Surfaced in the live avocado-25 trace at
     `runs/avocado_2026-05-21_pr28_claude_25results/` where
     `AnthropicWebSearchBackend.search()` was rejected upfront
     by the SDK.  `messages.stream(...).get_final_message()`
     returns the same `Message` shape so the downstream parsing
     path is unchanged regardless of request size.  The
     threshold pattern from `AnthropicAdapter` (text-only chat)
     doesn't transfer here because plain chat has predictable
     `max_tokens`-bound wall-clock while server tools don't.

  Two live-trace-surfaced bugs also fixed alongside (8) and (9):

  - **Wrong `code_execution` tool version**: `code_execution_20250522`
    was incorrect; the version Anthropic actually pairs with
    `web_search_20260209` is `code_execution_20260120`.  The
    API rejected the call with a 400 listing the correct version.
  - **Wrong `tool_choice` routing for dynamic filtering**: when
    `web_search_20260209` is in the tools list,
    `web_search` is NOT directly callable by the model -- it
    can only be invoked from inside `code_execution`.
    `tool_choice` now routes to `code_execution` when dynamic
    filtering is on (`web_search_20260209`) and stays on
    `web_search` for the basic `web_search_20250305`.

  New module-level constant
  `_ANTHROPIC_WEB_SEARCH_SUBMIT_RESULTS_TOOL` holds the tool
  definition; new helper
  `_extract_anthropic_submit_results_tool_use` parses
  Claude's `tool_use` block with `name == "submit_results"`
  and hands the results off to the existing
  `_normalise_backend_results` normaliser.

  **Breaking change for direct consumers**: `model` default
  changed from `claude-opus-4-7` to `claude-sonnet-4-6`.  Pass
  `model="claude-opus-4-7"` explicitly if you need Opus for
  quality reasons.  Also: `blocked_domains` default changed
  from `None` to `["reddit.com", "quora.com", "pinterest.com"]`.
  Pass `blocked_domains=[]` to restore the prior unfiltered
  behavior.

  11 new tests in `TestAnthropicWebSearchBackend` covering
  each axis above.  Pre-existing tests updated to handle the
  new 2-3-tool structure (web_search + submit_results +
  optional code_execution) in the captured tools list.
- **Raise the per-summary truncation cap that Stage-3 sees from
  200 to 2500 chars (new module-level constant
  `_MAX_WEB_SUMMARY_CHARS_IN_MAP_PROMPT`).**  Stage-3
  (`_extract_map_data_from_analysis`) reads parsed analysis text
  plus a per-result web-summary excerpt to extract structured
  map data (`receiving_countries`, flows, etc.).  The per-result
  excerpt was hardcoded to `summary[:200]` chars — sufficient
  for short hand-written summaries but too short for the new
  PR #18 / PR #24 era of 200-400 word `model_summary` blocks,
  especially destination-list sources like UN Comtrade tables
  where country names typically appear several hundred chars
  into the summary after dataset / methodology boilerplate.

  Concrete example from the avocado-25-results trace's W3
  (Mexico avocado exports by partner country in 2024): the full
  W3 summary lists 19+ destinations (USA, Canada, Japan, El
  Salvador, Honduras, Costa Rica, Spain, UAE, Netherlands,
  Kuwait, Guatemala, Bahrain, Hong Kong, China, Saudi Arabia,
  France, Belize, UK, Singapore, Malaysia).  The first 200
  chars of that summary contain only dataset-description
  boilerplate — Stage-3 saw none of those countries via the
  web-summary path and could only pick up destinations from the
  parsed analysis text (which itself emphasised a subset).
  Result: Stage-3's `receiving_countries` was `['USA', 'SLV',
  'HND']`, dropping Canada and Japan even though Stage-1 had
  emitted both at 0.9 confidence.

  At 2500 chars the substantive content of typical
  `model_summary` blocks survives intact, including the
  destination-list sentences.  The cap remains as a defensive
  ceiling against pathologically long future summaries.
  Stage-3 still has no count cap of its own — it just gets
  better grounding data.

  Note: Stage-3's destination count is governed by LLM judgment
  (no explicit cap in the prompt, schema, or post-extraction
  validator).  Stage-1 retains its `max_targets=6` and
  `min_confidence=0.7` defaults.  Relaxing Stage-1's caps was
  considered for the same PR but the Stage-3 truncation was
  identified as the higher-leverage bottleneck on map coverage,
  so PR #26 ships that alone.  Stage-1 cap tuning is a separate
  follow-up if traces still show sparse coverage.
- **Run BOTH ADM1 (subnational) and country-level pericoupling
  validations when an analysis spans both scales.**  Previously
  `_validate_pericoupling()` returned early once
  `_validate_adm1_pericoupling()` succeeded, so the country-level
  pair classification (e.g., MEX↔USA pericoupled, MEX↔CAN
  telecoupled) was never computed when a focal subnational
  region was also identified.  The formatted output for the
  Mexico-avocado trace (focal Jalisco/MEX014 + destinations
  USA/CAN/JPN/SLV/HND) showed only the subnational neighbor
  list — the user-visible country-level pair classifications
  silently disappeared.

  Both validators now run independently:
  - `parsed.pericoupling_info` holds the ADM1 result (as
    before); stays None when no subnational focal region is
    resolvable.
  - `parsed.country_pericoupling_info` is a new
    `ParsedAnalysis` field that holds the country-level pair
    classification result; stays None when only one country
    is detected.

  The formatter (`output/formatter.py`) renders both blocks
  when both are populated:
  ```
  PERICOUPLING DATABASE VALIDATION (SUBNATIONAL)
  ----------------------------------------
    Focal Region: Jalisco (MEX014)
    Domestic Neighbors: Aguascalientes (MEX001), ...
    Note: LLM classification is consistent with the ADM1 pericoupling database.

  PERICOUPLING DATABASE VALIDATION
  ----------------------------------------
    Focal Country: Mexico (MEX)
    Pair Results:
      Mexico (MEX) ↔ United States (USA): PERICOUPLED
      Mexico (MEX) ↔ Canada (CAN): TELECOUPLED
      Mexico (MEX) ↔ Japan (JPN): TELECOUPLED
    Note: LLM classification is consistent with the pericoupling database.
  ```

  **Breaking change for direct consumers of
  `parsed.pericoupling_info`**: for country-only analyses (no
  ADM1 focal region resolvable), the country-level result now
  lives in `country_pericoupling_info`, not `pericoupling_info`.
  Code that read `pericoupling_info` for country results in
  that case needs to also check `country_pericoupling_info`.
  Internal callers and the formatter are updated; external
  consumers of the public `AnalysisResult.pericoupling_info`
  attribute should migrate.
- **Wire Stage-1 supranational flows through to the map
  renderer (deferred piece from PR #24).**  PR #24 fixed the
  Stage-1 validator so LLM-emitted supranational receivers and
  flows (European Union / ASEAN / USMCA / NAFTA) survive
  normalization with their `target_supranational_members` and
  `source_supranational_members` fields populated.  But
  `_structured_web_flow_dicts()` — the function that repackages
  Stage-1's flows into the renderer's input shape — was still
  stripping the supranational marker fields, so a Stage-1
  Brazil → European Union flow arrived at the renderer with
  `target: "European Union"` and no member list.  The renderer
  would then try to look up "European Union" in the world
  shapefile, fail, and silently draw no arrow.

  This change makes `_structured_web_flow_dicts()` preserve
  `target_supranational`, `target_supranational_members`,
  `source_supranational`, and `source_supranational_members`
  when present, matching the shape produced by Stage-3's
  `_extract_map_data_from_analysis`.  Also updated the type
  hints on `_merge_map_flows` from `dict[str, str]` to
  `dict[str, object]` (the member-list values are not strings)
  and audited the merge logic to confirm flow dicts are
  appended verbatim without field-stripping — so supranational
  fields survive the merge.

  **User-visible effect today**: none, because Stage-3 reliably
  emits supranational flows itself and wins the merge dedupe on
  `(category, direction)`.  The value is defense-in-depth: if
  Stage-3 ever fails to emit an EU flow (timeout, JSON parse
  error, prompt regression), Stage-1's repackaged flow now
  carries enough data for the renderer to draw the EU bloc on
  its own.

  Source-side supranational rendering (e.g., a EU → Brazil
  capital flow) is still gated on the renderer's
  `_resolve_flow_endpoints` only consulting
  `target_supranational_members`.  The data plumbing is correct
  now so a future worldmap.py change can extend source-side
  bloc rendering without re-touching the Stage-1 path.
- **Fix two distinct silent failures in Stage-1 web extraction
  that both surfaced as `Structured web extraction accepted 0
  receiving systems`.**

  (a) `OpenAIWebSearchBackend.search()` previously capped output
  at `max_output_tokens=8000`, sufficient for 10 results × 300
  word summaries but too small for 25+ results.  Excess content
  was truncated, the JSON output broke, and the code silently
  fell back to URL-only source citations (titles==urls, no
  summaries).  Downstream Stage-1 then correctly emitted zero
  receivers because it had no grounded data — 21 of 25 evidence
  cards literally said *"No usable summary text was provided
  for map extraction"*.  Fix: bump the dataclass default to
  `max_output_tokens=12000` and scale
  `effective_max_tokens = max(self.max_output_tokens,
  max_results * 1000)` per call (1000 tokens/result is a
  generous margin: ~500 for a 400-word summary plus ~500 for
  title/URL/JSON overhead).  Also added two diagnostic warnings
  so the silent fallback never goes unnoticed again: a partial
  warning when `parsed_results < 0.5 * max_results` and a total
  warning when JSON parsing failed entirely but URL-only source
  citations exist.

  (b) `_normalise_country_entry` and `_normalise_flow_entry` in
  `extract_web_map_signals` silently dropped any LLM-emitted
  receiver/flow whose country was a supranational name
  (European Union / ASEAN / USMCA / NAFTA), because
  `resolve_country_code()` only checks ISO codes and
  `_ALIASES`, not `_SUPRANATIONAL_ALIASES`.  PR #22 taught the
  Stage-1 LLM to emit such names; the validator was never
  updated to accept them, making PR #22 effectively a no-op for
  Stage-1.  Observed in the Brazil-soy → EU trace where the
  raw LLM response correctly emitted
  `{"country":"European Union","confidence":0.98,...}` and
  three flows targeting/sourcing the EU, but the final
  pipeline output had `receiving_systems: []` and `flows: []`.

  Fix: extracted a `_resolve_country_or_supranational()` helper
  that tries `resolve_country_code()` first; on None, falls
  back to `expand_supranational()`.  If supranational lookup
  succeeds, the entry's `country` is set to the canonical
  display name (e.g., `"European Union"` regardless of whether
  the LLM said `"EU"`, `"e.u."`, or `"european union"`), and a
  `supranational_members` field is added carrying the member
  ISO codes.  Flow entries get parallel
  `source_supranational_members` / `target_supranational_members`
  fields.

  Downstream `_structured_web_receiving_codes()` and
  `_structured_web_spillover_codes()` (`core.py:~4180`) detect
  the `supranational_members` field and expand to the member
  ISO codes when building their "map-ready code set" return
  values — keeping the set semantically pure (ISO codes only)
  so existing callers that pass codes through
  `get_country_name()` or `ISO_ALPHA3_NAMES` continue to work.
  The display name "European Union" stays in the source
  signals dict so `format_web_map_signals_context()` (no code
  change needed there) renders it verbatim into the Stage-2
  analysis prompt.

  **Conservative scope:** `_structured_web_flow_dicts` and the
  Stage-1 → map renderer integration for supranational flow
  endpoints are deferred to a future PR — Stage-3's existing
  supranational path (`core.py:2987-3022` + PR #23's dissolved
  bloc renderer) already handles the rendering correctly, so
  the conservative scope leaves nothing visibly broken and
  avoids a risky `_merge_map_flows` audit.
- **Teach both extraction prompts that supranational unions are
  valid map targets.**  The supranational-rendering infrastructure
  has been in place since PR #5/#6 — `countries.py` knows
  `"european union"` → 27 ISO codes, `expand_supranational()`
  resolves it, `target_supranational` fields drive single-region
  rendering at the union centroid with all members highlighted —
  but neither extraction LLM was ever told it could *emit* a union
  name.  The Stage-1 web-extraction prompt
  (`extract_web_map_signals` in `websearch.py:~1817`) said *"Only
  include countries"* and used a schema field literally named
  `country`.  The Stage-3 map-extraction prompt
  (`_extract_map_data_from_analysis` in `core.py:~2790`) opened
  with rule #1 *"Use ISO alpha-3 codes (USA, BRA, CHN, MEX, JPN,
  etc)"* and defined `target` as *"ISO alpha-3 code of the
  importer/receiver"*.  Result: when the May-2026 avocado trace's
  W8 source named *"United States, Japan, Canada, and the
  European Union"* as virtual-water destinations, the Stage-1 LLM
  obediently dropped EU and emitted only USA/CAN/JPN; the Stage-3
  LLM never had a chance to surface it either.  The existing
  fallback at `core.py:2987-3022` ("the LLM may slip past rule
  #1") rarely fired because nothing was inviting the LLM to slip.

  Both prompts now ship a paragraph naming the four accepted
  unions (European Union, ASEAN, USMCA, NAFTA) and the
  prefer-specific-members rule that avoids double-counting when
  a source names both the umbrella and its constituent states.
  No schema, parser, or renderer changes — the field types were
  already `string` and the downstream code was ready and waiting.
- **Render supranational map targets as a dissolved bloc colored by
  focal adjacency, not as 27 individually-outlined members.**  Before
  this change, `_draw_supranational_highlight` in `worldmap.py`
  plotted each EU member country separately with a translucent
  `#1f77b4` blue overlay and `#0d3a66` outlines — so the internal
  France/Germany/etc. borders inside the bloc remained visible and
  every member appeared with its own coupling-category color
  underneath (a mix of pericoupling / telecoupling / na depending on
  the focal country).

  The function now:

  1. Dissolves all member geometries into a single (multi)polygon via
     `GeoSeries.union_all()` (replacing the now-deprecated
     `unary_union` attribute), so the bloc reads as one logical
     region with a single outline and no internal member borders.
  2. Picks the bloc's fill color by intersecting member ISOs with the
     focal country's pericoupled neighbors (already encoded in the
     `classification` dict):
     - **Pericoupling (bright green)** when at least one member
       shares a border with the focal country.  Russia → EU
       (Finland, Estonia, Latvia, Lithuania, Poland all border RUS),
       Norway → EU, Switzerland → EU all paint the EU as
       pericoupling.
     - **Telecoupling (light blue)** otherwise.  Brazil → EU,
       China → EU, Mexico → EU all paint the EU as telecoupling.
  3. Uses full alpha (1.0 instead of 0.30) so the bloc fill covers
     whatever per-country color the base classification layer
     assigned to individual members — completing the single-region
     illusion.

  Applies uniformly to all four supported unions (European Union,
  ASEAN, USMCA, NAFTA).  Signature of
  `_draw_supranational_highlight` now takes the additional
  `classification: dict[str, str]` and `colors: CouplingColors`
  positional args; the single call site in `_render_map` already had
  both in scope and was updated to pass them through.
- **Richer prompt + `tool_choice="required"` +
  `max_output_tokens` + default `blocked_domains` on
  `OpenAIWebSearchBackend`.**  The web-search call's prompt was
  previously a terse 8-line instruction ("Search the web for: X.
  Return JSON.  Keep summaries concise.").  Replaced with a
  structured ~40-line template covering: explicit role ("web
  research collector"), DO-NOTs (don't answer the user's
  question, don't write a prose report), 6 source-selection
  rules (prefer peer-reviewed / government / international
  organizations, avoid SEO/forums/duplicates, no padding), 5
  grounding rules (no inventing URLs/titles/dates/findings, no
  false attribution).  Each `model_summary` now asked for
  200–400 words instead of "concise and factual".  Adapted from
  a ChatGPT recommendation; the "evidence card rules" pieces
  stay in `extract_web_map_signals` (call #2) since our
  architecture keeps search and extraction separate.

  Parameters added to the OpenAI Responses-API call:
  - `tool_choice="required"` (was `"auto"`) — forces the model
    to invoke `web_search` rather than answering from training
    data.  Verified against OpenAI docs.
  - `max_output_tokens=8000` — defensive ceiling on response
    size; configurable via `OpenAIWebSearchBackend(client=...,
    max_output_tokens=N)`.

  Default `blocked_domains` on `OpenAIWebSearchBackend` now
  `["reddit.com", "quora.com", "pinterest.com"]` instead of
  `None`.  Users who want everything can pass
  `blocked_domains=[]` or supply their own list.

  Also added grounding rules to `extract_web_map_signals`'s
  user prompt (no inventing flows, no extrapolation across
  sources).  Same content discipline as the search call.

  ChatGPT also suggested `web_search_call.results` in
  `include` and `text.verbosity="high"` — I verified neither
  against current OpenAI docs and SKIPPED both.

  Note: this bullet was originally part of PR #18 but failed to
  reach `main` because PR #18 was merged after PR #17 had already
  squash-merged, leaving its commits stranded on the deleted
  PR #17 branch.  Re-landed via PR #21 as a cherry-pick.
- **Constrained vocabulary + anti-coining rules for §5 Cross-coupling
  Interactions, plus a prose-paragraph instruction for the Coupling
  transformations bullet.**  Two related issues observed in the
  May-18 avocado trace's §5 output:

  1. The LLM emitted **"Pericoupling leakage"** as a coined compound
     term that does not appear in the bundled corpus, the framework
     knowledge layer, or any prompt.  The model combined "pericoupling"
     + "leakage" by analogy because the concept (displacement to
     adjacent regions) needed a label and the framework didn't
     provide one.  §7 EVIDENCE COVERAGE (shipped in PR #17) detects
     such extrapolation post-hoc but doesn't prevent it.

  2. The **"Coupling transformations"** bullet rendered as visually
     empty because the LLM wrote its content as nested sub-bullets
     (Coupling / Potential decoupling / Recoupling), and
     `_extract_bullets` (`parser.py:405-414`) `.strip()`s leading
     whitespace, promoting nested children to siblings of the
     empty parent.

  Both fixes are scoped to §5 only (sending/receiving/spillover
  terminology in §2-§4 is legitimately framework-defined and not
  affected).

  §5 now ships with an **approved vocabulary** drawn from the
  metacoupling framework (Liu 2017 / Liu 2023) and corpus papers
  (Yang et al. 2018 on feedback; Zhao et al. 2021 on synergies and
  tradeoffs across SDGs in a metacoupled world; broader telecoupling
  literature on displacement and cascading effects).  Accepted
  terms include: Amplification, Offset, Spatial tradeoffs, Temporal
  tradeoffs, Synergies, Cascading effects, Feedback loops,
  Displacement, Coupling transformations (with the four phases
  noncoupling/coupling/decoupling/recoupling), and Spillover
  effects.  An explicit rule forbids coining compound terms by
  combining framework concepts with environmental-economics jargon
  (the "pericoupling leakage" / "telecoupling resilience" /
  "metacoupling efficiency" failure mode), with the "pericoupling
  leakage" case listed verbatim as a counter-example so the LLM
  sees the specific compound it is being told to avoid.

  The §5 Coupling transformations bullet now explicitly instructs
  the LLM to weave the four phases into a single prose paragraph
  rather than splitting into nested sub-bullets.  The instruction
  explains the reason ("the downstream parser flattens nested
  bullets, which causes the parent bullet to render as visually
  empty while its children appear as orphan siblings") so the LLM
  understands the constraint rather than just being told to obey
  it.

  A **Coverage rule** added below the vocabulary list explicitly
  marks it as a permissive menu (allowed terms) rather than a
  prescriptive checklist (required terms), heading off the
  LLM-as-checklist failure mode where the model would force weak
  claims about every listed term to feel "complete".  Typical §5
  output is 2-4 of the 9 approved interactions, not all 9.

  Deeper parser-level fix — making `_extract_bullets` preserve
  nesting and the formatter render parent → child trees — is
  deferred to a separate PR.
- **Clarify in the framework prompts that pericoupling adjacency
  works at any spatial scale, not just ADM1 / national.**  The
  prior wording in `METHODOLOGY_LAYER` and the §3 pericoupling-
  section gate (`OUTPUT_FORMAT_LAYER`) listed examples as
  "neighboring states/provinces, shared watersheds, cross-border
  ADM1 regions" — reading as if pericoupling were capped at those
  scales.  In principle pericoupling applies to any pair of
  geographically adjacent units (villages, municipalities,
  ecosystems, nature reserves, etc.).  The package's bundled
  pericoupling databases and map renderers still focus on country
  and ADM1 — the examples in the prompts have not been expanded,
  only the qualifier "at any spatial scale" was added so the LLM
  doesn't reject a legitimately sub-ADM1 pericoupling claim on the
  grounds of scale.

  Out of scope for this PR: tightening the "shared watersheds"
  example to make clear that watershed-sharing only qualifies as
  pericoupling when the systems are ALSO geographically adjacent —
  Nile basin Rwanda↔Egypt is co-basin but not adjacent, so
  telecoupling.  Deferred per user request; addressable in a
  follow-up PR.
- **Strict `json_schema` mode for OpenAI web-search +
  `extract_web_map_signals`.**  Both calls previously asked the
  model to emit JSON via prompt instructions and relied on
  `_extract_json_object` to recover from malformed output.  They
  now declare an explicit schema and OpenAI guarantees the
  response conforms.
  - `OpenAIWebSearchBackend.search()` adds a
    `text={"format": {"type": "json_schema", "name":
    "web_search_results", "strict": True, "schema": ...}}` payload
    to the Responses-API call (schema defined as the
    module-level `_OPENAI_WEB_SEARCH_RESULTS_SCHEMA`).
  - `extract_web_map_signals` detects when the client is an
    `OpenAIAdapter` (or subclass) and passes
    `response_format={"type": "json_schema", "json_schema": {
    "name": "web_map_signals", "strict": True, "schema": ...}}`
    via the adapter's new keyword-only `response_format`
    parameter (schema is the module-level
    `_WEB_MAP_SIGNALS_SCHEMA`).
  Non-OpenAI clients keep the prompt-based JSON path unchanged.
  The `_extract_json_object` fallback parser stays as a
  defensive measure for both paths.
- **`OpenAIAdapter.chat()` gains a keyword-only `response_format`
  parameter.**  Optional; when omitted the kwarg is not sent to
  the OpenAI SDK so existing callers don't break.  The retry
  paths (temperature, max_completion_tokens, capped max_tokens,
  rate-limit backoff) carry `response_format` through unchanged.

### Changed (breaking — web-search return shape)

- **Web-search backend dict key renamed: `snippet` → `model_summary`.**
  All four web-search backends (OpenAI, Anthropic, Gemini, Grok)
  and the DuckDuckGo fallback now return result dicts shaped
  `{title, url, model_summary}` instead of `{title, url, snippet}`.
  The rename makes the field name match its actual contents: for
  OpenAI / Gemini / Grok the model writes a summary in response
  to a prompt instruction, and for the DDG fallback the field
  carries the page-metadata body — none of these are verbatim
  page text in the way "snippet" implied.  Only the Anthropic
  backend's `model_summary` is actually a verbatim excerpt
  (Claude's `cited_text` field), and the rename leaves a
  docstring note on
  `_extract_anthropic_web_results` calling that out so callers
  aren't misled.

  Migration: callers reading `result.web_results[0]["snippet"]`
  will get `KeyError`; switch to `["model_summary"]`.
  `_normalise_backend_results` silently maps a legacy `snippet`
  key from upstream provider responses to `model_summary` so
  half-migrated providers don't break during transition.  The
  three downstream consumers in `core.py`
  (`format_web_context`, `_format_web_sources`,
  `_extract_map_data_from_analysis`) accept either key for the
  same reason.

### Added

- **`OpenAIWebSearchBackend.blocked_domains`** parameter (mirrors
  the existing Anthropic backend's `blocked_domains` field).
  Passed to OpenAI's web_search tool as `filters.blocked_domains`.
  Useful for excluding low-quality sources (Reddit, Quora, content
  farms) from search results.  Coexists with `allowed_domains`
  when both are set.  Gemini and Grok backends do not yet support
  blocklists — deferred (their APIs would need post-filtering).
- **Evidence cards, combined coverage notes, and suggested
  follow-up queries** surface from the web-search + main-analysis
  pipeline:
  - **`evidence_cards`** on `web_map_signals`: a new top-level
    array emitted by the web-extraction LLM call
    (`extract_web_map_signals`).  One entry per web result, each
    with `source_id` (W1..Wn), `claims_supported` (1–4 specific
    factual phrases), `relevance_score` (0.0–1.0), and
    `source_type` (academic / government / news / industry /
    NGO / etc.).  Lets downstream consumers weight sources by
    type and reason about which source supports which claim.
  - **`AnalysisResult.evidence_coverage_note`** (`str`):
    self-assessment by the MAIN analysis LLM call (§7 in the
    output format) of where the analysis is well-grounded vs
    thin.  Considers BOTH RAG literature AND web evidence,
    since only the main call sees both streams — a web-only
    coverage note would be misleading because the RAG corpus
    may cover gaps the web search missed.  Mirrored from
    `parsed.evidence_coverage_note`; rendered into `formatted`
    as a "7. Evidence Coverage" block; empty string when the
    LLM omitted §7 (backward-compatible).
  - **`AnalysisResult.suggested_followup_queries`** (`list[str]`):
    3–5 short web-search query strings the web-extraction LLM
    proposes to fill gaps in the WEB evidence (RAG-side gaps are
    discussed in `evidence_coverage_note` instead).  Surfaces in
    `formatted` as a bullet footer beneath the EVIDENCE COVERAGE
    block.  Users can run them manually via
    `assistant.refine(query)` or programmatically for auto-
    deepening loops.
- **`web_search_max_results` default raised from 5 to 10** to
  give the new combined coverage assessment richer evidence by
  default.  Token cost on the search call rises ~30–50%; users
  who want the previous behaviour can pass `web_search_max_results=5`
  explicitly.

- **`AnalysisResult.flow_parse_warnings`** — list of flow-direction
  strings the legacy regex map path could not resolve into endpoints.
  Each entry carries `direction`, `category`, and `reason`. Empty when
  all flows parsed cleanly. Mirrored to `logger.warning(...)` at the
  moment of failure. Only populated by the legacy text-extraction map
  path; the structured (`parsed.map_data`) path keeps its existing
  `logger.debug` for dropped entries.
- **Supranational entity recognition** in the flow resolver: `EU`
  (27 members), `ASEAN` (10 members), and `NAFTA` / `USMCA`
  (3 members) are now recognised as flow endpoints.  The map
  renderer treats them as **single regions** — the union centroid
  is the arrow target, member countries are highlighted with a
  translucent overlay, and the arrow is labelled with the
  supranational name.  Expansion is **conditional**: if any member
  country is already mentioned in the same endpoint context, the
  supranational mention is treated as redundant and skipped to avoid
  visual double-counting.  New helpers `expand_supranational(name)`
  and `supranational_display_name(member_codes)` live in
  `metacouplingllm.knowledge.countries` (internal — not exported
  from the top-level package).
- **Supranational recognition on the primary map path**
  (`_extract_map_data_from_analysis`).  When the structured-extraction
  LLM call returns a flow whose `target` field is an umbrella name
  (`"European Union"`, `"EU"`, `"ASEAN"`, `"NAFTA"`, `"USMCA"`)
  rather than an ISO 3166-1 alpha-3 code, the post-processor stamps
  the same `target_supranational` / `target_supranational_members`
  fields the resolver path stamps.  Same conditional rule applies —
  the umbrella is dropped if any member ISO code is already in
  `receiving_countries` or `spillover_countries`.  Closes the gap
  where these flows were silently dropped on the primary path even
  after the resolver-path single-region rendering was added.
- **Flow-category alias table** (`_FLOW_CATEGORY_ALIASES`).  LLMs
  occasionally emit narrower or near-synonym category labels (e.g.
  `"goods"`, `"money"`, `"electricity"`, `"tourism"`, `"livestock"`,
  `"info"`) instead of the six canonical Liu 2017 categories.  The new
  module-level alias table maps 37 such labels onto the right canonical
  bucket (`matter` / `capital` / `information` / `energy` / `people` /
  `organisms`) before validation, so the flow survives instead of being
  silently dropped.  A new `_normalize_flow_category(raw)` helper does
  the lookup; both `_extract_map_data_from_analysis` and the structured-
  extraction supplement now use it (replacing two ad-hoc local rewrite
  dicts).  Genuinely ambiguous terms (`power`, `products`, `resources`,
  `services`, `economic`, `seeds`, `crops`) are still rejected on
  purpose, with rationale in code comments.

### Changed

- **`MetacouplingAssistant._resolve_flows_for_map`** now returns
  `tuple[list[dict], list[dict]]` — `(resolved_flows, parse_warnings)` —
  rather than a bare `list`.  Same for
  `_resolve_flows_for_adm1_map`.  Internal helpers; this is not part
  of the user-facing API, but downstream code that imports them must
  adopt the tuple unpack.
- Flow dicts emitted for supranational targets now carry two extra
  keys: `target_supranational` (the canonical display name) and
  `target_supranational_members` (the list of member ISO codes).
  Country-pair flows are unchanged.
- Loosened three prompt-budget caps in
  `_extract_map_data_from_analysis` so the second LLM call sees
  enough context to recover bilateral specifics:
  - Flow `description` cap raised from 100 → 500 chars.  Bilateral
    country lists ("Henan, Shanghai, Liaoning, …") frequently sit
    past char 100 and were being truncated mid-word.
  - System text cap raised from 400 → 800 chars **and** fields are
    now emitted in priority order — `name` and `geographic_scope`
    first, then `human_subsystem` / `natural_subsystem` /
    `description`, then any unexpected fields.  Guarantees the
    high-signal `geographic_scope` field reaches the LLM even when
    subsystems are verbose.
  - Hardcoded `[:10]` cap on web snippets replaced with a new
    module constant `_MAX_WEB_SNIPPETS_IN_MAP_PROMPT = 100`.  The
    user's `web_search_max_results` setting now flows through to the
    map-extraction LLM (previously, anything above 10 was silently
    dropped at the extraction step).  The constant acts purely as a
    defensive ceiling against pathological configs.
  No user-facing API changes; existing callers and tests unaffected.
- Loosened the draft-summary caps in
  `_structured_extract_supplement` to mirror the
  `_extract_map_data_from_analysis` changes above:
  per-flow `description` cap 100 → 500 chars, and the three
  per-system subfield caps (`human_subsystem`, `natural_subsystem`,
  `geographic_scope`) each raised 120 → 400 chars.  These caps
  govern the brief recap of the parsed analysis that the supplement
  LLM uses to identify what's already covered, so undersized caps
  were producing duplicate "additional mention" suggestions for
  content that the draft already had.  Per-call cost increase is
  ~+400 tokens; no user-facing API changes.
- Brought RAG-only LLM passage budget to parity with the framework
  path.  `_analyze_rag_only` previously sent only ~300-char excerpts
  of each retrieved chunk to the LLM (via the dual-purpose
  `format_evidence` helper), while the framework analysis path sent
  up to 5000 chars per passage via the prompt builder — a ~16×
  asymmetry that severely under-used retrieved evidence in RAG-only
  mode.  `format_evidence` gains a new keyword-only `max_chars`
  parameter; when provided, the function renders FULL chunk text
  capped at that value instead of a short excerpt.
  `_analyze_rag_only` now passes `max_chars=_LLM_PASSAGE_MAX_CHARS`
  (5000) so the LLM sees the same per-passage budget as the
  framework path.  Display-path callers (`_build_result`) keep the
  default `max_chars=None` and continue to render readable
  ~300-char excerpts for the user.  The new parameter is
  keyword-only with a backward-compatible default; existing public
  callers of `format_evidence` are unaffected.

### Fixed

- The user-facing map-type notice ("a country-level metacoupling
  map has been generated" vs the ADM1 variant) could disagree with
  the actual rendered figure when the renderer's ADM1 attempt fell
  through to a country-level map (e.g. `plot_focal_adm1_map`
  raised on a missing shapefile region).  Previously `_build_result`
  recomputed the type from `parsed.map_data["adm1_region"]` +
  `_resolve_adm1_from_analysis` — the same inputs the renderer
  used to *try* the ADM1 path — so a silent fall-through left the
  notice claiming "ADM1" while `result.map` was a country-level
  figure.  Fixed by recording the actually-rendered type in
  `self._last_map_type` from inside `_generate_map` (only after a
  successful render) and having `_build_result` read it instead of
  recomputing.  No public API change.
- `_validate_adm1_pericoupling` was unconditionally writing
  `"LLM classification is consistent with the ADM1 pericoupling
  database."` to `parsed.pericoupling_info["note"]` regardless of
  whether the LLM's classification actually agreed with the
  database — the country-level sibling `_validate_pericoupling`
  performed a real comparison, but the ADM1 path was just stamping
  the text.  Fixed by classifying each mentioned ADM1 region
  (via `_extract_mentioned_adm1_from_text`) against the focal's
  database neighbour set and emitting a "Consider revising."
  warning on mismatch (mirrors the country-level branching
  exactly: warn if DB shows adjacent partners but the LLM didn't
  classify as pericoupling, or if DB shows only non-adjacent
  partners but the LLM didn't classify as telecoupling).  When no
  `coupling_classification` text is present the note now reads
  neutrally instead of claiming consistency that wasn't checked.
  No public API change.
- `_chunk_markdown` (the corpus chunker) was capping chunks by
  **word count** (`max_chunk_words=250`) but had no char-based
  cap.  When a section's body was dense (long table cells,
  scientific identifiers, URLs), 250 "words" could exceed 5,000
  chars — and those chunks were then silently truncated by
  `format_evidence(..., max_chars=_LLM_PASSAGE_MAX_CHARS)` when
  the LLM received them.  Measured on the bundled corpus, 6 of
  10,032 chunks exceeded 5,000 chars; the worst was 13,488 chars
  (losing ~63% of its content on truncation).  Fixed by adding a
  new `_CHUNK_HARD_CHAR_CAP = 5000` constant plus a
  `_split_oversized` helper that splits oversized chunks on the
  best available boundary (paragraph → sentence → word → hard
  char position).  After this change every chunk fits within the
  LLM passage budget by construction.  Bundled
  `chunk_embeddings.npy` and `chunk_embeddings.manifest.json` are
  regenerated as part of this PR (chunk count rises from 10,032
  to 10,039; manifest fingerprint changes).
- `tests/test_rag.py::TestEmbeddingRetriever::test_query_returns_results_with_precomputed`
  fed the retriever a 384-dim precomputed embedding array but never
  overrode the embedder, so `EmbeddingRetriever.query()` lazily
  loaded the real `BAAI/bge-base-en-v1.5` model (768-dim) and the
  cosine-similarity matmul (`chunk_vecs @ query_vec`) failed with a
  `ValueError: ... size 768 is different from 384`.  Fixed by
  installing a small `_FakeEmbedder` that emits a 384-dim query
  vector — mirroring the pattern the sibling
  `test_deduplicates_by_paper_key` already uses.  No production-code
  change; pre-existing test bug (default embedding model has been
  768-dim since the initial commit) that was masked whenever
  `fastembed` was unavailable, since the whole class is skipped
  under `@pytest.mark.skipif(not HAS_FASTEMBED, ...)`.
- Auto-map dispatcher had two related bugs that combined to skip
  legitimately-renderable maps and over-zoom into ADM1 when the
  user asked at country scale.  Observed on a Mexican-avocado
  trace: the LLM's structured map extraction produced
  `focal_country="MEX", adm1_region="MEX016"` (Michoacán) cleanly,
  but the map was skipped entirely and the user-facing notice
  claimed the focal geography was "below country/ADM1 scale".
  Two fixes, both in `_generate_map` / `_has_unsupported_automap_scope`
  in `core.py`:
  - **Trust the LLM's structured extraction.**
    `_has_unsupported_automap_scope` previously ran a regex over
    the prose looking for keywords like `watershed`, `municipality`,
    `reserve`, etc. and short-circuited the map render whenever
    any matched — even when the LLM had already produced a
    validated `focal_country` / `adm1_region` ISO code.  Now the
    structured extraction is authoritative: if `parsed.map_data`
    has either field set, the regex never fires.  The prose-keyword
    fallback only runs when the LLM produced no structured focal at
    all (preserving the helpful "sub-ADM1 geography" notice for the
    genuine watershed-only case).
  - **Respect the user's framing for map scale.**  When the LLM
    stamps an `adm1_region` in the extraction but the user's
    original query named no ADM1 region (e.g. "avocado trade in
    Mexico" mentions no Mexican state), the dispatcher now drops
    the ADM1 and renders country-level.  Explicit subnational
    queries ("avocado production in Michoacán, Mexico") still
    render ADM1 as before — the override only fires when the user's
    framing is country-level.  Country names that double as state
    names ("Mexico" the country = Estado de México the state) are
    correctly treated as country mentions via a
    `resolve_country_code` precedence check in the new
    `_user_query_mentions_adm1` helper.  ISO-style codes ("MEX016")
    in the query are also recognised as ADM1 mentions.  No public
    API change.

## [0.1.3] — Turn-scoped citation markers `[Tk:N]`

Multi-turn citation disambiguation. **Recommended upgrade for anyone
using `refine()` or follow-up `analyze()` calls in RAG-only mode.**

**Breaking change**: the pre-v0.1.3 public citation API is gone.
Callers that imported ``sanitize_citations``, ``extract_cited_ids``,
or ``CITATION_PATTERN`` from ``metacouplingllm`` /
``metacouplingllm.knowledge.citations`` must migrate to the new
turn-scoped equivalents:

| Removed (pre-v0.1.3)          | New (v0.1.3+)                   |
|-------------------------------|---------------------------------|
| ``sanitize_citations(text, n_valid)`` | ``sanitize_turn_citations(text, turn_passage_counts, turn_web_counts, current_turn)`` |
| ``extract_cited_ids(text)``   | ``extract_turn_cited_ids(text)`` (returns ``set[tuple[int, str, int]]``) |
| ``CITATION_PATTERN``          | ``TURN_CITATION_PATTERN``       |

End-user code that only uses ``MetacouplingAssistant.analyze()`` /
``refine()`` and consumes ``AnalysisResult`` / ``RAGResult`` is
unaffected — sanitization is internal.

- **Grammar change.** Literature citations are now emitted as
  ``[Tk:N]`` and web citations as ``[Tk:Wn]``, where `k` is the
  1-indexed turn number and `N`/`n` is the passage/web-source ID
  within that turn. Previously, each turn re-numbered passages from
  `[1]` — so turn 1's `[1]` and turn 2's `[1]` could refer to
  different papers, and conversation history preserved both messages
  verbatim. With the new grammar, once a citation is emitted it is
  unambiguous forever: turn 1's `[T1:3]` always means turn 1's 3rd
  passage, even when read inside a later turn.
- **Back-references allowed.** The LLM may now cite prior-turn
  evidence by copying the exact token verbatim — e.g., a turn-2
  answer can say *"extending [T1:3] with the new data shows..."* —
  because past-turn tokens stay valid.
- **New sanitizer.** ``metacouplingllm.knowledge.citations.sanitize_turn_citations(text, turn_passage_counts, turn_web_counts, current_turn)``
  validates each ``[Tk:N]`` and ``[Tk:Wn]`` against per-turn passage /
  web counts recorded on the assistant. Forward references
  (``[Tk:N]`` with `k > current_turn`), out-of-range IDs, and bare
  ``[N]`` / ``[W1]`` slips from the LLM are stripped — the latter
  silently as a defensive measure, with no public legacy API kept.
- **Prompt rewrite.** ``CITATION_RULES_LAYER`` and
  ``_RAG_ONLY_SYSTEM_PROMPT`` were rewritten to teach the new grammar
  explicitly. Each ``<retrieved_literature>`` / ``<web_search_results>``
  block now carries a ``turn="k"`` attribute, and each ``<passage>``
  inherits the same. The LLM uses these attributes to assemble the
  citation token.
- **Renderer change.** ``REFERENCES`` and ``WEB SOURCES`` blocks in
  ``AnalysisResult.formatted`` and ``RAGResult.formatted`` now show
  each entry with its turn-scoped label (``[T1:1] Title…``,
  ``[T2:W3] Title…``). The ``_renumber_citations_sequentially``
  helper that re-mapped sparse ``[N]`` markers in RAG-only mode is
  removed — the LLM emits stable tokens directly, so no remapping is
  needed. ``RAGResult.references`` lists only **current-turn**
  citations; prior-turn back-references are deliberately excluded
  (they belong to the prior turn's reference block).
- **Plumbed through.** ``PromptBuilder.build_initial_message`` and
  ``build_refinement_message`` gained a ``turn`` kwarg.
  ``format_web_context``, ``format_evidence``, ``annotate_citations``,
  and ``annotate_web_citations`` likewise. ``MetacouplingAssistant``
  records ``_turn_passage_counts``, ``_turn_web_counts`` (and the
  RAG-mode equivalents) on every retrieval so the sanitizer can
  validate any back-reference, no matter how old.
- **Migration note.** Older saved conversations contain bare ``[N]``
  / ``[W1]``; the new run will silently strip those on the next
  refine and emit ``[Tk:N]`` going forward. History is not
  retroactively rewritten — the fix is preventive (correct grammar
  from the start), not corrective.

## [0.1.2] — Bundled-data loader fix (Critical)

Bug fix release. **All v0.1.0 / v0.1.1 users should upgrade.**

- **Fixed**: `resources.files("metacoupling")` was hardcoded in three
  places (`knowledge/literature.py`, `knowledge/pericoupling.py`,
  `knowledge/adm1_pericoupling.py`) — leftover from the
  `metacoupling -> metacouplingllm` rename. The `try/except
  ModuleNotFoundError` wrapper made it fail silently, returning an
  empty database. Symptoms users saw:
  - `[RAG] WARNING: No literature database loaded.`
  - `[MetacouplingAssistant] WARNING: RAG engine loaded but has 0 chunks.`
  - `Pre-retrieval RAG: 0 passages.`
  - `total_papers: 0` from `get_database_info()`
  - All paper recommendations and country/ADM1 pericoupling lookups
    silently returning empty results.
- **Fix**: All three `resources.files(...)` calls now correctly
  reference `"metacouplingllm"`. After upgrading, the bundled
  literature database loads with **265 papers** (249 with
  keywords) and the RAG engine finds **10,032 chunks** as designed.
- **No API or behavior changes** beyond the loader fix — same module
  layout, same public interface.

## [0.1.1] — Author metadata fix

Metadata-only release. No code or behavior changes.

- **Fixed**: PyPI's project page now shows all six authors (Xiang Yu,
  Yingjie Li, Zihan Zheng, Nan Jia, Xin Lan, Jianguo Liu) on the
  `Author:` line. In `0.1.0`, Xiang Yu's name was being rendered only
  on the `Maintainer:` line because PEP 621 splits authors with emails
  into a separate `Author-email:` METADATA field, and PyPI's UI then
  showed only the email-less authors under "Author:". Removing the
  email from Xiang Yu's `authors` entry (it stays on the
  `maintainers` entry) consolidates all six names under a single
  `Author:` field. The maintainer email is unchanged.
- **No source-code changes** — this is purely a packaging metadata fix.

## [0.1.0] — Initial Release

The initial public release. This version packages a complete
metacoupling-framework research assistant built around:

- A multi-layer prompt builder that injects framework definitions,
  curated case examples, country/ADM1 pericoupling-database hints,
  and optional web-search context into the system prompt.
- A pre-retrieval RAG pipeline over a bundled corpus of 262 peer-
  reviewed telecoupling papers (~6,400 text chunks), using BGE-small
  semantic embeddings with a TF-IDF fallback.
- Structured-output parsing, pericoupling validation against a curated
  country-pair and ADM1-adjacency database, optional literature
  recommendation, and optional country-level / subnational map
  generation.

### RAG pipeline

- **Default `rag_mode="pre_retrieval"`.** Corpus passages are retrieved
  from the research description **before** the LLM call, embedded in
  the user message as a labeled `<retrieved_literature>` XML block,
  and the LLM cites them inline as `[1]..[N]`. A legacy
  `rag_mode="post_hoc"` path remains available for users who prefer
  keyword-overlap citation annotation after generation.
- **`CITATION_RULES_LAYER`** — a system-prompt layer that defines the
  `[1]..[N]` citation grammar, forbids inventing citations, requires
  passages to directly support the claim, and notes that citation
  numbering is **turn-local** (the same number may refer to different
  papers across turns).
- **`sanitize_citations(text, n_valid)`** in
  `metacoupling.knowledge.citations`: strips out-of-range bracket
  tokens from the LLM response, logs a `WARNING` naming the stripped
  IDs, and runs an idempotent whitespace/punctuation cleanup so the
  text reads naturally after stripping.
- **Labeled merged query on `refine()`.** Follow-up calls re-run
  retrieval using a structured query that combines the original
  research question and the refinement text, anchoring retrieval to
  the original topic while letting the refinement steer the new
  passage pool.
- **Default `rag_top_k=8`** (enough passages for literature-grounded
  generation without a cross-encoder reranker). Reranking is deferred
  to a future phase.
- **Multi-chunk retrieval per paper (`rag_max_chunks_per_paper`,
  default 3).** A single highly-relevant paper often has its key
  evidence scattered across several sections (Introduction / Methods
  / Results / Discussion), AND long sections are often split by the
  chunker into multiple distinct sub-topic chunks (e.g., § 4.1
  Inbound vs § 4.2 Outbound both tagged as "4. Results"). The legacy
  retriever enforced "one chunk per paper", collapsing all of that
  into a single chunk — typically a high-level Discussion summary
  for abstract queries. The new retriever uses a **two-pass**
  selection: pass 1 prefers chunks from distinct sections per paper
  (buys section diversity first); pass 2 fills any remaining
  per-paper budget from the deferred same-section chunks (so long
  sections split into several sub-topic chunks can still contribute
  multiple chunks when budget allows). The total is capped by
  ``rag_max_chunks_per_paper`` (default ``3``). Set to ``1`` to
  restore legacy behavior, or raise to ``5`` (or higher) for a
  systematic framework paper where bilateral / quantitative case-
  study data is spread across many sub-topics of one section. The
  cap applies to both the embeddings backend and the TF-IDF fallback
  and is plumbed through ``RAGEngine.retrieve()`` and
  ``retrieve_for_analysis()``.
- **Optional structured-extraction supplement
  (`rag_structured_extraction=True`).** In pre-retrieval mode, a
  second schema-validated LLM pass scans the already-retrieved
  passages for systems and flows the free-form draft may have
  under-specified — addressing the common failure mode where content
  about receiving systems or flows is scattered across multiple
  sections of the same paper. The supplement covers all three
  system roles (sending, receiving, spillover) plus a supplementary-
  flows list, each item carrying evidence-passage IDs. Results are
  rendered as a visibly labelled ``SUPPLEMENTARY STRUCTURED
  EXTRACTION`` block between the main analysis and the
  ``SUPPORTING EVIDENCE FROM LITERATURE`` section, and exposed
  programmatically as ``AnalysisResult.structured_supplement``. The
  main analysis body is never silently rewritten — the reader can
  always tell LLM-authored content from RAG-extracted content.
  Disabled by default because it adds one LLM call per turn.
  - **Supplement → map bridge.** When ``auto_map=True`` and the
    supplement surfaced specific countries, they are merged into
    ``parsed.map_data['receiving_countries']`` and
    ``['flows']`` via a new
    ``MetacouplingAssistant._merge_supplement_into_map_data()``
    helper. The merge is additive — it never removes or overwrites
    entries produced by the primary
    ``_extract_map_data_from_analysis`` pass; it only fills in
    bilateral partners the analysis text abstracted as "foreign
    countries". Country names are resolved to ISO alpha-3 via the
    existing ``resolve_country_code`` helper; unresolvable names are
    logged and skipped. Supplementary-flow ``direction`` strings are
    parsed into source/target ISO codes using the shared
    ``_FLOW_ARROW_RE`` regex.
  - **Full-length passages to both LLMs.** Removed the 600-char
    passage truncation in the structured-extraction helper and
    raised ``_PASSAGE_MAX_CHARS`` in the main prompt builder from
    800 → 5000. The prior caps were artifacts of early
    cost-sensitivity but were losing ~50–65% of every chunk's
    content — specifically the bilateral country data that lives
    past the section-opening summary in long Results sections (e.g.,
    "Korea (2.65 MtCO2), Japan (1.92 MtCO2)" at char 689 in Duan
    2022 § 4. Results). The chunker already caps chunk size
    naturally (p99 = 1927 chars, max observed = 4687), so 5000 is
    effectively a no-op cap for legitimate chunks while still
    bounding pathological outliers.
  - **Per-country extraction rule.** The structured-extraction
    prompt now explicitly instructs the LLM that when a passage
    names specific countries with numeric values (e.g., "Korea
    (2.65 MtCO2), Japan (1.92 MtCO2), Russian Federation (1.46
    MtCO2)"), each country must be emitted as its own entry in the
    appropriate ``additional_{sending,receiving,spillover}_mentions``
    list — not collapsed into a grouped abstraction like "Pacific
    Rim countries". Numeric values are retained parenthetically in
    the ``name`` field. Per-list caps raised from 6 → 12 so bilateral
    breakdowns have room.

### Corpus quality and chunk integrity

- **Cross-platform chunk ordering.** `RAGEngine.load()` sorts
  `glob("*.md")` by the pure-string filename (`key=lambda p: p.name`)
  so the chunk index is identical on Windows, Linux, and macOS.
  Without this, Python's case-insensitive `Path.__lt__` on Windows
  would place filenames beginning with lowercase letters (`da Silva`,
  `de Lucio`, ...) in different positions from Linux/macOS, silently
  corrupting the chunks-to-embeddings mapping.
- **Manifest integrity check.** `chunk_embeddings.manifest.json`
  ships alongside `chunk_embeddings.npy` with a SHA-256 fingerprint
  of the chunk order and build metadata. `EmbeddingRetriever.__init__`
  recomputes the fingerprint at load time and raises `RuntimeError`
  on mismatch, so chunk-vs-embedding drift can never silently corrupt
  retrieval.
- **Three-layer reference filter** in `_chunk_markdown()`:
  - `_truncate_at_references(text)` drops everything from the first
    "References"/"Bibliography"/"Literature Cited" heading onward
    before chunking starts.
  - `_is_reference_heading(heading)` rejects bibliography-style
    headings, including numeric-prefixed artifacts like `"2016. ..."`
    and `"708. ..."`.
  - `_looks_like_reference_chunk(text)` is a multi-signal heuristic
    that scores each chunk on year-count, author-pattern count,
    reference-term count (doi, proceedings, editors, pages, …), URL
    presence, and page-reference presence. Chunks that look like
    bibliography entries are dropped even when they survive the
    heading-level filters.
  - Net effect: ~33% of chunks that old versions of the chunker
    produced were bibliography junk. The current corpus indexes
    roughly 6,400 chunks instead of ~9,600, raising the precision of
    the `SUPPORTING EVIDENCE FROM LITERATURE` block.
- **Topic-relevance rule for web extraction.** The structured
  `extract_web_map_signals()` LLM call is explicitly told to ignore
  trade data about unrelated products, sectors, or commodities even
  when the focal country is mentioned — so a study about feed barley
  no longer pulls Hong Kong sheep-offal exports onto the map.
- **Correct country-arrow origins.** `_get_country_centroid()` merges
  all rows matching a given ISO code via `union_all()` and picks the
  representative point of the **largest polygon**, so the UK centroid
  lands on Great Britain rather than on the British Sovereign Base
  Areas in Cyprus, France's centroid on the European mainland rather
  than on an overseas territory, etc.

### Tests

- 629+ unit and integration tests spanning prompt assembly, RAG
  retrieval, citation sanitization, refine() behavior, manifest
  integrity, chunk-order determinism, reference-filter heuristics,
  pericoupling databases, map rendering, and LLM-client adapters.

### Known limitations

- **Stale citation tokens across turns.** *Resolved in v0.1.3* — see
  the [0.1.3] entry. Turn 1's `[1]` and turn 2's `[1]` no longer
  collide because the new grammar emits `[T1:N]` and `[T2:N]`
  respectively. Pre-v0.1.3 saved conversations still contain bare
  `[N]` tokens; running `refine()` under v0.1.3 will silently strip
  them and emit the new turn-scoped form going forward.
- **No second-stage reranking.** A cross-encoder reranker over the
  top-k retrieved passages is planned but not shipped; pre-retrieval
  currently uses the raw BGE-small ranking.
