# Changelog

All notable changes to the `MetacouplingLLM` package are documented in this
file. The format is loosely based on
[Keep a Changelog](https://keepachangelog.com/), and this project follows
[Semantic Versioning](https://semver.org/).

## [Unreleased]

### Added

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
