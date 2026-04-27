# Changelog

All notable changes to the `metacoupling` package are documented in this
file. The format is loosely based on
[Keep a Changelog](https://keepachangelog.com/), and this project follows
[Semantic Versioning](https://semver.org/).

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

- **Stale citation tokens across turns.** Because conversation history
  persists across `refine()` calls and each refine re-retrieves a
  fresh passage set, the same number `[1]` may refer to different
  papers in turn 1 and turn 2. The system-prompt rule says "cite only
  from the most recent block" and the sanitizer strips out-of-range
  tokens, but neither can detect a token that is in-range yet
  semantically wrong. Treat each turn's `SUPPORTING EVIDENCE` block
  as the authoritative mapping for that turn's citations. A future
  release is expected to introduce turn-scoped markers (e.g.,
  `[T2:1]`).
- **No second-stage reranking.** A cross-encoder reranker over the
  top-k retrieved passages is planned but not shipped; pre-retrieval
  currently uses the raw BGE-small ranking.
