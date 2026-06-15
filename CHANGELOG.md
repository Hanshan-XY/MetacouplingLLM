# Changelog

All notable changes to the `MetacouplingLLM` package are documented in this
file. The format is loosely based on
[Keep a Changelog](https://keepachangelog.com/), and this project follows
[Semantic Versioning](https://semver.org/).

## [Unreleased]

### Added

- **ADM1 English-exonym alias table (`adm1_aliases.csv`) + Strategy 0 resolver.**  A new pre-validated CSV ships 1,145 English exonyms and alternative spellings for 863 WB ADM1 regions across 136 countries (e.g. `"bavaria"→DEU002`, `"tuscany"→ITA016`, `"saxony"→DEU013`, `"andalusia"→ESP002`, `"brittany"→FRA007`).  `resolve_adm1_code` now tries this table first (Strategy 0) before the existing name/substring strategies, so natural English names resolve without needing the exact WB canonical form.  The CSV is validated by deterministic rules (N0/G/S1/W1/DR/R1/O1/X1/U1/U2/U3/D0/C1); the country filter is always respected — a hint that doesn't match the alias's country falls through gracefully.  New public `get_adm1_aliases(code)` getter returns the alias keys for any ADM1 region.  The build script (`scripts/build_adm1_aliases.py`) and its companion validator (`scripts/validate_adm1_aliases.py`) are included for full dataset provenance.  (#60, #61)

- **`build_adjacency` — auto-fill the `adjacent` flag from the bundled pericoupling DB.**  New helper in `metacouplingllm.indicators` that bridges the curated ADM0 / ADM1 pericoupling databases (§8) with the indicator pipeline: pass a table of `origin_id` / `destination_id` pairs and it fills `adjacent` with `1` (pericoupled / bordering) or `0` (telecoupled / not bordering).  Works at country level (`level="adm0"`; IDs may be country names or ISO alpha-3) and subnational level (`level="adm1"`; IDs may be WB ADM1 codes like `MEX008`, region names like `Chihuahua`, or `"Region, Country"` like `Chihuahua, Mexico`).  Self-pairs are dropped — *entity-level*, so the same place written two ways (a code and a name, e.g. `MEX008` and `Chihuahua, Mexico`) is recognised as one and dropped, not mislabelled — and duplicate pairs collapsed; any ID the database cannot resolve is left as `<NA>` and reported in a single `UserWarning` (never silently assumed telecoupled).  `de_facto_borders` / `coupling_standard` pass through to the lookups, and the result plugs straight into `classify_coupling`.  Documented in MANUAL §16 ("Auto-filling `adjacent` from the bundled database") with a verifying test.  As part of this, `classify_coupling` now treats a `<NA>` / `NaN` adjacency flag as not-adjacent (previously `bool(NaN)` was truthy in the symmetric-set build).

### Fixed

- **ADM1 alias/resolver follow-up from an Opus adversarial double-check of #59/#60 (3 code defects + 33 bad aliases removed).**  A full multi-agent re-review of both ADM1 PRs surfaced — and this change fixes — defects that re-introduced the very "confident wrong answer" class #59 set out to eliminate: (1) **Folded shadow** — the alias validator's uniqueness rules only consulted the *accented* name index, so `resolve_adm1_code("ash sharqiyah")` resolved to **Oman (OMN002)** instead of **Saudi Arabia (SAU008)** with no hint; a new fold-aware no-hint rule (`X1`) drops any alias whose key resolves (no hint) to a different region, and `ash sharqiyah` now correctly returns `SAU008`.  (2) **Direction-B substring hole** — bare tokens `"york"` / `"hampshire"` / `"brunswick"` resolved to New York / New Hampshire / New Brunswick because the "New X" guard only fired in Direction A; `_substring_match` now rejects a query preceded by a distinguishing qualifier in *either* direction (hyphen-aware, so the French `nouveau-brunswick` is caught too).  (3) **Loader fragility** — a malformed `adm1_aliases.csv` raised inside `_ensure_loaded` and broke *all* ADM1 lookups; `_load_adm1_aliases` is now `None`-safe and degrades to "no Strategy 0".  On the data side, a curated review denylist (`DR`) removes 51 out-of-scope aliases (wrong-region pointers like `loire valley`→Pays-de-la-Loire and `jubaland`→a single region; capital-city-for-region namesakes like `mosul`/`new delhi`; historical names like `smyrna`/`mysore`; acronyms like `gbao`/`snnpr`) while deliberately keeping four still-ubiquitous former names (`saigon`, `rangoon`, `orissa`, `pondicherry`).  Net: the shipped table drops from 1,178 to 1,145 validated aliases.  Provenance counts corrected (was "67 countries / 886 regions"; actually 136 countries / 863 regions).  (#61)

- **ADM1 region-name resolver no longer returns a confident wrong answer for "name + extra word" queries.**  `resolve_adm1_code` (and therefore `build_adjacency(level="adm1")`, the subnational map validators, and any `"Region, Country"` lookup) used a loose *bidirectional* substring match: a query that merely *contained* a canonical region name as a whole word resolved to that region regardless of the surrounding words — so `resolve_adm1_code("Mexico City", country="Mexico")` returned `MEX015` (the *State* of México) instead of CDMX, and `"New Mexico"` (a US state) likewise mapped to a Mexican state.  The match is now **direction-aware** (`_substring_match`): a short query inside a longer official name (e.g. `Michoacan` → `Michoacán de Ocampo`) is still accepted, but a query padded with a meaningful extra word is rejected (`None`); only administrative / connector words (`State`, `of`, …) are tolerated.  An added **ambiguity guard** also makes a query that substring-matches more than one distinct region return `None` instead of the arbitrary first match (e.g. `Carolina` → North vs South Carolina).  (#59)

### Docs

- **Documented the ADM1 English-exonym alias table in MANUAL + INTRODUCTION.**  The user-facing docs now describe the alias capability added in #60/#61: the MANUAL ADM1 API table notes that `resolve_adm1_code` resolves English exonyms / alternative spellings (Bavaria→DEU002, Tuscany→ITA016) and adds a `get_adm1_aliases(code)` row; the subnational-validator troubleshooting note mentions exonym recognition; and both the MANUAL ADM1 section and INTRODUCTION §4.5 + the bundled-data inventory now list `adm1_aliases.csv` (1,145 aliases · 863 regions · 136 countries).  (#62)

- **MANUAL §16 reworked around file-based input.**  Formulas now lead the section; a new "Preparing your data (CSV or Excel)" subsection documents the flows + adjacency table schema, the column-name kwargs, and the openpyxl-for-xlsx caveat; and the worked example now reads shipped sample files (`examples/brazil_soybean_flows.csv`, `examples/brazil_soybean_adjacency.csv`) via `pd.read_csv` instead of an in-line DataFrame literal.  A new `test_shipped_indicator_csvs_match_manual` runs those sample CSVs through the pipeline and asserts the documented indicator values (IFS/PFS/TFS 0.1/0.2/0.7, MFE 0.73, IFCI 1.00, PFCI 0.25, TFCI 0.33).

- **MANUAL accuracy pass (55 verified fixes + 2 §17 gaps).**  A full
  code-grounded scan of MANUAL.md (every claim checked against source /
  live runs, each finding adversarially re-verified) surfaced 9 high-,
  26 medium- and 20 low-severity issues; all are fixed.  Highlights:
  repaired broken examples (§7 structured-data access now uses the real
  `iter_*` API; §7 `format_component` valid names; §17 end-to-end
  workflow — `row_count` key, DataFrame input to `interpret_results`,
  adapter instead of nonexistent `advisor.llm_client`; §14
  `web_map_signals` is an output field, not a kwarg); replaced the §7
  sample report with the real coupling-first numbered layout; updated
  §8 to the moderate-default counts (322 exposed of 325) and §12
  pericoupling signatures to show `de_facto_borders` +
  `coupling_standard`; removed the fictional `DuckDuckGoBackend` class
  (3×); corrected `blocked_domains` / `search_context_size` defaults
  and the auto-enabled `web_structured_extraction`; documented the
  MFCI edge-case conventions, the long-format `compute_mfci` output,
  six flow categories (organisms was omitted), the
  `attrs.get("llm_classify_trace")` pattern, the §10 disputed-overlay
  legend entry, and the abstract's standalone (citation-free) design.
  Every corrected executable claim was re-run live (15/15 pass).

- **Executable-docs CI guard (`tests/test_docs_examples.py`).**  Turns
  the accuracy pass into a build break: every fenced `python` block in
  MANUAL.md must compile, and an allowlist of deterministic sections
  (§7 structured access + formatting, §8 standalone pericoupling, §16
  worked example incl. `group_cols`) is *executed* with assertions that
  the documented outputs still match the package.  Allowlisted blocks
  are checked to stay LLM/network-free.

### Changed

- **Indicator naming + provenance finalised (pre-publication).**
  (1) The three indicator families are now the **Metacoupled** Flow
  Shares / Flow Evenness (MFE) / Flow Concentration Index (MFCI) —
  "Metacoupled", not "Metacoupling", modifying the flows being
  measured.  (2) Partner-count columns renamed `m_I`/`m_P`/`m_T` →
  **`n_I`/`n_P`/`n_T`** (long format `m_partners` → `n_partners`) to
  match the MFCI formula's $n_{ic}$ and the n in `effective_n_partners`.
  (3) `ENP_I`/`ENP_P`/`ENP_T` are spelled out as the **Equivalent
  Number of Intra-/Peri-/Telecoupled Partners**.  (4) Provenance
  corrected to the earliest sources: normalised HHI per **Hannah & Kay
  (1977)** (replacing Cracau & Lima 2016) and ENP per **Laakso &
  Taagepera (1979)**; Shannon (1948) and Hirschman (1945) unchanged.
  Applied across the indicators code, `write_methods` prompt, MANUAL,
  INTRODUCTION, README, and references.

### Added

- **`__version__` now derives from the installed distribution metadata**
  (falling back to `pyproject.toml` for source-tree imports), so it can
  no longer drift from the packaged version the way the hardcoded
  `"0.1.0"` had (actual version: 0.1.3).

### Fixed

- **Italy↔Vatican corrected from water-only to a land border.**  The Lazio↔Vatican pair (`ITA007`↔`VAT001`) was geometrically mis-flagged as a Tiber water-only crossing and so was wrongly dropped under the default `coupling_standard="moderate"`.  The Vatican is a ~2.7 km land enclave within Rome (the Tiber is ~0.6 km away, not the border), so it is now a land border, pericoupled under every standard.  Shipped water-only counts: ADM1 315→**314** (108/206), ADM0 18→**17** (15/2).  Default-view effective edges: ADM1 8,234→**8,235**, ADM0 322→**323**; stringent ADM1 8,129→**8,130**, ADM0 307→**308**.  Regression test added.

- **3-letter acronyms no longer count as country mentions.**
  `get_country_name()` returns the input code itself for unknown codes,
  so the truthiness checks in the ADM1 mention-extraction and relevance
  guards treated any all-caps token (`GDP`, `USD`, `PNG`…) as a country
  mention — which could activate the relevance guard and suppress a
  directly-named region.  All three sites now use an explicit
  `ISO_ALPHA3_NAMES` membership check; 2 regression tests added.

- **`coupling_standard` for water-separated adjacency (`stringent` /
  `moderate` / `lenient`, default `moderate`).**  The pericoupling loaders
  (`pericoupling.py`, `adm1_pericoupling.py`) accept `coupling_standard`,
  orthogonal to `de_facto_borders`.  For the 314 ADM1 (and 17 rolled-up ADM0)
  pairs that share **only** a river/lake border, `moderate` (the new default)
  keeps a pair only if a **fixed crossing open to traffic** links the two units;
  `stringent` drops every water-only pair; `lenient` keeps all (the prior
  behaviour).  So Kinshasa↔Brazzaville (Congo, ferry only) is no longer
  pericoupled by default, while the Rio Grande and the Mekong Friendship-bridge
  crossings remain.  Bridge presence was classified from OpenStreetMap and then
  **independently verified** (web search + a geometric province check + manual
  review); see `data/water_separated_pairs.csv` and
  `docs/BRIDGE_CLASSIFICATION_METHODOLOGY.md`.  ADM1 pericoupled edges under the
  default fall 8,381 → **8,235**; ADM0 country pairs 325 → **323**.  A
  structurally complete bridge on a politically *closed* border still counts
  (pericoupling is structural; under-construction links do not).

- **`de_facto_borders` toggle for disputed-territory adjacency.**  The
  pericoupling loaders (`pericoupling.py`, `adm1_pericoupling.py`) now accept
  `de_facto_borders` (default `True`).  WB's standard boundary layers exclude
  the NDLSA disputed-areas tracts, opening multi-km gaps between neighbours that
  meet only across a de-facto line of control.  The shipped data is the
  **de-facto** view (disputed land folded into its de-facto administrator),
  which at ADM0 treats China–Pakistan, Israel–Syria, and Morocco–Mauritania as
  adjacent (3 country pairs).  At ADM1 the overlay is derived **independently** —
  country adjacency does not imply province adjacency — from a separate authored,
  geometry-validated tract→province map, yielding **13** subnational pairs (e.g.
  Arunachal Pradesh↔Tibet, Northern↔Quneitra (Golan), Guelmim-Oued Noun /
  Laâyoune↔Mauritania (Western Sahara), Haa↔Tibet (Doklam)).  China–Pakistan has
  **no** ADM1 pair: Gilgit-Baltistan and Ladakh/J&K are disputed territories
  excluded from WB's ADM1 layer (not provinces), so that relationship is carried
  at ADM0 only.  Passing `de_facto_borders=False` returns the strict
  standard-layer view, which omits the overlay (3 ADM0 + 13 ADM1).  Both levels
  are **derived from geometry at build time** by `derive_disputed_overlay()`,
  which validates every authored unit against the tract polygons so a mislabel
  fails loudly instead of silently dropping a pair (this surfaced the
  previously-missed Western Sahara → Morocco fold and the subnational pairs hidden
  behind already-adjacent countries, e.g. Arunachal Pradesh↔Tibet).  The applied
  pairs ship in `data/disputed_overlay_pairs.csv` and the full per-tract candidate
  audit in `docs/ndlsa_tract_audit.csv`.  Attributions are authored (the source
  has no sovereignty field) and encode physical coupling, **not** a legal claim —
  see `data/PROVENANCE.md` and `docs/METHODS_adjacency.md`.  ADM1 edge count
  8,368 → 8,381 (13 overlay edges); ADM0 adds Morocco–Mauritania, 324 → 325.

### Fixed

- **Malta snapping-tolerance false positive removed.**  The ~55 m snapping
  tolerance bridged a ~31 m gap between Balzan (`MLT002`) and Iklin (`MLT019`),
  which do not share a frontier, fabricating a spurious ADM1 edge.  It is now
  dropped via `_ADM1_FALSE_POSITIVE_DENYLIST` in the build script (ADM1 edge
  count 8,369 → 8,368, before the de-facto overlay).  Added
  `docs/METHODS_adjacency.md` documenting the rook-contiguity rule, the
  snapping-tolerance and river-buffer sensitivity analyses, and the
  tolerance-sensitive-band audit.

- **Cross-country folded-name ambiguity in `resolve_adm1_code`
  (follow-up to #45/#50).**  Two ADM1 provinces fold to the same
  name across different countries — the DRC stores its province
  accented as `Équateur` (`COD013`) while the Central African
  Republic stores its unaccented as `Equateur` (`CAF002`).  A bare
  `Equateur` query previously resolved arbitrarily to `CAF002`
  (an exact Strategy-1 match), silently returning the wrong
  country.  A new `_is_cross_country_folded_collision` guard makes
  an unaccented query whose accent-folded spelling is shared across
  countries return `None` unless a `country` hint is given; an exact
  accented query (`Équateur`) and any hinted query are unaffected.
  Data-driven, so future cross-country collisions are covered
  automatically.

### Changed

- **Pericoupling databases regenerated from World Bank Official
  Boundaries (2026-05-14) + ISO-3 modernization (PR #50).**
  - Both bundled adjacency datasets rebuilt by a new committed,
    reproducible build script (`scripts/build_pericoupling_db.py`)
    using a strict shared-land-border method: STRtree adjacency,
    inland-water filter (open-water meetings dropped, shore/river-
    following borders kept), **geodesic kilometre** border lengths
    (replacing latitude-distorted degrees), and an explicit disputed-
    border allowlist (`CHN/PAK`, `ISR/SYR`). ADM1: 8,290→**8,369**
    edges, 3,366→**3,373** regions, 195→**196** countries. ADM0:
    308→**324** land-border pairs on 264 units.
  - `countries.py` **migrated from legacy to current ISO 3166-1
    alpha-3** codes (`ZAR`→`COD`, `ROM`→`ROU`, `YUG`→`SRB`,
    `TMP`→`TLS`) and extended with the 44 codes the new data uses, so
    every ISO code in the bundled CSVs resolves. The legacy→package
    remap in `visualization/worldmap.py` (`_ISO_CODE_FIXES`) is
    removed — map geometry and classification now share one modern
    vocabulary. A new guard test asserts every CSV ISO code resolves.
  - Provenance, method, and known limitations (disputed-area
    allowlist, river-separated province pairs, Flevoland polder)
    documented in `data/PROVENANCE.md`.

- **RAG embeddings relevance floor raised 0.3 → 0.60 (PR #49).**  The
  backend-aware `rag_min_score` default for the embeddings backend
  (BGE-base cosine) is now `0.60` instead of `0.3`
  (`knowledge/rag.py`); the TF-IDF fallback floor (`0.01`) is
  unchanged.  This favors precision over recall — only strong matches
  are retrieved.  The two backends score on different scales, so a
  value tuned for embeddings (0.60) would filter out ~everything on
  TF-IDF; the floor stays backend-specific.

### Docs

- **INTRODUCTION / MANUAL / README accuracy pass (PR #49).**
  - §3 architecture diagram: web-search box relabelled from
    `(DuckDuckGo)` to `(native + DuckDuckGo)` (DDG is the *fallback*,
    not the only backend); Map Generator `(world/ADM1)` → `(ADM0/ADM1)`;
    `Pericoupling Validation` → `Coupling Validation` (matches the
    `COUPLING DATABASE VALIDATION` output block and the parallel
    MANUAL diagram).
  - RAG corpus described accurately: "420 papers" with **full text
    for the 296 open-access papers and structured summaries for the
    124 non-open-access papers** (was the inaccurate "420 full-text
    papers"; the OA/non-OA split is recorded in
    `paper_citation_counts.csv`).
  - RAG retrieval description: "at most one chunk per paper" corrected
    to the real per-paper cap (default 3, configurable via
    `rag_max_chunks_per_paper`); added the backend-aware `rag_min_score`
    note (0.60 embeddings / 0.01 TF-IDF); example snippets bumped
    `rag_min_score=0.15` → `0.60`.
  - "Structured web map hints": clarified that "validated" means the
    extracted countries/flows are grounded (resolve to a real
    country/union + cite a real retrieved snippet, confidence ≥ 0.7),
    not that the spillover *role* is validated.
  - Literature DB described as "265 empirical journal articles
    (2013–2025)" (all `@article`; year range verified).
  - `(result, LLMTrace) … per spec §15.4` cross-reference (no such
    section/spec exists) repointed to MANUAL §17 "LLM-Assisted
    Indicator Helpers"; dangling `spec §15.4` refs inside MANUAL
    removed.
  - `Option A integration with classify_coupling()` reworded to
    "Automatic LLM resolution of ambiguous edges" / "LLM-assisted edge
    resolution" (the "Option A" label was an internal PR #36 design
    tag, meaningless to readers).

### Added / Refactored

- **Coupling-validation consistency + naming cleanup (PR #48).**
  Three related cleanups on the post-LLM coupling-validation
  surface:

  **(a) ADM1-scope spillover filter.**  PR #44 taught the
  *country* validator to drop partners the LLM placed in the
  spillover role (geography can't validate the framework's
  direct-vs-indirect distinction).  The *ADM1* validator
  (`_validate_adm1_pericoupling`) still did a pure 2-way
  geographic split, so a subnational region framed as spillover
  got hard-labeled PERICOUPLED/TELECOUPLED.  A new
  `_extract_adm1_with_roles` helper (ADM1 sibling of
  `_extract_countries_with_roles`, resolving via
  `resolve_adm1_code`) now lets the ADM1 validator drop
  spillover-roled regions before bucketing — the same fix at
  subnational scale.

  **(b) Supranational unions in the country validator.**
  `resolve_country_code("EU")` returns None (a union isn't a
  country), so a study framed as "Brazil → EU" silently dropped
  the Brazil↔EU relationship from the `COUPLING DATABASE
  VALIDATION` block even though the map renderer dissolves
  unions (PR #22/#23).  A new `_extract_unions_with_roles`
  helper detects EU / ASEAN / USMCA / NAFTA in both system
  entries and flow text (via `expand_supranational` +
  `supranational_display_name`), and the country validator now
  emits a union line.  A per-member adjacency check surfaces any
  genuinely-pericoupled member on its own line (rare — e.g. a
  focal country bordering a metropolitan EU member) and
  collapses the usual all-distant case to one
  `Brazil (BRA) ↔ European Union: TELECOUPLED` line.  Union
  members named individually (e.g. "EU" and "Germany" both
  present) are de-duplicated — the individual country line wins.
  Spillover-roled unions are filtered, same as spillover
  countries.  The per-partner verdict logic is factored into a
  shared closure so the country loop and the union expansion
  classify identically (respecting PR #44 v3.2's national-vs-
  subnational mode).

  **(c) Rename `_build_pericoupling_hint` → `_build_coupling_hint`.**
  The pre-LLM hint builder (`prompts/builder.py`) emits BOTH
  pericoupled and telecoupled per-pair lines, so the
  `pericoupling` name understated its scope (deferred since
  PR #43).  Pure mechanical rename of the private method + its
  call site + docstring cross-ref + 4 test call sites.  The
  LLM-facing `## PERICOUPLING DATABASE LOOKUP (REFERENCE)`
  heading and in-body prompt literals are intentionally left
  unchanged (no Stage-2 prompt change, no retrace needed).

  Tests added: 8 (3 ADM1-filter, 5 union).  Total: 1184 → 1192.

### Fixed

- **Stage-3 map extraction: retry + alert on non-JSON responses
  (PR #47).**  The structured map-extraction LLM call
  (`_extract_map_data_from_analysis`) occasionally received a
  prose summary instead of the requested JSON object — observed
  in ~1 of 13 live GPT-5.5 traces (the Jalisco `pr45_resolver`
  run).  When that happened, `_extract_json_object` returned
  `None`, the method returned `None`, and the map silently did
  not render — the run "succeeded" with no error surfaced and a
  misleading `map_notice` blaming "no resolvable focal country"
  (the analysis text *did* name a focal; the map step just
  couldn't parse the model's output).

  Two changes:
  - **Retry.**  The extractor now retries the call once on a
    format failure (call exception / unparseable / non-dict).
    The retry samples a fresh completion (GPT-5 models run at
    effective temperature 1.0 even when 0.0 is requested), so a
    single retry almost always recovers.  A valid-JSON-but-no-
    focal response is *not* retried — that is a content outcome
    a retry won't change.
  - **Alert.**  The previous silent `print()` calls are now
    `logger.warning()` (surfaced through standard logging), and
    when every attempt fails the method records
    `_last_map_extraction_error` so `result.map_notice` reports
    an accurate `extraction_format_error` reason instead of the
    misleading "no focal country" message.

  Net effect: the common transient slip self-heals via the
  retry; the rare double failure is now loud and correctly
  attributed instead of silent.  This makes adapter-level
  strict-output for Stage-3 unnecessary for current models
  (and avoids its supranational-shorthand / information-loss
  trade-offs).

  Tests added: 2 (retry-recovers, retry-exhausted-alerts).
  Total: 1182 → 1184.

### Added

- **Doc-capability drift CI guard (PR #46).**  PR #42 was a manual
  audit that uncovered ~15 places where the package's marketing-
  style capability lists in INTRODUCTION.md, MANUAL.md, and
  README.md had drifted out of sync with actually-shipped features
  (Gemini/Grok adapters, scholar export, quantitative indicators,
  `evidence_coverage_note`, …).  In each case the feature was
  implemented and tested; it just wasn't advertised anywhere a
  reader would look.  This PR adds three pytest cases in
  `tests/test_docs_capabilities.py` that compare the three docs'
  capability surfaces against a curated `EXPECTED_FEATURES`
  registry whose entries are each anchored to a real code check
  (a symbol in `__all__`, an importable submodule, a method on a
  class, or a function name in the source text).

  How it works:
  - `EXPECTED_FEATURES` is a list of `MarketingFeature` dataclass
    entries.  Each has a `keyword` (lenient substring, case-
    insensitive, hyphen ≡ space, split-word tolerance — so
    `"web search"` matches the `"Web-Search Backends"` heading),
    a `code_check` lambda (when False, the feature is treated as
    no longer required in docs — automatic teardown), and a
    `docs` set restricting which docs must mention it.
  - Three section extractors pull just the marketing-list regions
    out of INTRODUCTION §1, README "Core Capabilities", and
    MANUAL §12 "API Reference" (including all sub-tables —
    Core Classes, LLM Adapters, Web-Search Backends, Pericoupling
    Functions, Visualization Functions, Quantitative Indicator
    Functions, LLM-Assisted Helpers, Enums).
  - Three test cases fail with a list of missing keywords + the
    per-feature `note` (so the failure message names exactly what
    needs to be added and where).

  Bootstrap drift fixed in the same PR:
  - **README.md "Core Capabilities"**: added an "Automated map
    generation" bullet to the Qualitative LLM analysis subsection
    (`plot_focal_country_map`, `plot_analysis_map`,
    `plot_focal_adm1_map`) — map generation was a shipped
    capability not mentioned in README.
  - **MANUAL.md §12**: split the Pericoupling Functions table
    into "country level" and a new "ADM1 (Subnational)
    Pericoupling Functions" sub-table — 7 ADM1 functions
    (`lookup_adm1_pericoupling`, `is_adm1_pericoupled`,
    `get_adm1_neighbors`, `get_cross_border_neighbors`,
    `get_adm1_codes_for_country`, `get_adm1_info`,
    `get_adm1_country`, `resolve_adm1_code`) were silently absent
    from §12 despite being in `__all__`.
  - **MANUAL.md §12 Visualization Functions table**: added
    `plot_focal_adm1_map` (was exported in `__all__` but missing
    from the table).

  Going forward: a developer who ships a new feature without
  updating the relevant capability sections gets a build break
  with a message naming exactly the missing keyword.  This turns
  "did you remember to update the marketing docs?" from a
  reviewer's checklist item into a forcing function.

  Tests added: 3 (1179 → 1182).

### Fixed

- **ADM1 resolver robustness: detect Michoacán-class regions
  through possessive, unaccented, and hyphenated surface forms
  (PR #45).**  PR #44's `COUPLING DATABASE VALIDATION` block
  silently dropped Michoacán de Ocampo (MEX016) — Mexico's
  dominant avocado state — from a Jalisco-scope live trace,
  even though the LLM mentioned it ~95 times.  Root cause:
  the resolver and the text-extraction pipeline couldn't
  recover the canonical accented DB name (`Michoacán de
  Ocampo`) from the noisy surface forms the LLM actually
  produces.  Four targeted fixes:

  **(a) Possessive stripping** in `_clean_candidate_text` /
  `_resolve_candidate` (`core.py`) — drop trailing `'s` /
  `’s` (ASCII and curly apostrophe) so `Michoacán's avocado
  belt` cleans down to a resolvable token.

  **(b) Accent-folded resolver fallback** in
  `resolve_adm1_code` (`knowledge/adm1_pericoupling.py`) — new
  Strategy 3 builds a lazily-cached NFKD-folded index and
  retries direct + substring lookup against it.  Recovers
  `Michoacan`, `Michoacan de Ocampo`, `Yucatan`, `Nuevo Leon`,
  `Sao Paulo`, and any other unaccented form of an accented
  DB region globally.  Ambiguity is contained by reusing
  `_pick_best_candidate` (single ISO or matching country
  filter required).

  **(c) Accent-aware capitalized-word-group regex** (new
  module-level `_CAPITALIZED_WORD_GROUP_RE` constant in
  `core.py`).  The previous ASCII-only `[A-Z][a-z]+` truncated
  every accented region name at the first diacritic
  (`Michoacán` → `Michoac`, `Nuevo León` → `Nuevo Le`,
  `São Paulo` → `Paulo`), which broke the country-mention
  relevance guard in `_extract_mentioned_adm1_from_text` and
  the ADM1 detector in `_user_query_mentions_adm1`.  Replaced
  with Latin-1 Supplement-aware
  `[A-ZÀ-ÖØ-Þ][a-zà-öø-ÿ]+(?:\s+[A-ZÀ-ÖØ-Þ][a-zà-öø-ÿ]+)*` at
  all four call sites.

  **(d) Extended chunk-splitter delimiters** in
  `_extract_mentioned_adm1_from_text` (`core.py`) — split also
  on `-`, `—`, and `/` so the LLM's compound forms like
  `Michoacán-Jalisco`, `Michoacán—labor`, and
  `Aguascalientes/Jalisco` each yield isolated tokens that the
  resolver can match.

  **(e) Surface LLM-mentioned ADM1 partners in the validator
  block.**  `_validate_adm1_pericoupling` already computed
  `peri_partners` / `tele_partners` from the LLM-mentioned
  ADM1 set, but only used them to drive the consistency
  `note` text — then discarded them.  The formatter buckets
  pairs from `pair_results`, so these partners never reached
  the COUPLING DATABASE VALIDATION block even when the
  resolver detected them.  Now the validator emits each
  partner as a `"focal ↔ partner: VERDICT"` line in
  `pair_results`, anchored on the focal ADM1 region (same
  convention PR #44 v3.2 established for the country
  validator).  Closes the loop: the LLM names `Michoacán`,
  the resolver finds it via A/B/C/D, and the validator now
  surfaces it as `Jalisco (MEX014) ↔ Michoacán de Ocampo
  (MEX016): PERICOUPLED` in the user-facing block.

  Tests added: 7 new resolver-fallback tests (unaccented
  Michoacán/Yucatán/Nuevo León/São Paulo, accented regression,
  country-filter ambiguity containment) + 1 extractor test
  (possessive + hyphenated + unaccented surface forms all
  produce MEX016 / MEX014) + 1 ADM1-validator emission test
  (Jalisco→Michoacán pair surfaces correctly).  Total: 1170
  → 1179.

### Added / Refactored

- **Coupling-DB surface improvements: single-country hint trigger
  + role-aware validator + clearer formatter blocks (PR #44).**
  This PR improves three related surfaces that all touch the
  coupling/pericoupling database:

  **(a) Pre-LLM hint — single-country queries now get neighbor
  context.**  Before, `_build_pericoupling_hint`
  (`prompts/builder.py`) only fired when the query mentioned ≥2
  countries (`if len(other_codes) < 2: return None`).  Single-
  country queries got no hint at all.  Now the gate is `< 1`:
  when exactly one country is named, the function enumerates
  the focal country's pericoupled neighbors (via
  `get_pericoupled_neighbors`) and injects them as REFERENCE
  context — e.g., a Mexico-only query surfaces Belize,
  Guatemala, and the United States as candidate partners the
  LLM may want to consider.  Returns None for island nations
  with zero pericoupled neighbors (Australia, etc.).  Reuses
  the same REFERENCE framing PR #43 introduced.

  **(b) Post-LLM validator — role-aware filtering.**
  Before, `_validate_country_pericoupling` (`core.py`)
  classified every receiving or spillover country at the
  country-adjacency scale, mixing framework-level spillover
  judgments into a geography-only block.  Live trace example:
  an avocado-focused Mexico query in which the LLM placed
  Chile and Peru in `parsed.systems["spillover"]` (competitor
  producers, indirect market effects) was getting them
  labeled TELECOUPLED in the validator output as if they were
  direct distant trade partners.  Now a new
  `_extract_countries_with_roles` helper walks `parsed.systems`
  per-role, and the validator **filters spillover-roled
  countries out of its block entirely** — geography alone
  can't validate the framework's TELECOUPLED-vs-SPILLOVER
  distinction (it depends on whether flows are direct or
  indirect), so the validator declines to opine on those
  pairs.  Spillover info still lives in
  `parsed.systems["spillover"]` for the framework systems
  analysis (§3–§4) and the map renderer.  Surviving pairs are
  classified PERICOUPLED (adjacent) or TELECOUPLED (distant)
  per the database.

  Role priority for the helper: sending > focal > receiving >
  spillover > adjacent.  `_extract_country_names` is
  preserved as a thin backward-compat wrapper that returns
  ISO codes via the new helper.

  **(c) Formatter — single flat block with grouped lists.**
  Before, the formatter emitted two separate validation
  blocks (`PERICOUPLING DATABASE VALIDATION — SUBNATIONAL
  (ADM1)` and `PERICOUPLING DATABASE VALIDATION —
  COUNTRY-LEVEL`).  Splitting by database scale obscured that
  a focal region's coupled partners can span both scales.
  Now: one unified `COUPLING DATABASE VALIDATION` heading
  with a flat layout —
  - `Focal System:` line (with optional `Core subnational
    regions:` sub-line for national-scope queries),
  - mode-aware grouped lists: `Pericoupled Countries:` /
    `Telecoupled Countries:` for national-scope queries;
    `Pericoupled Countries/Subnational Regions:` /
    `Telecoupled Countries/Subnational Regions:` for
    subnational-scope queries (where pairs can mix country
    and ADM1 partners),
  - single `Note:`.

  The intro paragraph also adapts to mode: national-scope
  queries get a country-only intro that explains the `Core
  subnational regions:` sub-line; subnational-scope queries
  get an intro covering both country-adjacency and
  ADM1-adjacency databases.

  Spillover-roled countries are filtered out of the validator
  block entirely (they remain in `parsed.systems["spillover"]`
  for the framework systems analysis above).  Geography alone
  can't validate framework-level spillover claims, so the
  validator declines to opine.

  Mode-aware dispatch: the validator now respects the user's
  query scope.  For national-scope queries (user named a
  country, no subnational region), the ADM1 sub-validator is
  SKIPPED to avoid surfacing the DB neighbours of an
  auto-picked focal state; the country validator instead
  populates a `core_subnational_regions` key listing the
  LLM-mentioned subnational regions inside the focal country.
  For subnational-scope queries, both validators run as
  before.  Reuses existing helpers: `_user_query_mentions_adm1`
  and `_extract_mentioned_adm1_from_text`.

  Heading rename PERICOUPLING → COUPLING because the unified
  block validates both pericoupled and telecoupled
  classifications.  Schema unchanged — both
  `pericoupling_info` and `country_pericoupling_info`
  `ParsedAnalysis` fields stay separate; the merge happens at
  render time only.

  **Region-scale adjacency in subnational mode (v3.2 Option A).**
  In subnational mode, foreign partner countries are now
  classified by the focal REGION's adjacency, not the focal
  COUNTRY's.  An interior focal state (e.g., Jalisco) borders
  no foreign country, so foreign partners are TELECOUPLED even
  though the focal COUNTRY (Mexico) borders the USA.  A border
  state (e.g., Chihuahua) keeps the adjacent foreign country
  PERICOUPLED.  Pairs are anchored on the focal region (e.g.
  `Jalisco (MEX014) ↔ United States (USA): TELECOUPLED`) so the
  verdict reads coherently under the `Focal System: Jalisco`
  header.  National mode is unchanged — pairs stay country-
  anchored and country-scale.  Uses the existing
  `get_cross_border_neighbors()` helper.

  Tests: 3 new builder tests (single-country / island-nation /
  no-country) + 3 validator tests
  (`test_validator_omits_spillover_country`,
  `test_validator_falls_back_to_geography_when_no_spillover_role`,
  `test_extract_countries_with_roles_priority`) + 2 v3 dispatch
  tests (`test_validator_skips_adm1_in_national_mode`,
  `test_validator_includes_core_subnational_regions_in_national_mode`)
  + 3 v3.2 region-scale tests
  (`test_validator_subnational_interior_region_telecouples_foreign_partner`,
  `test_validator_subnational_border_region_pericouples_foreign_partner`,
  `test_validator_national_mode_keeps_country_scale`).  Three
  pre-existing formatter tests updated for the flat v3 layout.
  Total: 1159 → 1170.

### Refactored

- **Harmonize country-level pericoupling-hint framing with the
  ADM1 hint (PR #43).**  Before: the country-level hint
  (`_build_pericoupling_hint` in `prompts/builder.py:302`) told
  the LLM the database was *"ground-truth adjacency
  relationships"*, while the ADM1 hint (modernized in PR #20)
  said *"adjacency alone is NOT evidence of pericoupling — find
  independent flow evidence"*.  Two parts of the same package
  silently disagreed on the framework's epistemology.  This PR
  rewrites the country-level hint to match the ADM1 hint's
  reference-only framing.

  Specific changes:
  - Section heading: `## PERICOUPLING DATABASE INFORMATION` →
    `## PERICOUPLING DATABASE LOOKUP (REFERENCE)`.  More
    accurate name since the function returns both pericoupled
    and telecoupled lines; "LOOKUP" is what the operation
    actually is, and "(REFERENCE)" parallels the ADM1 hint's
    "(REFERENCE ONLY)" tag.
  - Body: replaces *"Use these ground-truth adjacency
    relationships..."* with two-clause guidance covering both
    pericoupled bodies (*"database adjacency is a strong
    indicator but NOT proof — confirm with independent flow
    evidence"*) and telecoupled bodies (*"geographic distance
    is corroborating context but the classification still
    depends on whether actual cross-system flows exist"*).
  - Function docstring: rewritten to reflect that the function
    builds a country-pair CLASSIFICATION hint (pericoupled OR
    telecoupled), not just a pericoupling hint.

  Per-pair line format is unchanged (still emits "X and Y are
  **pericoupled/telecoupled**...").  Trigger conditions and
  mutual-exclusivity-with-ADM1-hint are unchanged.  No
  behavior change beyond what the LLM sees in the system
  prompt of the Stage-2 main analysis call.

  Tests updated: `test_system_prompt_uses_country_hint_when_no_adm1`
  assertion now expects "LOOKUP" instead of "INFORMATION";
  new test `test_country_hint_wording_is_reference_not_ground_truth`
  asserts the softer framing, mirroring the ADM1 framing test
  added in PR #20.

### Documentation

- **Rewrite INTRODUCTION §4.2 pre-LLM injection bullet list
  (PR #43).**  The user flagged that the *"Michigan and
  Indiana are pericoupled"* example was misleading — Michigan
  and Indiana trigger the ADM1 hint (which never says "X and
  Y are pericoupled" since PR #20), not the country hint.
  The four-bullet structure also obscured that the
  pericoupling/ADM1 bullets were mutually exclusive flavors of
  one mechanism, and trigger conditions for each injection
  weren't called out.  Rewrote as a nested-bullet list with:
  (a) corrected pericoupling-hint example using country pairs,
  (b) explicit nested structure for the two hint flavors,
  (c) trigger condition for every injection, (d) reflection of
  the new harmonized country-level framing.

- **Documentation completeness sweep (PR #42).**  A second-pass
  audit beyond PR #40 found ~15 additional issues — capability
  lists missing major feature families, architecture diagrams
  behind the times on new adapters, and a few stale values PR #40
  missed.  All fixed in one sweep.

  **Completeness fixes (capability lists were silently incomplete):**
  - INTRODUCTION §1 "Key capabilities": added entries for
    quantitative indicators, scholar export, LLM-assisted helpers,
    and `evidence_coverage_note`
  - INTRODUCTION §1 "What Users Get": same expansion
  - INTRODUCTION §3 main architecture diagram: "LLM Client" box
    now lists Gemini + Grok (was OpenAI / Anthropic / custom only)
  - INTRODUCTION §3 PR #35 indicators side-track diagram: LLM
    helpers box now lists all five helpers (was missing
    `classify_ambiguous_edges`)
  - INTRODUCTION §5 Step 3 "The output includes": added scholar
    export + `evidence_coverage_note` entries
  - MANUAL §3 "How the Package Works" diagram: LLM Provider box
    now lists all four adapters; Output box now lists scholar
    export and indicators sidecar
  - MANUAL §12 "Core Classes" table: added all four LLM adapter
    classes (OpenAIAdapter / AnthropicAdapter / GeminiAdapter /
    GrokAdapter), `LLMClient` protocol, and `RAGResult` (RAG-only
    mode) — readers consulting the "Core Classes" reference
    would otherwise miss the adapters entirely
  - MANUAL §12 "AnalysisResult exporters" expanded to
    "Properties + exporters": added `result.formatted`, `.parsed`,
    `.raw`, `.turn_number`, `.usage`, `.map` (structural properties
    used throughout examples)
  - README "Core Capabilities" Qualitative subsection: explicitly
    names Gemini and Grok as natively supported providers
    (previously only mentioned in install block and in passing)

  **Factual fixes (PR #40 missed):**
  - INTRODUCTION §4.3 + §7: `BAAI/bge-small-en-v1.5` → actual
    `BAAI/bge-base-en-v1.5` (the actual model the package ships;
    confirmed via `DEFAULT_EMBEDDING_MODEL` in `rag.py`)
  - INTRODUCTION §5 Step 7 + §4.10: Brazil-soybean example output
    `PFCI=0.36, TFCI=0.62` → actual `PFCI=0.25, TFCI=0.33` (PR #40
    fixed this in MANUAL §16 and README but missed both
    INTRODUCTION occurrences)

  **Consistency fixes (defaults drifted in examples):**
  - `web_search_max_results=5` in 4 code examples across
    INTRODUCTION / MANUAL / README → `web_search_max_results=10`
    (matches the documented default in MANUAL §14 table; the
    actual code default is 10 per PR #24)
  - `rag_top_k=10` in 5 code examples across INTRODUCTION /
    MANUAL / README → `rag_top_k=8` (matches actual code default)

  No code or behavior changes — docs only.

### Maintenance

- **Regenerate `paper_citation_counts.csv` against the rebuilt
  corpus + add reusable regeneration script (PR #41).**  The CSV
  was stale from before the corpus rebuild — 262 rows reflecting
  the pre-rebuild state with `oa_status` valued only as `OA` /
  `non-OA` (170 / 92), whereas the rebuilt corpus contains 420
  papers using a richer mixed strategy (296 OA full-text + 124
  truly closed-access papers with copyright-safe structured
  summaries; another 104 papers technically OA under gold / green
  / hybrid / bronze but using the same summary format for
  copyright safety).

  This PR regenerates the CSV from the current corpus state with
  the rule: `oa_status='non-OA'` if and only if the in-file
  header says `OA status: closed`; everything else is `OA`.  Final
  CSV: 420 rows, 296 OA / 124 non-OA, matching live in-corpus
  ground truth.

  Citation counts are preserved for the 163 papers that were
  already in the old CSV with a populated `cited_by` field; the
  remaining 257 rows have an empty `cited_by` until someone
  re-runs `scripts/check_oa_status.py` against Unpaywall.

  Added `scripts/regenerate_paper_metadata.py` (~150 LOC, one-off
  maintainer script) that documents the regeneration logic:
  preserve old metadata where possible, parse the in-corpus NOA
  header for new closed-access papers, and BibTeX-lookup for new
  OA-fulltext papers.  Future corpus rebuilds can re-run this
  script to refresh the CSV.

  No code or behavior changes; the CSV is not loaded at runtime
  by any `src/` module (it's a maintainer-facing audit artifact).

### Documentation

- **Documentation accuracy sweep (PR #40).**  A paragraph-by-
  paragraph audit of `INTRODUCTION.md` and `MANUAL.md` against
  current code/data surfaced ~15 mismatches; this PR fixes them
  all:
  - Stale paper counts: `262` (full-text) / `262` / `~297`
    (BibTeX) → actual `420` / `265`.  Live counts confirmed via
    `get_database_info()` and a `zipfile.ZipFile(Papers.zip)`
    namelist count.
  - Broken example model identifier: `model="gpt-5.2"` (does not
    exist in OpenAI's API) → `model="gpt-4o"` in 8 places across
    INTRODUCTION, MANUAL, and README.  Removed `gpt-5` from the
    "supported models" list for the same reason.
  - Misleading "embeddings + TF-IDF" notation in INTRODUCTION
    §3 architecture diagram → "embeddings or TF-IDF fallback"
    (the code does OR-with-graceful-degradation, not hybrid).
  - Stale test count: "429 tests" → "1158 tests" in INTRODUCTION
    §8.
  - `FlowCategory` enum in MANUAL §12 listed 5 values including
    a non-existent `FINANCIAL` → corrected to the actual 6 values
    (`CAPITAL`, `ENERGY`, `INFORMATION`, `MATTER`, `ORGANISMS`,
    `PEOPLE`).
  - `web_search_max_results` default documented as `5` → actual
    `10` (changed in PR #24 but never reflected in §14 table).
  - Brazil-soybean canonical example expected output: `PFCI=0.36,
    TFCI=0.62` → actual `PFCI=0.25, TFCI=0.33` (verified by
    running the example).  Fixed in both MANUAL §16 and README
    Quick Start.
  - `get_database_info()` example output: removed non-existent
    `'with_abstracts'` field; fixed `total_papers` (297→265) and
    `total_citations` (10051→7626).
  - Three stale `pip install metacoupling[X]` commands using
    the legacy package name → `metacouplingllm[X]`.
  - Two leftover `rag_mode="pre_retrieval"` references in
    `src/metacouplingllm/core.py` docstrings (lines ~1305, 1551)
    that PR #38 missed → rewritten without the deleted parameter.

  Added a "Call `get_database_info()` for live counts if the
  corpus drifts" hint to the INTRODUCTION §4.6 + §6 and MANUAL
  §9 sections so future drift is auditable.  No code or test
  changes besides the 2 docstring fixes.

### Refactored

- **Drop dead `include_citation_rules` parameter from
  `PromptBuilder.build_system_prompt` (PR #39).**  After PR #38
  removed `post_hoc` RAG mode, the single in-tree caller at
  `core.py:1485` always passed `True`.  The parameter and the
  `if include_citation_rules:` conditional in the builder are
  now gone; `CITATION_RULES_LAYER` is unconditionally injected.
  Backwards-compatible for callers that omitted the kwarg
  (default behavior was already always-on after PR #38).
  Callers that explicitly passed `include_citation_rules=True`
  must drop the kwarg or get a `TypeError`.  Three files touched:
  `prompts/builder.py` (drop param + conditional + docstring),
  `core.py` (drop kwarg from the one call site), `prompts/templates.py`
  (update the Layer 5b explanatory comment).

### Documentation

- **MANUAL §5 "How RAG citations work" expanded with concrete
  examples (PR #39).**  The post-PR-#38 single-mode paragraph
  was correct but thin.  The rewritten subsection adds: (a) a
  concrete `<retrieved_literature turn="k">` XML block showing
  what the LLM actually sees in the user message; (b) a short
  output snippet showing inline `[Tk:N]` and `[Tk:Wn]` markers
  plus the `SUPPORTING EVIDENCE` resolution block; (c) an
  explanation of what the `sanitize_turn_citations` function
  strips and why (out-of-range tokens, forward references,
  bare-legacy tokens) with the warning-log message users will
  see.  Existing intro paragraph, `refine()` merged-query
  paragraph, and turn-scoped callout preserved verbatim.

### Removed

- **Legacy `post_hoc` RAG mode and the public `annotate_citations`
  function (PR #38).**  The `post_hoc` mode generated the analysis
  blind to the corpus and then stamped `[Tk:N]` citations on by
  surface-level keyword overlap (>=3 shared tokens + >=20% overlap)
  -- a 1990s information-retrieval technique that catches
  paraphrases badly and gets fooled by topically-adjacent-but-
  irrelevant content.  The mode was internally labelled "legacy"
  in two `core.py` comments and carried a known unfixed
  limitation (post_hoc runs silently dropped RAG evidence from
  `result.to_markdown()` / `result.to_docx()` exports).
  `pre_retrieval` -- where the LLM sees the retrieved passages
  and cites them inline as it writes -- has always been the
  default and is now the only mode.

  **Breaking changes** (pre-1.0 cleanup; per the in-tree
  versioning rule):
  - `MetacouplingAssistant(..., rag_mode=...)` parameter removed.
    Callers passing `rag_mode=...` will get `TypeError:
    __init__() got an unexpected keyword argument 'rag_mode'`.
    No migration needed for callers who relied on the default --
    that path is unchanged.
  - `annotate_citations` is no longer exported from
    `metacouplingllm`.  `from metacouplingllm import
    annotate_citations` will raise `ImportError`.  The underlying
    function in `metacouplingllm/knowledge/rag.py` is also
    deleted.  Anyone relying on it can re-vendor from this PR's
    deletion diff.
  - Internal `_VALID_RAG_MODES` constant and the post-hoc-only
    `_citation_policy` helper are deleted.

  **Net diff**: ~ -1100 lines code/tests, ~ +30 lines docs.

  **Tests removed**: 8 `TestAnnotateCitations` cases in
  `test_rag.py`, 5 post_hoc-specific tests in
  `test_rag_pipeline.py` (incl. `TestPostHocBackwardCompat`),
  1 test in `test_structured_extraction.py`, and 2 parameter-
  validation tests (`test_default_rag_mode_is_pre_retrieval`,
  `test_invalid_rag_mode_raises`).  The `advisor_post_hoc`
  fixture in `conftest.py` is also removed.

### Documentation

- **Sweeping refresh of README.md, MANUAL.md, INTRODUCTION.md
  covering all 20 PRs shipped since PR #16 (PR #37).**  Until
  this PR the three user-facing doc files were frozen at roughly
  PR #16: indicator submodule, scholar export, LLM-assisted
  helpers, Anthropic / Gemini / Grok web-search backends,
  supranational handling, dual pericoupling, and the §7 Evidence
  Coverage note were all invisible to anyone reading the docs.

  - **README.md** — install extras refresh (`[indicators]`,
    `[export]`, `[anthropic]`, `[gemini]`, `[grok]`), two-track
    Quick Start (qualitative LLM analysis with scholar export +
    quantitative Brazil-soybean indicator one-liner), refreshed
    Core Capabilities pitch grouped by track, install name typo
    fix (`metacoupling` → `metacouplingllm`).
  - **MANUAL.md** — surgical edits to §1 (install extras +
    package name fix), §4 (web-search auto-wiring matrix), §7
    (`result.abstract`, `result.to_markdown`, `result.to_docx`,
    `evidence_coverage_note`), §11 (combining qualitative +
    quantitative analysis), §12 (API ref additions for
    exporters, adapters, web-search backends, indicator
    functions, LLM helpers), §13 (new ImportError entries for
    pandas / docx).  Four new sections appended: §14 Web Search
    & Web-Sourced Evidence, §15 Scholar Export, §16 Quantitative
    Indicators (with Brazil-soybean worked example), §17
    LLM-Assisted Indicator Helpers (with the end-to-end
    `define_study → check_inputs → classify → summarize →
    interpret → write_methods` workflow).  References renumbered
    to §18 and Shannon / Hirschman / Hannah-Kay / Laakso-Taagepera added.
  - **INTRODUCTION.md** — "Two complementary tracks" paragraph
    added to §1, indicators side-track diagram added to §3
    Architecture, §4.4 web-search section refreshed with native
    backend auto-wiring + evidence coverage + supranational
    handling, §4.8 LLM client abstraction refreshed with Gemini
    + Grok adapters, three new subsections §4.9 (Scholar
    Export), §4.10 (Quantitative Indicators), §4.11
    (LLM-Assisted Helpers), Step 7 added to §5 Operation
    Procedure with Brazil-soybean indicator one-liner, new §7
    "Deterministic-first for quantitative analysis" design
    principle added.

  **No code changes.**  Docs only.

- **Optional LLM-assisted helpers for the indicators submodule
  (`metacouplingllm.indicators.llm`).**  PR #35 shipped the
  deterministic indicator core; PR #36 adds five optional LLM
  helpers covering the workflow steps where natural-language
  judgment helps (study setup, data validation, ambiguous
  classification, interpretation, writing).  Each helper takes
  any existing `LLMClient` from `metacouplingllm.llm.client`
  (OpenAI / Anthropic / Gemini / Grok adapters) and returns a
  `(result, trace)` tuple per spec §15.4 reproducibility rule.

  **Five public helpers:**

  | Function | What it does |
  |---|---|
  | `define_study(description, *, llm_client)` | Natural-language description → structured study config dict (focal_system, flow_unit, intracoupling/peri/tele rules, required columns, warnings) |
  | `check_inputs(data_summary, sample_rows, *, llm_client)` | Validate user data: which indicator families can be computed, what's missing, unit / intracoupling-self-loop warnings |
  | `classify_ambiguous_edges(edges, study_config, *, llm_client)` | Classify edges deterministic rules couldn't resolve.  Returns DataFrame with `suggested_coupling_type` (`"I"` / `"P"` / `"T"` / `"unknown"`), `confidence`, `reason`, `needs_user_confirmation` |
  | `interpret_results(results, *, llm_client, audience)` | Plain-language interpretation of a computed indicator table.  Audience presets: `"academic"` / `"general"` / `"policy"` |
  | `write_methods(indicator_spec, *, llm_client)` | Manuscript-ready Methods text with formulas + standard citations (Shannon 1948, Hirschman 1945, Hannah & Kay 1977, Laakso & Taagepera 1979, Liu 2017) |

  **Option A integration with `classify_coupling()`:** the PR #35
  `classify_coupling()` gains three new kwargs (`llm_client`,
  `study_config`, `model`).  When `llm_client` is supplied AND
  the deterministic pass leaves some edges as `NaN`, the function
  automatically calls `classify_ambiguous_edges()` on just those
  rows and merges results back.  When the LLM returns `"unknown"`,
  the row stays `NaN` (per spec §16 item 3: the package never lets
  the LLM invent adjacency facts silently).  The LLM trace is
  surfaced via pandas `out.attrs["llm_classify_trace"]` so the
  function signature doesn't change.  **Backwards-compat:** omitting
  the new kwargs preserves PR #35 behaviour exactly; all 19 PR #35
  tests still pass without modification.

  **Strict structured JSON output** for the three helpers that
  return structured data (`define_study`, `check_inputs`,
  `classify_ambiguous_edges`).  Reuses the adapter-dispatch
  pattern from PR #28 (Anthropic) + PR #30 (Gemini / Grok) via a
  shared `_call_with_strict_json` private helper.  Each adapter
  gets its native strict-output mode:
  - OpenAI / Grok: `response_format = {"type": "json_schema", ...}`
  - Anthropic: `tools = [submit_tool]` + `tool_choice` (the same
    submit-tool pattern used by `extract_web_map_signals`)
  - Gemini: `response_schema` + `response_mime_type="application/json"`

  Falls back to `_extract_json_object` parsing when the strict
  path returns malformed output.  Raises `RuntimeError` only when
  both paths fail to produce a dict.

  **Reproducibility via `LLMTrace` dataclass** (per spec §15.4
  item 7): every helper returns an `LLMTrace` alongside the
  result, carrying `timestamp_utc` (ISO 8601 with Z suffix),
  `model`, `prompt_version` (e.g., `"define_study_v1"`),
  `system_prompt`, `user_prompt`, `raw_response`, and `usage`.
  Users save the trace however they like — the package doesn't
  make filesystem assumptions.

  **Guardrails per spec §15.4 baked into every prompt:**
  - LLM MUST NOT invent numerical flow values
  - LLM MUST NOT calculate final indicator values (deterministic
    code does that)
  - LLM MUST say "unknown" / surface gaps rather than guess
  - All output is structured JSON (no markdown fences, no prose
    commentary) for the three JSON-output helpers

  **No new required dependencies.**  Pandas remains optional under
  the existing `metacouplingllm[indicators]` extra from PR #35.

  **18 new tests** in `tests/test_indicators_llm.py`:
  - `TestDefineStudy` (3): structured output parsed; empty input
    rejected; unparseable response raises `RuntimeError`
  - `TestCheckInputs` (3): can-compute-all case; missing partner
    data blocks MFCI; self-loop intracoupling warning surfaced
  - `TestClassifyAmbiguousEdges` (4): high-confidence classification;
    `"unknown"` for insufficient info; multi-row batch;
    preserves input DataFrame index for downstream merge
  - `TestInterpretResults` (2): academic-audience system prompt
    selected; invalid audience raises `ValueError`
  - `TestWriteMethods` (1): output prose contains framework terms;
    `LLMTrace` returned
  - `TestClassifyCouplingIntegration` (3): `llm_client=None`
    preserves PR #35 behaviour; `llm_client=mock` resolves NaN
    edges and attaches trace via `attrs`; `"unknown"` LLM
    suggestion leaves the row as `NaN` (no silent fabrication)
  - `TestLLMTrace` (2): all fields populated; timestamp matches
    ISO 8601 Z-suffix format

  **Combined indicators suite: 37/37 pass** (19 PR #35 + 18 PR #36).

  **New private modules** factored out for testability:
  - `src/metacouplingllm/indicators/_schemas.py` — JSON schemas for
    the three structured-output helpers
  - `src/metacouplingllm/indicators/_prompts.py` — system + user
    prompt templates with `*_PROMPT_VERSION` constants so old
    `LLMTrace` records stay attributable when prompt wording
    evolves
  - `src/metacouplingllm/indicators/llm.py` — the five public
    helpers + `LLMTrace` + the two `_call_with_*` dispatch helpers

  **Out of scope** (deferred):
  - Async / batch / streaming LLM modes
  - `LLMTrace.to_json(path)` persistence helper (trivial follow-up)
  - Multi-turn refinement (e.g., "the LLM's classification looks
    wrong, ask it to reconsider")
  - CLI for the helpers
  - Vignette / sample-data integration showing the full
    `define_study → check_inputs → classify → summarize →
    interpret → write_methods` workflow end-to-end

- **Quantitative metacoupling indicators submodule
  (`metacouplingllm.indicators`).**  Three indicator families
  from the established metacoupling literature, implemented as
  deterministic Python functions (no LLM calls in the calculation
  path):

  1. **Metacoupled Flow Shares** (`compute_flow_shares`) -- IFS,
     PFS, TFS.  Relative size of intra-, peri-, and telecoupled
     flows, per Liu (2017) framework + spec §6.
  2. **Metacoupled Flow Evenness** (`compute_mfe`) -- normalised
     Shannon (1948) entropy across the three coupling-type shares.
     Uses the standard `0·ln(0) = 0` convention.  Returns 1 when
     shares are perfectly balanced, 0 when one type dominates.
  3. **Metacoupled Flow Concentration Index**
     (`compute_mfci`) -- normalised Herfindahl-Hirschman Index
     within each coupling type (Hirschman 1945; normalised per Hannah & Kay 1977), producing
     IFCI, PFCI, TFCI.  Returns 0 for perfectly distributed
     partners, 1 for single-partner concentration.

  Plus two utilities:

  - `classify_coupling(edges, focal_id, adjacency)` -- assigns
    I/P/T to each flow edge from a user-supplied adjacency table
    (no hardcoded geography per spec §4).  Self-loops classify
    as intracoupling without needing adjacency; cross-system
    edges require an adjacency DataFrame or raise `ValueError`.
  - `summarize_metacoupling()` -- one-shot combined indicator
    table per spec §12.5 with columns
    `[focal_system_id, *group_cols, F_I, F_P, F_T, F_total,
    IFS, PFS, TFS, MFE, IFCI, PFCI, TFCI, n_I, n_P, n_T,
    ENP_I, ENP_P, ENP_T]`.

  **Pandas is an OPTIONAL dependency.**  Base install
  (`pip install metacouplingllm`) stays lean (numpy + fastembed
  only).  Users opt in with
  `pip install metacouplingllm[indicators]`.  When pandas is
  missing, calling any indicator function raises an `ImportError`
  with the install command in the message.  The submodule itself
  imports cleanly without pandas thanks to lazy imports + a
  `TYPE_CHECKING` guard.

  **Design principles documented in
  `src/metacouplingllm/indicators/__init__.py`:**
  - Deterministic-first: indicator math never calls an LLM
  - Established statistics, not invented indices (Shannon entropy
    + normalised HHI sourced from Shannon 1948 / Hirschman 1945 /
    Hannah & Kay 1977; ENP per Laakso & Taagepera 1979)
  - User supplies adjacency: no hardcoded geography
  - **Intracoupling data is required**: when `F_I = 0` but other
    coupling types are non-zero, the package emits a
    `UserWarning` so users don't misread missing-data as
    "no intracoupling".  This is a deliberate guardrail per the
    PR #35 scope discussion.

  **Flow type vs coupling type are orthogonal dimensions.**  The
  Liu 2017 framework distinguishes flow TYPES (matter / capital /
  information / energy / people / organisms) from COUPLING TYPES
  (intra / peri / tele).  These are orthogonal: a flow can be
  `(matter, telecoupling)` or `(information, intracoupling)`.  The
  schema supports both: pass `group_cols=["flow_type"]` (or
  whatever you name the column) and all indicators compute
  per-(focal × flow_type) automatically via pandas groupby.  No
  special-case code; just documentation.

  **Edge cases handled per spec §8/§13:**
  - `F_total == 0`: shares = NaN + `UserWarning`
  - `F_ic == 0`: `MFCI = NaN` + `UserWarning`
  - `n_ic == 1`: `MFCI = 1` by convention + `UserWarning`
    (intracoupling self-loop case explicitly flagged via the
    spec §14.6 wording about needing internal subunits for
    meaningful intracoupling concentration)
  - Unrecognised `coupling_type` labels: dropped from totals +
    one-shot `UserWarning` listing how many edges were excluded
  - Negative or non-numeric `flow_value`: dropped + `UserWarning`

  **End-to-end Brazil-soybean example reproduced exactly** (spec
  §11): IFS=0.10, PFS=0.20, TFS=0.70, MFE≈0.730, TFCI≈0.617,
  ENP_T≈1.34.  Lives as `test_brazil_soybean_end_to_end` in the
  test suite.

  **19 new tests in `tests/test_indicators.py`** covering all
  7 worked test cases from spec §20 + the Brazil end-to-end +
  pandas-optional gating:
  - `TestFlowShares` (4): spec test 1, sum-to-one, zero-flow,
    multi-system batch
  - `TestMFE` (4): spec tests 2-4, `0·ln(0) = 0` convention
  - `TestMFCI` (4): spec tests 5-7, ENP = 1/HHI_raw companion
  - `TestClassifyCoupling` (4): self-loop intracoupling,
    adjacent → P, non-adjacent → T, missing adjacency raises
  - `TestSummarizeMetacoupling` (2): end-to-end + grouped-by-year
  - `TestPandasOptional` (1): ImportError message contains
    install hint when pandas is missing

  **Submodule layout** (`src/metacouplingllm/indicators/`):
  - `__init__.py` -- public API exports
  - `_math.py` -- shannon_entropy_normalised, normalised_hhi,
    raw_hhi (pure numpy, no pandas)
  - `core.py` -- compute_flow_shares, compute_mfe, compute_mfci,
    summarize_metacoupling
  - `classify.py` -- classify_coupling

  Access via `from metacouplingllm import indicators` then
  `indicators.compute_flow_shares(df, ...)`.  We deliberately do
  NOT re-export individual indicator functions to the top-level
  namespace -- keeps the top namespace focused on the LLM
  analysis API and signals that the quantitative side is a
  separate, opt-in part of the package.

  **PR #36 (planned)** will add five optional LLM-assisted
  helpers (`mc_llm_define_study`, `mc_llm_check_inputs`,
  `mc_llm_classify_ambiguous_edges`, `mc_llm_interpret_results`,
  `mc_llm_write_methods`) that reuse the existing
  `metacouplingllm.llm.client.LLMClient` adapters.  Guardrails
  per spec §15.4: LLM never invents numerical flow values;
  structured JSON for classifier + data-check responses;
  prompt/model/timestamp/response logged for reproducibility.

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
