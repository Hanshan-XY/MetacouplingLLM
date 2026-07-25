"""Doc-capability drift CI guard (PR #46).

Compares the marketing-style capability lists in INTRODUCTION.md §1,
README.md "Core Capabilities", and MANUAL.md §12 "API Reference"
against a curated registry of features that exist in code.  When a
feature is shipped without being advertised — the pattern PR #42
manually caught for Gemini/Grok adapters, scholar export, quantitative
indicators, evidence_coverage_note, and others — the test fails with
a list of missing keywords so the doc update becomes a build-break
instead of a silent oversight.

Adding a new feature:
    1. Add a ``MarketingFeature`` entry to ``EXPECTED_FEATURES``
       below.  ``code_check`` must return True iff the feature
       actually exists (otherwise the test treats the feature as
       not-required-in-docs).
    2. Run the test — it will tell you exactly which doc(s) need
       to mention the new keyword.
    3. Add the keyword to those doc sections.

Removing a feature: either delete the ``MarketingFeature`` entry or
let the ``code_check`` start returning False — the test will then
stop requiring the keyword.
"""
from __future__ import annotations

import importlib.util
import inspect
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

import metacouplingllm as mll

REPO_ROOT = Path(__file__).parent.parent
INTRODUCTION_MD = REPO_ROOT / "INTRODUCTION.md"
README_MD = REPO_ROOT / "README.md"
MANUAL_MD = REPO_ROOT / "MANUAL.md"


# ---------------------------------------------------------------------------
# Registry — curated subset of __init__.py __all__ that's marketing-relevant
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class MarketingFeature:
    """One feature that should be advertised in the capability lists.

    Attributes
    ----------
    keyword:
        Case-insensitive substring to look for in the doc section.
        Lenient: ``"Gemini"`` matches both ``"Google Gemini"`` (in
        INTRODUCTION marketing prose) and ``"GeminiAdapter"`` (in
        MANUAL §12 API tables).
    code_check:
        Callable returning True iff the feature actually exists in
        code.  When False, the test treats the feature as no longer
        required in docs (so removing a feature from code
        automatically drops the doc requirement, no double-update).
    docs:
        Which docs must mention the keyword.  Default: all three.
        Use a narrower set for features that legitimately only
        appear in one or two (e.g., concepts vs. APIs).
    note:
        Optional rationale shown in the failure message.
    """

    keyword: str
    code_check: Callable[[], bool]
    docs: frozenset[str] = field(
        default_factory=lambda: frozenset({"intro", "readme", "manual"})
    )
    note: str = ""


def _indicators_importable() -> bool:
    return importlib.util.find_spec("metacouplingllm.indicators") is not None


def _indicators_has_helper(name: str) -> bool:
    """Check whether the indicators.llm submodule exposes a helper."""
    if not _indicators_importable():
        return False
    spec = importlib.util.find_spec("metacouplingllm.indicators.llm")
    if spec is None:
        return False
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)  # type: ignore[union-attr]
    return hasattr(module, name)


def _source_contains(module, needle: str) -> bool:
    """Return True iff the module's source code contains `needle`."""
    try:
        return needle in inspect.getsource(module)
    except (OSError, TypeError):
        return False


def _analysis_result_has(attr: str) -> bool:
    return hasattr(mll.AnalysisResult, attr)


EXPECTED_FEATURES: list[MarketingFeature] = [
    # --- LLM provider adapters (all 4 must appear in all 3 docs) ---
    MarketingFeature(
        "OpenAI", lambda: "OpenAIAdapter" in mll.__all__,
    ),
    MarketingFeature(
        "Anthropic", lambda: "AnthropicAdapter" in mll.__all__,
    ),
    MarketingFeature(
        "Gemini", lambda: "GeminiAdapter" in mll.__all__,
    ),
    MarketingFeature(
        "Grok", lambda: "GrokAdapter" in mll.__all__,
    ),

    # --- RAG ---
    MarketingFeature(
        "RAG", lambda: "RAGEngine" in mll.__all__,
        note="Retrieval-Augmented Generation over the bundled corpus.",
    ),
    MarketingFeature(
        "RAGResult", lambda: "RAGResult" in mll.__all__,
        docs=frozenset({"manual"}),
        note="RAG-only Q&A mode container; API-doc surface only.",
    ),

    # --- Web search ---
    MarketingFeature(
        "web search", lambda: "search_web" in mll.__all__,
    ),
    MarketingFeature(
        "evidence_coverage_note",
        lambda: _source_contains(mll.llm.parser, "evidence_coverage_note"),
        docs=frozenset({"intro", "readme"}),
        note="Web-search self-assessment field on ParsedAnalysis.",
    ),

    # --- Pericoupling (country + ADM1) ---
    MarketingFeature(
        "pericoupling",
        lambda: _source_contains(mll.core, "_validate_pericoupling"),
    ),
    MarketingFeature(
        "ADM1", lambda: "resolve_adm1_code" in mll.__all__,
        note="Subnational adjacency database.",
    ),

    # --- Maps / visualization ---
    MarketingFeature(
        "map",
        lambda: "plot_focal_country_map" in mll.__all__,
        docs=frozenset({"intro", "readme"}),
        note="Map generation is a marketing feature in intro/readme; "
             "the MANUAL §12 Visualization Functions table is checked "
             "separately via the plot_* function-name keywords below.",
    ),
    MarketingFeature(
        "plot_focal_country_map",
        lambda: "plot_focal_country_map" in mll.__all__,
        docs=frozenset({"manual"}),
    ),
    MarketingFeature(
        "plot_analysis_map",
        lambda: "plot_analysis_map" in mll.__all__,
        docs=frozenset({"manual"}),
    ),
    MarketingFeature(
        "plot_focal_adm1_map",
        lambda: "plot_focal_adm1_map" in mll.__all__,
        docs=frozenset({"manual"}),
        note="PR #23-era ADM1 map function; ensure MANUAL §12 lists it "
             "alongside the country / analysis map functions.",
    ),

    # --- Supranational dissolution (PR #22-#23 feature) ---
    MarketingFeature(
        "supranational",
        lambda: _source_contains(
            mll.knowledge.countries, "expand_supranational"
        ),
        docs=frozenset({"intro", "readme"}),
        note="EU / ASEAN / USMCA member-state dissolution at map render.",
    ),

    # --- Literature ---
    MarketingFeature(
        "literature", lambda: "recommend_papers" in mll.__all__,
    ),

    # --- Scholar export (PR #31, #32) ---
    MarketingFeature(
        "scholar",
        lambda: _analysis_result_has("to_markdown")
            and _analysis_result_has("to_docx"),
        note="Markdown + Word export for manuscript-ready output.",
    ),
    MarketingFeature(
        "to_markdown",
        lambda: _analysis_result_has("to_markdown"),
        docs=frozenset({"intro", "readme", "manual"}),
    ),
    MarketingFeature(
        "to_docx",
        lambda: _analysis_result_has("to_docx"),
        docs=frozenset({"intro", "readme", "manual"}),
    ),

    # --- Quantitative indicators (PR #35) ---
    MarketingFeature(
        "quantitative indicators",
        lambda: _indicators_importable(),
    ),
    MarketingFeature(
        "classify_coupling",
        lambda: _indicators_has_helper("classify_ambiguous_edges"),
        docs=frozenset({"manual"}),
        note="Public indicators API; must be listed in MANUAL §12 "
             "Quantitative Indicator Functions table.",
    ),
    MarketingFeature(
        "compute_flow_shares",
        lambda: _indicators_importable(),
        docs=frozenset({"manual"}),
    ),
    MarketingFeature(
        "compute_mfe",
        lambda: _indicators_importable(),
        docs=frozenset({"manual"}),
    ),
    MarketingFeature(
        "compute_mfci",
        lambda: _indicators_importable(),
        docs=frozenset({"manual"}),
    ),

    # --- LLM-assisted indicator helpers (PR #36) ---
    MarketingFeature(
        "LLM-assisted",
        lambda: _indicators_has_helper("define_study"),
    ),
    MarketingFeature(
        "define_study",
        lambda: _indicators_has_helper("define_study"),
        docs=frozenset({"intro", "readme", "manual"}),
    ),
    MarketingFeature(
        "check_inputs",
        lambda: _indicators_has_helper("check_inputs"),
        docs=frozenset({"intro", "readme", "manual"}),
    ),
    MarketingFeature(
        "interpret_results",
        lambda: _indicators_has_helper("interpret_results"),
        docs=frozenset({"intro", "readme", "manual"}),
    ),
    MarketingFeature(
        "write_methods",
        lambda: _indicators_has_helper("write_methods"),
        docs=frozenset({"intro", "readme", "manual"}),
    ),
    MarketingFeature(
        "LLMTrace",
        lambda: _indicators_has_helper("LLMTrace"),
        docs=frozenset({"manual"}),
    ),
]


# ---------------------------------------------------------------------------
# Section extractors — pull just the marketing-list region out of each doc
# ---------------------------------------------------------------------------

def _extract_introduction_section1() -> str:
    """Return the bullet block under 'Key capabilities:' in INTRODUCTION.md.

    The section is bounded by the inline ``**Key capabilities:**`` label
    and the next blank-line-followed-by-bold-paragraph or heading.  We
    take from that label down to the first line starting with ``**`` or
    ``##`` after the bullets, exclusive.
    """
    text = INTRODUCTION_MD.read_text(encoding="utf-8")
    m = re.search(
        r"\*\*Key capabilities:\*\*\n+(.*?)\n\n\*\*",
        text, re.DOTALL,
    )
    if m is None:
        raise AssertionError(
            "INTRODUCTION.md is missing the '**Key capabilities:**' "
            "marker — this test depends on that section's existence."
        )
    return m.group(1)


def _extract_readme_core_capabilities() -> str:
    """Return all text under '## Core Capabilities' in README.md, up
    to the next ``## `` heading.
    """
    text = README_MD.read_text(encoding="utf-8")
    m = re.search(
        r"^## Core Capabilities\n(.*?)(?=^## )",
        text, re.DOTALL | re.MULTILINE,
    )
    if m is None:
        raise AssertionError(
            "README.md is missing the '## Core Capabilities' section."
        )
    return m.group(1)


def _extract_manual_section12() -> str:
    """Return all text under '## 12. API Reference' in MANUAL.md, up
    to the next ``## `` heading.  Spans all sub-tables (Core Classes,
    AnalysisResult exporters, LLM Adapters, Web-Search Backends,
    Pericoupling / Literature / Visualization / Indicator function
    tables, Enums).
    """
    text = MANUAL_MD.read_text(encoding="utf-8")
    m = re.search(
        r"^## 12\. API Reference\n(.*?)(?=^## )",
        text, re.DOTALL | re.MULTILINE,
    )
    if m is None:
        raise AssertionError(
            "MANUAL.md is missing the '## 12. API Reference' section."
        )
    return m.group(1)


# ---------------------------------------------------------------------------
# Shared assertion helper
# ---------------------------------------------------------------------------

def _keyword_present(keyword: str, haystack: str) -> bool:
    """Return True iff every whitespace-separated word in ``keyword``
    appears in ``haystack`` (case-insensitive, hyphens treated as
    spaces).

    This is intentionally lenient: ``"quantitative indicators"``
    matches both ``"Quantitative indicators"`` and the variant
    ``"Quantitative metacoupling indicators"`` that INTRODUCTION
    uses, and ``"web search"`` matches the hyphenated
    ``"Web-Search Backends"`` heading in MANUAL.  Stylistic
    differences shouldn't break the test; only genuine missing
    concepts should.
    """
    normalized = haystack.lower().replace("-", " ")
    return all(
        word in normalized
        for word in keyword.lower().replace("-", " ").split()
    )


def _missing_features(section_text: str, doc_tag: str) -> list[str]:
    """Return the keywords expected-but-missing from this doc."""
    missing: list[str] = []
    for feature in EXPECTED_FEATURES:
        if doc_tag not in feature.docs:
            continue
        if not feature.code_check():
            # Feature was removed from code — drop the doc requirement.
            continue
        if not _keyword_present(feature.keyword, section_text):
            label = feature.keyword
            if feature.note:
                label += f"  ({feature.note})"
            missing.append(label)
    return missing


# ---------------------------------------------------------------------------
# The three tests
# ---------------------------------------------------------------------------

def test_introduction_section1_covers_all_marketing_features():
    """PR #46: INTRODUCTION §1 'Key capabilities' must mention every
    marketing feature that exists in code."""
    section = _extract_introduction_section1()
    missing = _missing_features(section, "intro")
    assert not missing, (
        f"INTRODUCTION.md §1 'Key capabilities' is missing "
        f"{len(missing)} feature(s):\n  - "
        + "\n  - ".join(missing)
        + "\n\nEither add the keyword to the bullet list, or "
        "(if the feature was intentionally retired) remove the "
        "corresponding entry from EXPECTED_FEATURES in "
        "tests/test_docs_capabilities.py."
    )


def test_readme_core_capabilities_covers_all_marketing_features():
    """PR #46: README.md 'Core Capabilities' must mention every
    marketing feature that exists in code."""
    section = _extract_readme_core_capabilities()
    missing = _missing_features(section, "readme")
    assert not missing, (
        f"README.md 'Core Capabilities' is missing "
        f"{len(missing)} feature(s):\n  - "
        + "\n  - ".join(missing)
        + "\n\nAdd the keyword to the relevant subsection "
        "(Qualitative LLM analysis / Quantitative indicators / "
        "Optional LLM-assisted helpers), or update "
        "EXPECTED_FEATURES."
    )


def test_manual_section12_api_reference_covers_all_features():
    """PR #46: MANUAL.md §12 'API Reference' must list every public
    API symbol (classes, functions, helpers) that exists in code."""
    section = _extract_manual_section12()
    missing = _missing_features(section, "manual")
    assert not missing, (
        f"MANUAL.md §12 'API Reference' is missing "
        f"{len(missing)} symbol(s):\n  - "
        + "\n  - ".join(missing)
        + "\n\nAdd the symbol to the appropriate table "
        "(Core Classes / LLM Adapters / Web-Search Backends / "
        "Pericoupling Functions / Visualization Functions / "
        "Quantitative Indicator Functions / LLM-Assisted Helpers "
        "/ Enums), or update EXPECTED_FEATURES."
    )


# ---------------------------------------------------------------------------
# docs/METHODS_adjacency.md count-drift guard
# ---------------------------------------------------------------------------

METHODS_ADJACENCY_MD = REPO_ROOT / "docs" / "METHODS_adjacency.md"


def test_methods_adjacency_doc_counts_match_live_data():
    """docs/METHODS_adjacency.md must carry the CURRENT shipped counts.

    The methodology doc once drifted three overlay generations behind
    (shipped 8,381 / water-only 314 / moderate 8,235 / ADM0 325) because
    the doc guards only covered MANUAL.md and INTRODUCTION.md.  This test
    derives the headline counts from the live loaders and requires each
    to appear in the doc, so any future data change (a new overlay, a
    rebuild) breaks the build until the methodology doc is updated too.
    """
    from metacouplingllm.knowledge import adm1_pericoupling as adm1
    from metacouplingllm.knowledge import pericoupling as adm0

    adm1.is_adm1_pericoupled("USA044", "MEX028")  # warm the loaders
    adm0.is_pericoupled("USA", "MEX")

    base = adm1._adm1_pairs
    assert base is not None and adm1._adm1_water_nobridge is not None
    shipped_adm1 = len(base)
    moderate_adm1 = len(base - adm1._adm1_water_nobridge)
    water_adm1 = len(adm1._adm1_water_all)
    shipped_adm0 = len(adm0._pericoupled_pairs)
    water_adm0 = len(adm0._water_all_adm0)

    text = METHODS_ADJACENCY_MD.read_text(encoding="utf-8")
    expected = {
        f"{shipped_adm1:,}": "shipped ADM1 edge count",
        f"{moderate_adm1:,}": "moderate-view ADM1 edge count",
        f"{water_adm1} ADM1": "water-only ADM1 pair count",
        f"{shipped_adm0}": "shipped ADM0 pair count",
        f"{water_adm0} ADM0": "ADM0 water-only roll-up count",
    }
    missing = [
        f"{desc}: {token!r}"
        for token, desc in expected.items()
        if token not in text
    ]
    assert not missing, (
        "docs/METHODS_adjacency.md is out of date with the shipped "
        "adjacency data; these current counts are missing from the doc:"
        "\n  - " + "\n  - ".join(missing)
        + "\n\nUpdate the shipped-count footnote (section 3), the Malta "
        "paragraph (section 7), and the coupling-standard Data paragraph "
        "(section 8)."
    )


# ---------------------------------------------------------------------------
# Pinned-input SHA guard (protects `build_all.py --full` reproducibility)
# ---------------------------------------------------------------------------

BRIDGE_CSV = REPO_ROOT / "build_data" / "bridge_classified_authoritative.csv"
PROVENANCE_MD = REPO_ROOT / "src" / "metacouplingllm" / "data" / "PROVENANCE.md"
REPRODUCING_MD = REPO_ROOT / "docs" / "REPRODUCING.md"


def test_bridge_csv_sha_matches_pin():
    """The bridge CSV's hash must match build_all.py's pin AND both docs.

    ``build_all.py --full`` verifies every pinned input's SHA-256 before
    rebuilding, so a stale pin breaks full reproduction -- silently, because CI
    runs pytest only and nothing else reads PINNED_SHA256.  This drifted once
    already: the ru1 campaign re-pinned PROVENANCE.md and docs/REPRODUCING.md
    but missed scripts/build_all.py, leaving `--full` failing on main.

    The digest is LF-normalised, via the same ``build_all._sha256(text=True)``
    the verifier uses.  The repo has no ``.gitattributes``, so with
    ``core.autocrlf=true`` a Windows checkout renders this CSV CRLF while git
    stores LF -- two different raw digests for identical content.  Hashing the
    raw bytes therefore made the pin platform-dependent, and this test caught it:
    it passed on the Windows checkout the pin was computed on and failed on CI.

    Re-pin in all three places whenever the CSV changes.
    """
    spec = importlib.util.spec_from_file_location(
        "build_all_for_pin_check", REPO_ROOT / "scripts" / "build_all.py"
    )
    build_all = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(build_all)

    actual = build_all._sha256(BRIDGE_CSV, text=True)
    pinned = build_all.PINNED_SHA256["bridge_csv"]
    assert actual == pinned, (
        f"bridge_classified_authoritative.csv hashes to {actual}\n"
        f"but scripts/build_all.py pins {pinned}.\n"
        "`build_all.py --full` will abort. Re-pin the CSV in build_all.py, "
        "data/PROVENANCE.md, and docs/REPRODUCING.md."
    )

    # PROVENANCE.md carries the full 64-char hash.
    assert actual in PROVENANCE_MD.read_text(encoding="utf-8"), (
        f"data/PROVENANCE.md does not carry the current bridge-CSV hash {actual}"
    )
    # docs/REPRODUCING.md carries an abbreviated first-8...last-7 form.
    abbrev = f"{actual[:8]}…{actual[-7:]}"
    assert abbrev in REPRODUCING_MD.read_text(encoding="utf-8"), (
        f"docs/REPRODUCING.md does not carry the abbreviated hash {abbrev}"
    )


def test_bridge_csv_pin_is_line_ending_independent(tmp_path):
    """The bridge-CSV digest must not depend on the checkout's line endings.

    This is the property that actually makes the pin portable: the same content
    written CRLF and LF must hash identically, so `--full` verifies on a Windows
    autocrlf checkout and on CI/Linux alike.  Regressing ``_sha256`` to raw bytes
    fails here rather than only on whichever platform did not compute the pin.
    """
    spec = importlib.util.spec_from_file_location(
        "build_all_for_eol_check", REPO_ROOT / "scripts" / "build_all.py"
    )
    build_all = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(build_all)

    lf_bytes = BRIDGE_CSV.read_bytes().replace(b"\r\n", b"\n")
    lf, crlf = tmp_path / "lf.csv", tmp_path / "crlf.csv"
    lf.write_bytes(lf_bytes)
    crlf.write_bytes(lf_bytes.replace(b"\n", b"\r\n"))

    assert build_all._sha256(lf, text=True) == build_all._sha256(crlf, text=True)
    # ...and the pin is that shared value, i.e. the LF/canonical digest.
    assert build_all._sha256(lf, text=True) == build_all.PINNED_SHA256["bridge_csv"]
    # The GeoPackage path must stay raw-byte: normalising binary would corrupt it.
    assert build_all._sha256(crlf) != build_all._sha256(lf)
